import argparse
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
import os
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import cv2
import depthai as dai
import numpy as np


WINDOW_NAME = "OAK-D Phase 2 - Depth Quality"
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PANEL_WIDTH, PANEL_HEIGHT = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

FPS = 30
STEREO_SIZE = (640, 400)
RGB_SIZE = (PANEL_WIDTH, PANEL_HEIGHT)
MAX_DEPTH_MM = 5000
CONFIDENCE_THRESHOLD = 200  # Lower = stricter/better confidence, lower fill rate.
SYNC_THRESHOLD_MS = 1000 / (2 * FPS)
QUALITY_WINDOW_FRAMES = 30
MIN_DEPTH_COVERAGE = 0.5
MIN_TEMPORAL_VALID_RATIO = 0.8
MAX_DEPTH_MAD_MM = 100.0


@dataclass(slots=True)
class FramePacket:
    seq: int
    timestamp: float
    rgb: np.ndarray
    depth_mm: np.ndarray
    depth_coverage: float
    sync_delta_ms: float


@dataclass(slots=True, frozen=True)
class DepthFeature:
    median_depth_mm: float | None
    min_depth_mm: int | None
    max_depth_mm: int | None
    depth_coverage: float
    invalid_depth_ratio: float


@dataclass(slots=True, frozen=True)
class DepthQualityFeature:
    temporal_median_depth_mm: float | None
    temporal_variance_mm2: float | None
    mad_mm: float | None
    temporal_valid_ratio: float
    window_samples: int
    state: str
    reason: str


@dataclass(slots=True)
class HealthMetrics:
    cpu_percent: float = 0.0
    process_ram_mb: float = 0.0
    temperature_c: float | None = None
    throttle: str = "N/A"


def put_latest(queue: Queue, packet: FramePacket) -> None:
    """Keep one fresh packet; stale frames only add latency."""
    try:
        queue.put_nowait(packet)
        return
    except Full:
        pass

    try:
        queue.get_nowait()
    except Empty:
        pass
    queue.put_nowait(packet)


def extract_depth_feature(depth_roi: np.ndarray) -> DepthFeature:
    if depth_roi.size == 0:
        return DepthFeature(None, None, None, 0.0, 1.0)

    valid_mask = (depth_roi > 0) & (depth_roi <= MAX_DEPTH_MM)
    valid_depth = depth_roi[valid_mask]
    coverage = valid_depth.size / depth_roi.size

    if valid_depth.size == 0:
        return DepthFeature(None, None, None, 0.0, 1.0)

    return DepthFeature(
        median_depth_mm=float(np.median(valid_depth)),
        min_depth_mm=int(valid_depth.min()),
        max_depth_mm=int(valid_depth.max()),
        depth_coverage=coverage,
        invalid_depth_ratio=1.0 - coverage,
    )


class DepthQualityTracker:
    def __init__(
        self,
        window_size: int = QUALITY_WINDOW_FRAMES,
        min_depth_coverage: float = MIN_DEPTH_COVERAGE,
        min_temporal_valid_ratio: float = MIN_TEMPORAL_VALID_RATIO,
        max_mad_mm: float = MAX_DEPTH_MAD_MM,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 0.0 <= min_depth_coverage <= 1.0:
            raise ValueError("min_depth_coverage must be between 0 and 1")
        if not 0.0 <= min_temporal_valid_ratio <= 1.0:
            raise ValueError("min_temporal_valid_ratio must be between 0 and 1")
        if max_mad_mm < 0.0:
            raise ValueError("max_mad_mm must be zero or greater")

        self.window_size = window_size
        self.min_depth_coverage = min_depth_coverage
        self.min_temporal_valid_ratio = min_temporal_valid_ratio
        self.max_mad_mm = max_mad_mm
        self._medians: deque[float | None] = deque(maxlen=window_size)

    def update(self, feature: DepthFeature) -> DepthQualityFeature:
        temporal_value = (
            feature.median_depth_mm
            if feature.depth_coverage >= self.min_depth_coverage
            else None
        )
        self._medians.append(temporal_value)
        valid_values = np.asarray(
            [value for value in self._medians if value is not None],
            dtype=np.float64,
        )
        window_samples = len(self._medians)
        valid_ratio = valid_values.size / window_samples

        if valid_values.size:
            temporal_median = float(np.median(valid_values))
            temporal_variance = float(np.var(valid_values))
            mad = float(np.median(np.abs(valid_values - temporal_median)))
        else:
            temporal_median = temporal_variance = mad = None

        if feature.median_depth_mm is None:
            state, reason = "UNKNOWN", "NO_DATA"
        elif feature.depth_coverage < self.min_depth_coverage:
            state, reason = "LOW_QUALITY", "LOW_COVERAGE"
        elif window_samples < self.window_size:
            state, reason = "UNKNOWN", "WARMUP"
        elif valid_ratio < self.min_temporal_valid_ratio:
            state, reason = "UNKNOWN", "TEMPORAL_GAPS"
        elif mad is not None and mad > self.max_mad_mm:
            state, reason = "LOW_QUALITY", "UNSTABLE_DEPTH"
        else:
            state, reason = "VALID", "OK"

        return DepthQualityFeature(
            temporal_median_depth_mm=temporal_median,
            temporal_variance_mm2=temporal_variance,
            mad_mm=mad,
            temporal_valid_ratio=valid_ratio,
            window_samples=window_samples,
            state=state,
            reason=reason,
        )


def read_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", encoding="utf-8") as file:
            values = [int(value) for value in file.readline().split()[1:]]
        return sum(values[:8]), values[3] + values[4]
    except (OSError, ValueError, IndexError):
        return None


def read_process_ram_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="utf-8") as file:
            resident_pages = int(file.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def read_temperature_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", encoding="utf-8") as file:
            return int(file.read()) / 1000
    except (OSError, ValueError):
        return None


def read_throttle_status() -> str:
    try:
        with open(
            "/sys/devices/platform/soc/soc:firmware/get_throttled",
            encoding="utf-8",
        ) as file:
            bits = int(file.read().strip(), 0)
    except (OSError, ValueError):
        return "N/A"

    if bits == 0:
        return "OK"

    names = ((0, "UV"), (1, "FREQ"), (2, "THROT"), (3, "TEMP"))
    current = [name for bit, name in names if bits & (1 << bit)]
    past = [name for bit, name in names if bits & (1 << (bit + 16))]
    if current:
        return "NOW:" + ",".join(current)
    if past:
        return "PAST:" + ",".join(past)
    return f"0x{bits:x}"


def build_pipeline():
    pipeline = dai.Pipeline()
    platform = pipeline.getDefaultDevice().getPlatform()

    rgb_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # CROP preserves geometry. STRETCH would distort pixels used by RGB-depth alignment.
    rgb_output = rgb_camera.requestOutput(
        size=RGB_SIZE,
        # NV12 uses half the USB bandwidth of BGR888; getCvFrame converts on host.
        type=dai.ImgFrame.Type.NV12,
        fps=FPS,
        resizeMode=dai.ImgResizeMode.CROP,
        enableUndistortion=True,
    )
    left_output = left_camera.requestOutput(
        size=STEREO_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )
    right_output = right_camera.requestOutput(
        size=STEREO_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.initialConfig.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    left_output.link(stereo.left)
    right_output.link(stereo.right)

    # DepthAI v3 aligns differently on RVC2 and RVC4.
    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_output.link(align.inputAlignTo)
        aligned_depth = align.outputAligned
    else:
        rgb_output.link(stereo.inputAlignTo)
        aligned_depth = stereo.depth

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=SYNC_THRESHOLD_MS))
    sync.setSyncAttempts(-1)
    rgb_output.link(sync.inputs["rgb"])
    aligned_depth.link(sync.inputs["depth"])

    # One queued group is enough: real-time code wants newest data, not backlog.
    output_queue = sync.out.createOutputQueue(maxSize=1, blocking=False)
    return pipeline, output_queue, platform


def capture_packets(group_queue, latest: Queue, errors: Queue, stop: Event) -> None:
    try:
        while not stop.is_set():
            group = group_queue.tryGet()
            if group is None:
                time.sleep(0.001)
                continue

            rgb_message = group["rgb"]
            depth_message = group["depth"]
            depth = depth_message.getFrame()
            rgb_timestamp = rgb_message.getTimestampDevice().total_seconds()
            depth_timestamp = depth_message.getTimestampDevice().total_seconds()
            usable_depth = (depth > 0) & (depth <= MAX_DEPTH_MM)

            packet = FramePacket(
                seq=rgb_message.getSequenceNum(),
                timestamp=rgb_timestamp,
                rgb=rgb_message.getCvFrame(),
                depth_mm=depth,
                depth_coverage=np.count_nonzero(usable_depth) / depth.size,
                sync_delta_ms=abs(rgb_timestamp - depth_timestamp) * 1000,
            )
            put_latest(latest, packet)
    except Exception as exc:
        if not stop.is_set():
            errors.put(exc)
        stop.set()


def window_should_close() -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return True
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def render(
    packet: FramePacket,
    canvas: np.ndarray,
    depth_u16: np.ndarray,
    depth_u8: np.ndarray,
    fps: float,
    uptime: float,
    frame_interval_ms: float,
    dropped: int,
    usb_speed: str,
    health: HealthMetrics,
    quality_tracker: DepthQualityTracker,
) -> DepthQualityFeature:
    rgb_view = canvas[:PANEL_HEIGHT, :PANEL_WIDTH]
    depth_view = canvas[:PANEL_HEIGHT, PANEL_WIDTH:]
    packet_view = canvas[PANEL_HEIGHT:, :PANEL_WIDTH]
    system_view = canvas[PANEL_HEIGHT:, PANEL_WIDTH:]

    rgb_view[:] = packet.rgb
    packet_view[:] = (24, 24, 24)
    system_view[:] = (24, 24, 24)

    np.minimum(packet.depth_mm, MAX_DEPTH_MM, out=depth_u16)
    cv2.convertScaleAbs(
        depth_u16,
        dst=depth_u8,
        alpha=-255.0 / MAX_DEPTH_MM,
        beta=255.0,
    )
    cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET, dst=depth_view)
    depth_view[packet.depth_mm == 0] = 0

    half = 40
    cy, cx = packet.depth_mm.shape[0] // 2, packet.depth_mm.shape[1] // 2
    roi = packet.depth_mm[cy - half : cy + half, cx - half : cx + half]
    center_feature = extract_depth_feature(roi)
    quality = quality_tracker.update(center_feature)
    cv2.rectangle(rgb_view, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 1)
    cv2.rectangle(depth_view, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), 1)

    cv2.putText(rgb_view, "RGB - NV12", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_view, "DEPTH ALIGNED TO RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    packet_lines = (
        "FRAME PACKET",
        f"seq              {packet.seq}",
        f"device timestamp {packet.timestamp:.3f} s",
        f"uptime           {uptime:.1f} s",
        f"depth coverage   {packet.depth_coverage * 100:.1f} %",
        (
            f"center median    {center_feature.median_depth_mm / 1000:.3f} m"
            if center_feature.median_depth_mm is not None
            else "center median    N/A"
        ),
        (
            f"center range     {center_feature.min_depth_mm / 1000:.3f}-"
            f"{center_feature.max_depth_mm / 1000:.3f} m"
            if center_feature.min_depth_mm is not None
            else "center range     N/A"
        ),
        f"center coverage  {center_feature.depth_coverage * 100:.1f} %",
        f"center invalid   {center_feature.invalid_depth_ratio * 100:.1f} %",
        (
            f"temporal median  {quality.temporal_median_depth_mm / 1000:.3f} m"
            if quality.temporal_median_depth_mm is not None
            else "temporal median  N/A"
        ),
        (
            f"temporal var     {quality.temporal_variance_mm2:.1f} mm^2"
            if quality.temporal_variance_mm2 is not None
            else "temporal var     N/A"
        ),
        (
            f"temporal MAD     {quality.mad_mm:.1f} mm"
            if quality.mad_mm is not None
            else "temporal MAD     N/A"
        ),
        (
            f"window valid     {quality.temporal_valid_ratio * 100:.1f} % "
            f"({quality.window_samples})"
        ),
        f"quality state    {quality.state}",
        f"quality reason   {quality.reason}",
    )
    temperature = f"{health.temperature_c:.1f} C" if health.temperature_c is not None else "N/A"
    system_lines = (
        "SYSTEM",
        f"display FPS      {fps:.1f}",
        f"device interval  {frame_interval_ms:.1f} ms",
        f"RGB-depth delta  {packet.sync_delta_ms:.2f} ms",
        f"dropped frames   {dropped}",
        f"CPU usage        {health.cpu_percent:.1f} %",
        f"process RAM      {health.process_ram_mb:.1f} MB",
        f"temperature      {temperature}",
        f"throttle         {health.throttle}",
        f"USB speed        {usb_speed}",
        f"target FPS       {FPS}",
    )
    for view, lines in ((packet_view, packet_lines), (system_view, system_lines)):
        line_step = min(38, max(20, (PANEL_HEIGHT - 54) // max(1, len(lines) - 1)))
        font_scale = 0.48 if len(lines) > 11 else 0.55 if len(lines) > 8 else 0.65
        for row, text in enumerate(lines):
            cv2.putText(
                view,
                text,
                (24, 42 + row * line_step),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255) if row else (0, 200, 255),
                2 if row == 0 else 1,
                cv2.LINE_AA,
            )
    return quality


def main(
    duration_s: float = 0.0,
    quality_window: int = QUALITY_WINDOW_FRAMES,
    min_depth_coverage: float = MIN_DEPTH_COVERAGE,
    min_temporal_valid_ratio: float = MIN_TEMPORAL_VALID_RATIO,
    max_depth_mad_mm: float = MAX_DEPTH_MAD_MM,
) -> None:
    pipeline, group_queue, platform = build_pipeline()
    latest: Queue[FramePacket] = Queue(maxsize=1)
    errors: Queue[Exception] = Queue()
    stop = Event()
    capture_thread = Thread(
        target=capture_packets,
        args=(group_queue, latest, errors, stop),
        name="oakd-capture",
        daemon=True,
    )
    app_started = None
    fps_samples = []
    dropped = 0
    packet_count = 0
    sync_total_ms = 0.0
    sync_max_ms = 0.0
    max_cpu = 0.0
    peak_ram = 0.0
    max_temperature = None
    health = HealthMetrics()
    quality_tracker = DepthQualityTracker(
        window_size=quality_window,
        min_depth_coverage=min_depth_coverage,
        min_temporal_valid_ratio=min_temporal_valid_ratio,
        max_mad_mm=max_depth_mad_mm,
    )
    quality_counts = {"VALID": 0, "LOW_QUALITY": 0, "UNKNOWN": 0}

    try:
        pipeline.start()
        device = pipeline.getDefaultDevice()
        usb_speed = device.getUsbSpeed().name
        print(f"DepthAI platform: {platform.name}")
        print(f"USB speed: {usb_speed}")
        print(f"Target FPS: {FPS}; sync threshold: {SYNC_THRESHOLD_MS:.1f} ms")
        print(
            f"Depth quality: window {quality_window} frames; "
            f"coverage >= {min_depth_coverage:.0%}; "
            f"window valid >= {min_temporal_valid_ratio:.0%}; "
            f"MAD <= {max_depth_mad_mm:.0f} mm"
        )
        if usb_speed not in ("SUPER", "SUPER_PLUS"):
            print("WARNING: USB 2.0 detected; RGB is NV12 and only RGB + depth leave the device.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        capture_thread.start()

        canvas = np.empty((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        depth_u16 = depth_u8 = None
        frames = 0
        fps = 0.0
        fps_started = time.monotonic()
        app_started = fps_started
        first_packet = True
        previous_seq = None
        previous_timestamp = None
        frame_interval_ms = 0.0
        previous_cpu_times = read_cpu_times()
        health = HealthMetrics(
            process_ram_mb=read_process_ram_mb(),
            temperature_c=read_temperature_c(),
            throttle=read_throttle_status(),
        )
        last_throttle_check = app_started

        while pipeline.isRunning():
            if not errors.empty():
                raise errors.get_nowait()
            if stop.is_set():
                break

            try:
                packet = latest.get(timeout=0.02)
            except Empty:
                if duration_s and time.monotonic() - app_started >= duration_s:
                    print(f"Duration reached: {duration_s:.0f} seconds")
                    break
                if window_should_close():
                    break
                continue

            if first_packet:
                if packet.rgb.shape[:2] != packet.depth_mm.shape:
                    raise RuntimeError(
                        f"RGB/depth alignment size mismatch: {packet.rgb.shape[:2]} != {packet.depth_mm.shape}"
                    )
                print(f"RGB: {packet.rgb.shape}; aligned depth: {packet.depth_mm.shape}")
                print("Host streams: RGB + aligned depth")
                depth_u16 = np.empty_like(packet.depth_mm)
                depth_u8 = np.empty(packet.depth_mm.shape, dtype=np.uint8)
                first_packet = False

            if previous_seq is not None:
                dropped += max(0, packet.seq - previous_seq - 1)
                frame_interval_ms = (packet.timestamp - previous_timestamp) * 1000
            previous_seq = packet.seq
            previous_timestamp = packet.timestamp
            packet_count += 1
            sync_total_ms += packet.sync_delta_ms
            sync_max_ms = max(sync_max_ms, packet.sync_delta_ms)

            frames += 1
            now = time.monotonic()
            elapsed = now - fps_started
            if elapsed >= 1.0:
                fps = frames / elapsed
                fps_samples.append(fps)
                frames = 0
                fps_started = now

                current_cpu_times = read_cpu_times()
                if previous_cpu_times and current_cpu_times:
                    total_delta = current_cpu_times[0] - previous_cpu_times[0]
                    idle_delta = current_cpu_times[1] - previous_cpu_times[1]
                    if total_delta > 0:
                        health.cpu_percent = 100 * (total_delta - idle_delta) / total_delta
                previous_cpu_times = current_cpu_times
                health.process_ram_mb = read_process_ram_mb()
                health.temperature_c = read_temperature_c()
                if now - last_throttle_check >= 5.0:
                    health.throttle = read_throttle_status()
                    last_throttle_check = now

                max_cpu = max(max_cpu, health.cpu_percent)
                peak_ram = max(peak_ram, health.process_ram_mb)
                if health.temperature_c is not None:
                    max_temperature = max(max_temperature or health.temperature_c, health.temperature_c)

            uptime = now - app_started
            if duration_s and uptime >= duration_s:
                print(f"Duration reached: {duration_s:.0f} seconds")
                break

            quality = render(
                packet,
                canvas,
                depth_u16,
                depth_u8,
                fps,
                uptime,
                frame_interval_ms,
                dropped,
                usb_speed,
                health,
                quality_tracker,
            )
            quality_counts[quality.state] += 1
            cv2.imshow(WINDOW_NAME, canvas)
            if window_should_close():
                break
    finally:
        stop.set()
        if capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
        try:
            pipeline.stop()
            pipeline.wait()
        except Exception as exc:
            print(f"Pipeline shutdown skipped: {exc}")
        cv2.destroyAllWindows()

        if app_started is not None:
            runtime = time.monotonic() - app_started
            print("\n========== Run summary ==========")
            print(f"Runtime          : {runtime:.1f} s")
            print(f"Packets displayed: {packet_count}")
            print(f"Dropped frames   : {dropped}")
            if fps_samples:
                print(
                    f"Display FPS      : min {min(fps_samples):.2f} / "
                    f"avg {sum(fps_samples) / len(fps_samples):.2f} / max {max(fps_samples):.2f}"
                )
            if packet_count:
                print(f"RGB-depth delta  : avg {sync_total_ms / packet_count:.3f} / max {sync_max_ms:.3f} ms")
            print(f"Peak CPU usage   : {max_cpu:.1f} %")
            print(f"Peak process RAM : {peak_ram:.1f} MB")
            if max_temperature is not None:
                print(f"Max temperature  : {max_temperature:.1f} C")
            print(f"Throttle status  : {health.throttle}")
            if packet_count:
                print(
                    "Depth quality    : "
                    + " / ".join(
                        f"{state} {count} ({count / packet_count * 100:.1f}%)"
                        for state, count in quality_counts.items()
                    )
                )
            print("=================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK-D Phase 2 depth-quality test")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Stop automatically after this many seconds; 0 runs until Q/ESC.",
    )
    parser.add_argument(
        "--quality-window",
        type=int,
        default=QUALITY_WINDOW_FRAMES,
        metavar="FRAMES",
        help="Sliding-window length for temporal depth quality.",
    )
    parser.add_argument(
        "--min-depth-coverage",
        type=float,
        default=MIN_DEPTH_COVERAGE,
        metavar="RATIO",
        help="Minimum usable-pixel ratio before depth is low quality.",
    )
    parser.add_argument(
        "--min-temporal-valid-ratio",
        type=float,
        default=MIN_TEMPORAL_VALID_RATIO,
        metavar="RATIO",
        help="Minimum valid-frame ratio inside the temporal window.",
    )
    parser.add_argument(
        "--max-depth-mad-mm",
        type=float,
        default=MAX_DEPTH_MAD_MM,
        metavar="MM",
        help="Maximum median absolute deviation before depth is unstable.",
    )
    parser.add_argument("-f", "--f", dest="kernel_file", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.duration < 0:
        parser.error("--duration must be zero or greater")
    if arguments.quality_window < 2:
        parser.error("--quality-window must be at least 2")
    if not 0.0 <= arguments.min_depth_coverage <= 1.0:
        parser.error("--min-depth-coverage must be between 0 and 1")
    if not 0.0 <= arguments.min_temporal_valid_ratio <= 1.0:
        parser.error("--min-temporal-valid-ratio must be between 0 and 1")
    if arguments.max_depth_mad_mm < 0:
        parser.error("--max-depth-mad-mm must be zero or greater")
    main(
        arguments.duration,
        arguments.quality_window,
        arguments.min_depth_coverage,
        arguments.min_temporal_valid_ratio,
        arguments.max_depth_mad_mm,
    )
