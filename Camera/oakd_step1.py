import argparse
from dataclasses import dataclass
from datetime import timedelta
import os
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import cv2
import depthai as dai
import numpy as np


WINDOW_NAME = "OAK-D synchronized RGB-D"
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PANEL_WIDTH, PANEL_HEIGHT = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

FPS = 30
STEREO_SIZE = (640, 400)
RGB_SIZE = (PANEL_WIDTH, PANEL_HEIGHT)
MAX_DEPTH_MM = 5000
CONFIDENCE_THRESHOLD = 200  # Lower = stricter/better confidence, lower fill rate.
SYNC_THRESHOLD_MS = 1000 / (2 * FPS)


@dataclass(slots=True)
class FramePacket:
    seq: int
    timestamp: float
    rgb: np.ndarray
    depth_mm: np.ndarray
    depth_coverage: float
    sync_delta_ms: float


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

            packet = FramePacket(
                seq=rgb_message.getSequenceNum(),
                timestamp=rgb_timestamp,
                rgb=rgb_message.getCvFrame(),
                depth_mm=depth,
                depth_coverage=np.count_nonzero(depth) / depth.size,
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
) -> None:
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
    valid_roi = roi[(roi > 0) & (roi <= MAX_DEPTH_MM)]
    center_depth = float(np.median(valid_roi)) if valid_roi.size else 0.0
    center_coverage = valid_roi.size / roi.size
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
        f"center distance  {center_depth / 1000:.3f} m" if center_depth else "center distance  N/A",
        f"center coverage  {center_coverage * 100:.1f} %",
        "coverage = valid depth / pixels",
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
        line_step = 28 if len(lines) > 8 else 38
        font_scale = 0.55 if len(lines) > 8 else 0.65
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


def main(duration_s: float = 0.0) -> None:
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

    try:
        pipeline.start()
        device = pipeline.getDefaultDevice()
        usb_speed = device.getUsbSpeed().name
        print(f"DepthAI platform: {platform.name}")
        print(f"USB speed: {usb_speed}")
        print(f"Target FPS: {FPS}; sync threshold: {SYNC_THRESHOLD_MS:.1f} ms")
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

            render(
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
            )
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
            print("=================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK-D RGB-D stability test")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Stop automatically after this many seconds; 0 runs until Q/ESC.",
    )
    parser.add_argument("-f", "--f", dest="kernel_file", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.duration < 0:
        parser.error("--duration must be zero or greater")
    main(arguments.duration)
