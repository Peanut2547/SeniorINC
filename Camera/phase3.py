import argparse
from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
from threading import Event, Thread
import tempfile
import time

import cv2
import numpy as np

import Phase2 as phase2


WINDOW_NAME = "OAK-D Phase 3 - Data Recording"
DEFAULT_OUTPUT_DIR = Path("/home/peanut/dataRecording")
DEFAULT_SAMPLE_INTERVAL_S = 1.0
SEGMENT_DURATION_S = 60.0
RETENTION_DURATION_S = 30 * 60.0
MIN_FREE_BYTES = 512 * 1024 * 1024


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE)
    return cleaned.strip("_.") or "unknown"


class SessionLogger:
    def __init__(
        self,
        output_dir: Path,
        location: str,
        scenario: str,
        sample_interval_s: float,
        session_metadata: dict,
        segment_duration_s: float = SEGMENT_DURATION_S,
        retention_duration_s: float = RETENTION_DURATION_S,
    ) -> None:
        if sample_interval_s < 0:
            raise ValueError("sample_interval_s must be zero or greater")
        if segment_duration_s <= 0 or retention_duration_s <= 0:
            raise ValueError("segment and retention durations must be greater than zero")

        session_id = (
            datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            + f"_{safe_label(location)}_{safe_label(scenario)}"
        )
        self.session_id = session_id
        self.session_dir = output_dir.expanduser().resolve() / session_id
        self.frames_dir = self.session_dir / "frames"
        self.samples_dir = self.session_dir / "samples"
        self.snapshots_dir = self.session_dir / "snapshots"
        self.frames_dir.mkdir(parents=True)
        self.samples_dir.mkdir()
        self.snapshots_dir.mkdir()

        self.location = location
        self.scenario = scenario
        self.sample_interval_s = sample_interval_s
        self.segment_duration_s = segment_duration_s
        self.retention_duration_s = retention_duration_s
        self.next_sample_s = 0.0
        self.frame_count = 0
        self.snapshot_count = 0
        self.deleted_segment_count = 0
        self.deleted_media_bytes = 0
        self.media_enabled = True
        self._closed = False
        self._segment_index = 0
        self._segment_id = None
        self._frames_file = None
        self._segments = {}
        self._session_data = {
            "schema_version": 2,
            "phase": "3.2",
            "session_id": session_id,
            "started_at": now_iso(),
            "location": location,
            "scenario": scenario,
            "baseline_scope": "development_test_site",
            "requires_site_recalibration": True,
            "recording_policy": {
                "metadata": "every displayed frame",
                "sample_interval_s": sample_interval_s,
                "segment_duration_s": segment_duration_s,
                "local_media_retention_s": retention_duration_s,
                "rolling_deletion_scope": "sample media only",
                "event_snapshot": "manual S key",
                "event_snapshot_retention": "protected",
            },
            **session_metadata,
        }
        self._write_session_file()
        self._events_file = (self.session_dir / "events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._manifest_file = (self.session_dir / "media_manifest.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )

    def _write_session_file(self) -> None:
        with (self.session_dir / "session.json").open("w", encoding="utf-8") as file:
            json.dump(self._session_data, file, ensure_ascii=False, indent=2)

    def _append_manifest(self, action: str, **details) -> None:
        record = {
            "record_type": "media_lifecycle",
            "action": action,
            "host_timestamp": now_iso(),
            "session_id": self.session_id,
            **details,
        }
        self._manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _open_segment(self, uptime_s: float) -> None:
        segment_index = int(uptime_s // self.segment_duration_s) + 1
        if segment_index == self._segment_index:
            return

        if self._frames_file is not None:
            self._frames_file.close()
            previous = self._segments[self._segment_index]
            previous["range_end"] = now_iso()
            self._append_manifest(
                "SEGMENT_CLOSED",
                segment_id=previous["segment_id"],
                range_start=previous["range_start"],
                range_end=previous["range_end"],
                sample_pairs=previous["sample_pairs"],
            )

        self._segment_index = segment_index
        self._segment_id = f"segment_{segment_index:04d}"
        sample_dir = self.samples_dir / self._segment_id
        sample_dir.mkdir()
        self._frames_file = (self.frames_dir / f"{self._segment_id}.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._segments[segment_index] = {
            "segment_id": self._segment_id,
            "range_start": now_iso(),
            "range_end": None,
            "range_end_uptime_s": segment_index * self.segment_duration_s,
            "sample_pairs": 0,
        }
        self._append_manifest(
            "SEGMENT_CREATED",
            segment_id=self._segment_id,
            range_start=self._segments[segment_index]["range_start"],
            frame_log=str(
                (self.frames_dir / f"{self._segment_id}.jsonl").relative_to(
                    self.session_dir
                )
            ),
            sample_directory=str(sample_dir.relative_to(self.session_dir)),
        )
        self._delete_expired_sample_segments(uptime_s)

    def _delete_expired_sample_segments(self, uptime_s: float) -> None:
        cutoff_s = uptime_s - self.retention_duration_s
        for segment_index, segment in self._segments.items():
            if segment_index == self._segment_index or segment.get("deleted_at"):
                continue
            if segment["range_end_uptime_s"] > cutoff_s:
                continue

            segment_dir = self.samples_dir / segment["segment_id"]
            if not segment_dir.exists():
                continue
            if (
                segment_dir.is_symlink()
                or segment_dir.resolve().parent != self.samples_dir.resolve()
            ):
                raise RuntimeError(f"unsafe retention target: {segment_dir}")

            files = [path for path in segment_dir.iterdir() if path.is_file()]
            deleted_bytes = sum(path.stat().st_size for path in files)
            shutil.rmtree(segment_dir)
            deleted_at = now_iso()
            segment["deleted_at"] = deleted_at
            self.deleted_segment_count += 1
            self.deleted_media_bytes += deleted_bytes
            self._append_manifest(
                "DELETED_LOCAL",
                segment_id=segment["segment_id"],
                range_start=segment["range_start"],
                range_end=segment["range_end"],
                sample_pairs=segment["sample_pairs"],
                deleted_files=len(files),
                deleted_bytes=deleted_bytes,
                deleted_at=deleted_at,
                reason="rolling_30_minute_retention",
                cloud_status="NOT_UPLOADED",
            )

    def _frame_record(
        self,
        packet: phase2.FramePacket,
        feature: phase2.DepthFeature,
        quality: phase2.DepthQualityFeature,
        health: phase2.HealthMetrics,
        fps: float,
        uptime_s: float,
        frame_interval_ms: float,
        dropped: int,
    ) -> dict:
        return {
            "record_type": "frame",
            "host_timestamp": now_iso(),
            "session_id": self.session_id,
            "segment_id": self._segment_id,
            "seq": int(packet.seq),
            "device_timestamp_s": float(packet.timestamp),
            "uptime_s": float(uptime_s),
            "performance": {
                "display_fps": float(fps),
                "frame_interval_ms": float(frame_interval_ms),
                "dropped_frames_total": int(dropped),
                "rgb_depth_delta_ms": float(packet.sync_delta_ms),
            },
            "depth": {
                "frame_coverage": float(packet.depth_coverage),
                "roi": {
                    "median_depth_mm": feature.median_depth_mm,
                    "min_depth_mm": feature.min_depth_mm,
                    "max_depth_mm": feature.max_depth_mm,
                    "coverage": float(feature.depth_coverage),
                    "invalid_ratio": float(feature.invalid_depth_ratio),
                },
                "temporal": {
                    "median_depth_mm": quality.temporal_median_depth_mm,
                    "variance_mm2": quality.temporal_variance_mm2,
                    "mad_mm": quality.mad_mm,
                    "valid_ratio": float(quality.temporal_valid_ratio),
                    "window_samples": int(quality.window_samples),
                    "state": quality.state,
                    "reason": quality.reason,
                },
            },
            "system": {
                "cpu_percent": float(health.cpu_percent),
                "process_ram_mb": float(health.process_ram_mb),
                "temperature_c": health.temperature_c,
                "throttle": health.throttle,
            },
        }

    def _write_media(
        self,
        directory: Path,
        prefix: str,
        packet: phase2.FramePacket,
    ) -> dict | None:
        if not self.media_enabled:
            return None
        if shutil.disk_usage(self.session_dir).free < MIN_FREE_BYTES:
            print("WARNING: less than 512 MB free; media recording disabled.")
            self.media_enabled = False
            return None

        base = f"{prefix}_{packet.seq:010d}"
        rgb_path = directory / f"{base}_rgb.jpg"
        depth_path = directory / f"{base}_depth.png"
        rgb_ok = cv2.imwrite(
            str(rgb_path),
            packet.rgb,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        depth_ok = cv2.imwrite(
            str(depth_path),
            packet.depth_mm,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not rgb_ok or not depth_ok:
            raise OSError("failed to write RGB/depth media")
        return {
            "rgb": str(rgb_path.relative_to(self.session_dir)),
            "depth": str(depth_path.relative_to(self.session_dir)),
        }

    def log_frame(
        self,
        packet: phase2.FramePacket,
        feature: phase2.DepthFeature,
        quality: phase2.DepthQualityFeature,
        health: phase2.HealthMetrics,
        fps: float,
        uptime_s: float,
        frame_interval_ms: float,
        dropped: int,
    ) -> None:
        self._open_segment(uptime_s)
        record = self._frame_record(
            packet,
            feature,
            quality,
            health,
            fps,
            uptime_s,
            frame_interval_ms,
            dropped,
        )
        if self.sample_interval_s > 0 and uptime_s >= self.next_sample_s:
            segment_dir = self.samples_dir / self._segment_id
            media = self._write_media(segment_dir, "sample", packet)
            if media is not None:
                record["sample"] = {
                    "media_id": f"{self.session_id}:{self._segment_id}:{packet.seq}",
                    **media,
                }
                self._segments[self._segment_index]["sample_pairs"] += 1
            self.next_sample_s = uptime_s + self.sample_interval_s

        self._frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.frame_count += 1

    def save_manual_snapshot(
        self,
        packet: phase2.FramePacket,
        feature: phase2.DepthFeature,
        quality: phase2.DepthQualityFeature,
        health: phase2.HealthMetrics,
        fps: float,
        uptime_s: float,
        frame_interval_ms: float,
        dropped: int,
    ) -> bool:
        self._open_segment(uptime_s)
        media = self._write_media(self.snapshots_dir, "event", packet)
        if media is None:
            return False

        event = {
            "record_type": "event_snapshot",
            "event_type": "manual",
            "trigger": "S key",
            "session_id": self.session_id,
            "segment_id": self._segment_id,
            "retention_class": "PROTECTED",
            "location": self.location,
            "scenario": self.scenario,
            "files": media,
            "frame": self._frame_record(
                packet,
                feature,
                quality,
                health,
                fps,
                uptime_s,
                frame_interval_ms,
                dropped,
            ),
        }
        metadata_path = self.snapshots_dir / f"event_{packet.seq:010d}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(event, file, ensure_ascii=False, indent=2)
        event["files"]["metadata"] = str(metadata_path.relative_to(self.session_dir))
        self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._append_manifest(
            "PROTECTED_SNAPSHOT_CREATED",
            segment_id=self._segment_id,
            seq=int(packet.seq),
            retention_class="PROTECTED",
            files=event["files"],
        )
        self.snapshot_count += 1
        print(f"Event snapshot saved: {metadata_path.parent}")
        return True

    def close(self) -> None:
        if self._closed:
            return
        if self._frames_file is not None:
            self._frames_file.close()
            segment = self._segments[self._segment_index]
            segment["range_end"] = now_iso()
            self._append_manifest(
                "SEGMENT_CLOSED",
                segment_id=segment["segment_id"],
                range_start=segment["range_start"],
                range_end=segment["range_end"],
                sample_pairs=segment["sample_pairs"],
            )
        self._events_file.close()
        self._manifest_file.close()
        self._session_data.update(
            {
                "ended_at": now_iso(),
                "frame_records": self.frame_count,
                "event_snapshots": self.snapshot_count,
                "segments_created": len(self._segments),
                "sample_segments_deleted": self.deleted_segment_count,
                "sample_media_bytes_deleted": self.deleted_media_bytes,
            }
        )
        self._write_session_file()
        self._closed = True


def read_window_input() -> tuple[int, bool]:
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return key, True
    try:
        return key, cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return key, True


def run(
    duration_s: float,
    output_dir: Path,
    location: str,
    scenario: str,
    sample_interval_s: float,
    quality_window: int,
    min_depth_coverage: float,
    min_temporal_valid_ratio: float,
    max_depth_mad_mm: float,
) -> None:
    pipeline, group_queue, platform = phase2.build_pipeline()
    latest: Queue[phase2.FramePacket] = Queue(maxsize=1)
    errors: Queue[Exception] = Queue()
    stop = Event()
    capture_thread = Thread(
        target=phase2.capture_packets,
        args=(group_queue, latest, errors, stop),
        name="oakd-capture",
        daemon=True,
    )
    logger = None
    app_started = None
    fps_samples = []
    quality_counts = {"VALID": 0, "LOW_QUALITY": 0, "UNKNOWN": 0}
    packet_count = dropped = 0
    sync_total_ms = sync_max_ms = 0.0
    max_cpu = peak_ram = 0.0
    max_temperature = None
    health = phase2.HealthMetrics()
    quality_tracker = phase2.DepthQualityTracker(
        window_size=quality_window,
        min_depth_coverage=min_depth_coverage,
        min_temporal_valid_ratio=min_temporal_valid_ratio,
        max_mad_mm=max_depth_mad_mm,
    )

    try:
        pipeline.start()
        device = pipeline.getDefaultDevice()
        usb_speed = device.getUsbSpeed().name
        logger = SessionLogger(
            output_dir=output_dir,
            location=location,
            scenario=scenario,
            sample_interval_s=sample_interval_s,
            session_metadata={
                "hardware": {
                    "depthai_platform": platform.name,
                    "usb_speed": usb_speed,
                },
                "camera": {
                    "target_fps": phase2.FPS,
                    "rgb_size": list(phase2.RGB_SIZE),
                    "stereo_size": list(phase2.STEREO_SIZE),
                    "max_depth_mm": phase2.MAX_DEPTH_MM,
                    "confidence_threshold": phase2.CONFIDENCE_THRESHOLD,
                    "sync_threshold_ms": phase2.SYNC_THRESHOLD_MS,
                },
                "depth_quality": {
                    "window_frames": quality_window,
                    "min_depth_coverage": min_depth_coverage,
                    "min_temporal_valid_ratio": min_temporal_valid_ratio,
                    "max_depth_mad_mm": max_depth_mad_mm,
                },
            },
        )

        print(f"DepthAI platform: {platform.name}")
        print(f"USB speed: {usb_speed}")
        print(f"Recording session: {logger.session_dir}")
        print(f"Metadata: every frame; media sample: every {sample_interval_s:g} s")
        print("Segments: 1 minute; sample media retention: rolling 30 minutes")
        print("Frame JSON, manifest, and event snapshots are retained.")
        print("Press S to save an event snapshot; Q/ESC to stop.")
        if usb_speed not in ("SUPER", "SUPER_PLUS"):
            print("WARNING: USB 2.0 detected.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, phase2.WINDOW_WIDTH, phase2.WINDOW_HEIGHT)
        capture_thread.start()

        canvas = np.empty(
            (phase2.WINDOW_HEIGHT, phase2.WINDOW_WIDTH, 3),
            dtype=np.uint8,
        )
        depth_u16 = depth_u8 = None
        frames = 0
        fps = 0.0
        fps_started = time.monotonic()
        app_started = fps_started
        first_packet = True
        previous_seq = previous_timestamp = None
        frame_interval_ms = 0.0
        previous_cpu_times = phase2.read_cpu_times()
        health = phase2.HealthMetrics(
            process_ram_mb=phase2.read_process_ram_mb(),
            temperature_c=phase2.read_temperature_c(),
            throttle=phase2.read_throttle_status(),
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
                    break
                _, should_close = read_window_input()
                if should_close:
                    break
                continue

            if first_packet:
                if packet.rgb.shape[:2] != packet.depth_mm.shape:
                    raise RuntimeError(
                        "RGB/depth alignment size mismatch: "
                        f"{packet.rgb.shape[:2]} != {packet.depth_mm.shape}"
                    )
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
                current_cpu_times = phase2.read_cpu_times()
                if previous_cpu_times and current_cpu_times:
                    total_delta = current_cpu_times[0] - previous_cpu_times[0]
                    idle_delta = current_cpu_times[1] - previous_cpu_times[1]
                    if total_delta > 0:
                        health.cpu_percent = 100 * (total_delta - idle_delta) / total_delta
                previous_cpu_times = current_cpu_times
                health.process_ram_mb = phase2.read_process_ram_mb()
                health.temperature_c = phase2.read_temperature_c()
                if now - last_throttle_check >= 5.0:
                    health.throttle = phase2.read_throttle_status()
                    last_throttle_check = now
                max_cpu = max(max_cpu, health.cpu_percent)
                peak_ram = max(peak_ram, health.process_ram_mb)
                if health.temperature_c is not None:
                    max_temperature = max(
                        max_temperature or health.temperature_c,
                        health.temperature_c,
                    )

            uptime = now - app_started
            if duration_s and uptime >= duration_s:
                break

            half = 40
            cy = packet.depth_mm.shape[0] // 2
            cx = packet.depth_mm.shape[1] // 2
            feature = phase2.extract_depth_feature(
                packet.depth_mm[cy - half : cy + half, cx - half : cx + half]
            )
            quality = phase2.render(
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
            logger.log_frame(
                packet,
                feature,
                quality,
                health,
                fps,
                uptime,
                frame_interval_ms,
                dropped,
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key, should_close = read_window_input()
            if key in (ord("s"), ord("S")):
                logger.save_manual_snapshot(
                    packet,
                    feature,
                    quality,
                    health,
                    fps,
                    uptime,
                    frame_interval_ms,
                    dropped,
                )
            if should_close:
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
        if logger is not None:
            logger.close()

        if app_started is not None:
            runtime = time.monotonic() - app_started
            print("\n========== Run summary ==========")
            print(f"Runtime          : {runtime:.1f} s")
            print(f"Packets displayed: {packet_count}")
            print(f"Dropped frames   : {dropped}")
            if fps_samples:
                print(
                    f"Display FPS      : min {min(fps_samples):.2f} / "
                    f"avg {sum(fps_samples) / len(fps_samples):.2f} / "
                    f"max {max(fps_samples):.2f}"
                )
            if packet_count:
                print(
                    f"RGB-depth delta  : avg {sync_total_ms / packet_count:.3f} / "
                    f"max {sync_max_ms:.3f} ms"
                )
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
            if logger is not None:
                print(f"Recording saved  : {logger.session_dir}")
            print("=================================")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = SessionLogger(
            output_dir=Path(temp_dir),
            location="test_site_v1",
            scenario="normal",
            sample_interval_s=0.1,
            session_metadata={"self_test": True},
            segment_duration_s=1.0,
            retention_duration_s=2.0,
        )
        depth = np.array([[1000, 0], [1010, 990]], dtype=np.uint16)
        feature = phase2.extract_depth_feature(depth)
        tracker = phase2.DepthQualityTracker(window_size=2)
        quality = tracker.update(feature)
        health = phase2.HealthMetrics(cpu_percent=10.0, process_ram_mb=100.0)
        packets = []
        for seq, uptime in enumerate((0.1, 1.1, 2.1, 3.1), start=1):
            packet = phase2.FramePacket(
                seq=seq,
                timestamp=uptime,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth_mm=depth,
                depth_coverage=0.75,
                sync_delta_ms=0.1,
            )
            packets.append(packet)
            logger.log_frame(
                packet, feature, quality, health, 30.0, uptime, 33.3, 0
            )

        first_packet = packets[0]
        assert logger.save_manual_snapshot(
            first_packet, feature, quality, health, 30.0, 3.1, 33.3, 0
        )
        session_dir = logger.session_dir
        logger.close()

        with (session_dir / "frames" / "segment_0001.jsonl").open(
            encoding="utf-8"
        ) as file:
            frame_record = json.loads(file.readline())
        with (session_dir / "events.jsonl").open(encoding="utf-8") as file:
            event_record = json.loads(file.readline())
        with (session_dir / "media_manifest.jsonl").open(encoding="utf-8") as file:
            manifest = [json.loads(line) for line in file]
        assert frame_record["seq"] == 1
        assert frame_record["session_id"] == session_dir.name
        assert frame_record["segment_id"] == "segment_0001"
        assert event_record["event_type"] == "manual"
        assert event_record["retention_class"] == "PROTECTED"
        assert (session_dir / event_record["files"]["rgb"]).is_file()
        assert not (session_dir / "samples" / "segment_0001").exists()
        assert (session_dir / "samples" / "segment_0004").is_dir()
        deleted = [
            record for record in manifest if record["action"] == "DELETED_LOCAL"
        ]
        assert deleted and deleted[0]["segment_id"] == "segment_0001"
        assert deleted[0]["session_id"] == session_dir.name
        assert deleted[0]["cloud_status"] == "NOT_UPLOADED"
        saved_depth = cv2.imread(
            str(session_dir / event_record["files"]["depth"]),
            cv2.IMREAD_UNCHANGED,
        )
        assert saved_depth is not None and saved_depth.dtype == np.uint16
        assert np.array_equal(saved_depth, depth)
    print("Phase 3.2 self-check: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK-D Phase 3 data recording")
    parser.add_argument("--duration", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--location", default="test_site_v1")
    parser.add_argument("--scenario", default="normal")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_S,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--quality-window",
        type=int,
        default=phase2.QUALITY_WINDOW_FRAMES,
        metavar="FRAMES",
    )
    parser.add_argument(
        "--min-depth-coverage",
        type=float,
        default=phase2.MIN_DEPTH_COVERAGE,
        metavar="RATIO",
    )
    parser.add_argument(
        "--min-temporal-valid-ratio",
        type=float,
        default=phase2.MIN_TEMPORAL_VALID_RATIO,
        metavar="RATIO",
    )
    parser.add_argument(
        "--max-depth-mad-mm",
        type=float,
        default=phase2.MAX_DEPTH_MAD_MM,
        metavar="MM",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("-f", "--f", dest="kernel_file", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.duration < 0:
        parser.error("--duration must be zero or greater")
    if arguments.sample_interval < 0:
        parser.error("--sample-interval must be zero or greater")
    if arguments.quality_window < 2:
        parser.error("--quality-window must be at least 2")
    if not 0.0 <= arguments.min_depth_coverage <= 1.0:
        parser.error("--min-depth-coverage must be between 0 and 1")
    if not 0.0 <= arguments.min_temporal_valid_ratio <= 1.0:
        parser.error("--min-temporal-valid-ratio must be between 0 and 1")
    if arguments.max_depth_mad_mm < 0:
        parser.error("--max-depth-mad-mm must be zero or greater")

    if arguments.self_test:
        self_test()
    else:
        run(
            duration_s=arguments.duration,
            output_dir=arguments.output_dir,
            location=arguments.location,
            scenario=arguments.scenario,
            sample_interval_s=arguments.sample_interval,
            quality_window=arguments.quality_window,
            min_depth_coverage=arguments.min_depth_coverage,
            min_temporal_valid_ratio=arguments.min_temporal_valid_ratio,
            max_depth_mad_mm=arguments.max_depth_mad_mm,
        )
