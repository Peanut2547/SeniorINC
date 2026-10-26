# ตรวจ regression โดยไม่ต้องต่อ OAK-D ทุกครั้ง ไฟล์เล็กและไม่กระทบ FPS ถ้าไม่เรียกใช้งาน
# ถ้าต้องการเก็บโปรเจกต์ให้เหลือเฉพาะโปรแกรมหลัก ลบได้ ไม่มีผลต่อ Phase2.py ครับ

import numpy as np

from Phase2 import DepthQualityTracker, extract_depth_feature


def main() -> None:
    feature = extract_depth_feature(
        np.array([[0, 1000, 2000], [6000, 3000, 0]], dtype=np.uint16)
    )
    assert feature.median_depth_mm == 2000.0
    assert feature.min_depth_mm == 1000
    assert feature.max_depth_mm == 3000
    assert feature.depth_coverage == 0.5
    assert feature.invalid_depth_ratio == 0.5

    unknown = extract_depth_feature(np.zeros((2, 2), dtype=np.uint16))
    assert unknown.median_depth_mm is None
    assert unknown.depth_coverage == 0.0
    assert unknown.invalid_depth_ratio == 1.0

    tracker = DepthQualityTracker(window_size=3, max_mad_mm=50.0)
    first = tracker.update(
        extract_depth_feature(np.full((2, 2), 1000, dtype=np.uint16))
    )
    second = tracker.update(
        extract_depth_feature(np.full((2, 2), 1010, dtype=np.uint16))
    )
    stable = tracker.update(
        extract_depth_feature(np.full((2, 2), 990, dtype=np.uint16))
    )
    assert first.state == second.state == "UNKNOWN"
    assert first.reason == second.reason == "WARMUP"
    assert stable.state == "VALID"
    assert stable.temporal_median_depth_mm == 1000.0
    assert np.isclose(stable.temporal_variance_mm2, 200 / 3)
    assert stable.mad_mm == 10.0

    unstable_tracker = DepthQualityTracker(window_size=3, max_mad_mm=50.0)
    for depth in (700, 1000):
        unstable_tracker.update(
            extract_depth_feature(np.full((2, 2), depth, dtype=np.uint16))
        )
    unstable = unstable_tracker.update(
        extract_depth_feature(np.full((2, 2), 1300, dtype=np.uint16))
    )
    assert unstable.state == "LOW_QUALITY"
    assert unstable.reason == "UNSTABLE_DEPTH"

    low_coverage = tracker.update(
        extract_depth_feature(np.array([[1000, 0], [0, 0]], dtype=np.uint16))
    )
    assert low_coverage.state == "LOW_QUALITY"
    assert low_coverage.reason == "LOW_COVERAGE"

    no_data = tracker.update(extract_depth_feature(np.zeros((2, 2), dtype=np.uint16)))
    assert no_data.state == "UNKNOWN"
    assert no_data.reason == "NO_DATA"

    gaps_tracker = DepthQualityTracker(
        window_size=5,
        min_temporal_valid_ratio=0.8,
    )
    for depth in (0, 0, 1000, 1000):
        gaps_tracker.update(
            extract_depth_feature(np.full((2, 2), depth, dtype=np.uint16))
        )
    gaps = gaps_tracker.update(
        extract_depth_feature(np.full((2, 2), 1000, dtype=np.uint16))
    )
    assert gaps.state == "UNKNOWN"
    assert gaps.reason == "TEMPORAL_GAPS"
    print("Phase 2.2 self-check: OK")


if __name__ == "__main__":
    main()
