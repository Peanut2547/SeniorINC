# ตรวจ regression โดยไม่ต้องต่อ OAK-D ทุกครั้ง ไฟล์เล็กและไม่กระทบ FPS ถ้าไม่เรียกใช้งาน
# ถ้าต้องการเก็บโปรเจกต์ให้เหลือเฉพาะโปรแกรมหลัก ลบได้ ไม่มีผลต่อ Phase2.py ครับ

import numpy as np

from Phase2 import extract_depth_feature


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
    print("DepthFeature self-check: OK")


if __name__ == "__main__":
    main()
