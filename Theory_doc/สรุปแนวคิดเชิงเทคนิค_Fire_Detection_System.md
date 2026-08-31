# สรุปแนวคิดเชิงเทคนิคทั้งหมด — ระบบตรวจจับอัคคีภัยด้วย Spatiotemporal Vision และ Sensor Fusion

เอกสารนี้สรุปการปรึกษาหารือทางเทคนิคทั้งหมดที่คุยกันมา เพื่อใช้เป็นข้อมูลอ้างอิงประกอบการแก้ไขเล่มโครงงานและการพัฒนาระบบจริงต่อไป

---

## 1. ภาพรวมฮาร์ดแวร์ที่ยืนยันแล้ว (ของจริงที่มีอยู่)

| อุปกรณ์ | บทบาท | สถานะ |
|---|---|---|
| Raspberry Pi 5 (16GB) + AI HAT+ (Hailo-8, 26 TOPS) | **Main Controller** — รัน AI model หลัก, orchestration, sensor fusion, database, dashboard | มีของจริงแล้ว |
| Luxonis OAK-D | **กล้อง RGB + Depth เท่านั้น** (ไม่รัน AI on-device) | มีของจริงแล้ว |
| MLX90640 | Thermal array sensor (32×24 px) | ยืนยันแล้วว่าเหมาะสม |
| ZE07-CO | CO sensor (เปลี่ยนจาก ZE03 เดิม) | มี temperature compensation ในตัว |
| PMS5003 | PM sensor (ให้ค่า PM1.0/PM2.5/PM10 แยกกัน) | คงเดิม |
| PC | ใช้ train/ทดสอบ deploy โมเดลก่อนลง Pi/OAK-D จริง | บทบาทเดิม |

**FPS เป้าหมาย**: เปลี่ยนจาก 5 FPS → **15-30 FPS**

---

## 2. เหตุผลเรื่องการปรับ FPS จาก 5 เป็น 15-30 (พื้นฐาน Nyquist Theorem)

- เปลวไฟแบบปั่นป่วน (turbulent flame) มีความถี่การกระพริบเฉพาะตัวประมาณ **10 Hz** โดยไม่ขึ้นกับชนิดเชื้อเพลิง
- ตามทฤษฎี Nyquist ต้องสุ่มตัวอย่างอย่างน้อย **2 เท่า** ของความถี่สัญญาณที่ต้องการจับ (20 Hz) จึงจะจับสัญญาณ 10 Hz ได้โดยไม่เกิด aliasing
- **สรุป**: ที่ 5 FPS เดิม ระบบไม่สามารถวิเคราะห์ "การกระพริบของเปลวไฟ" (ตามที่ระบุในข้อ 5.4.4 ของเล่ม) ได้อย่างถูกต้องตามหลักการเลย การขยับเป็น 15-30 FPS จึงไม่ใช่แค่ "ดีขึ้น" แต่เป็นการ**แก้จุดบกพร่องทางทฤษฎี**ของดีไซน์เดิม
- **ผลกระทบที่ต้องปรับตาม**:
  - Sliding window (5.4.1) ต้องออกแบบเป็นหน่วย "เวลา" (วินาที) ไม่ใช่หน่วย "จำนวนเฟรม" ตายตัว
  - Threshold ใน Risk Score/FSM (5.5, 5.6) ต้องแปลงจาก "ตรวจพบกี่เฟรมติดกัน" เป็น "ตรวจพบกี่วินาทีติดกัน"
  - ต้องทดสอบว่า pipeline ซอฟต์แวร์ทั้งหมด (ไม่ใช่แค่ตัว AI model) รองรับ throughput ที่สูงขึ้นได้จริง

---

## 3. วิวัฒนาการของสถาปัตยกรรม AI Model

### 3.1 เวอร์ชันแรก (ตามไดอะแกรมเดิม)
- YOLO (detection) รันบน **Raspberry Pi 5**
- DINOv3 (SSL verification) รันบน **Luxonis OAK-D**

**ปัญหาที่วิเคราะห์ได้**: การจัดวางนี้ "สลับกัน" กับที่ควรจะเป็น เพราะ:
- YOLO เป็น CNN เบา ใช้ TOPS น้อย เหมาะกับ Myriad X (OAK-D) อยู่แล้ว
- DINOv3 เป็น Vision Transformer (ViT) ที่กินโหลดต่อพารามิเตอร์สูงกว่า CNN มาก (attention คำนวณแบบ O(n²)) และ Myriad X (สถาปัตยกรรมปี 2017) ไม่รองรับ transformer ops ได้ดีเท่า Hailo-8 (สถาปัตยกรรมใหม่กว่า)

### 3.2 เวอร์ชันที่แนะนำให้สลับ
- YOLO → รันบน **OAK-D** (onboard Myriad X) — ส่งแค่ bbox/confidence ไปให้ Pi ไม่ต้องส่งภาพดิบ ลด bandwidth
- DINOv3 → รันบน **Raspberry Pi 5 + Hailo-8** — ไม่ต้อง real-time เพราะทำงานเฉพาะตอนตรวจพบ "miss prediction" เท่านั้น

### 3.3 เวอร์ชันสุดท้าย (ที่ตัดสินใจใช้จริง)
หลังพิจารณาเรื่อง scope และความเสี่ยงด้าน engineering เพิ่มเติม:
- **DINOv3 ถูกตัดออกจาก edge pipeline** (ดูรายละเอียดหัวข้อ 5)
- **OAK-D ใช้เป็นกล้อง RGB + Depth เท่านั้น ไม่รัน AI on-device เลย**
- **Raspberry Pi 5 + Hailo-8 เป็นศูนย์กลาง AI ทั้งหมด** รันโมเดลเดียว: **YOLO11n-seg หรือ YOLO11s-seg** (detection + segmentation ในโมเดลเดียวกัน)

**เหตุผลที่ segmentation ไม่ยากไปกว่า detection**: YOLOv8-seg/YOLO11-seg ไม่ใช่ 2 โมเดลต่อกัน แต่เป็นสถาปัตยกรรมเดียวที่มี 2 head แยกกัน (box+class head, และ mask-coefficient+prototype head) ทำงานใน forward pass เดียว ใช้ training pipeline เดียวกัน เปลี่ยนแค่ label จาก bbox เป็น polygon mask (ต้นทุนหลักคือเวลา label ไม่ใช่ความซับซ้อนของระบบ) — Hailo Model Zoo รองรับ YOLOv8/YOLO11-seg ทุกขนาดอย่างเป็นทางการ

---

## 4. ทางเลือกโมเดล AI ที่พิจารณา (Model Selection)

| โมเดล | สถานะการรองรับ | ข้อสรุป |
|---|---|---|
| **YOLO11n/s (-seg)** | รองรับเต็มรูปแบบทั้ง OAK-D และ Hailo-8 อย่างเป็นทางการ | **เลือกใช้เป็นหลัก** — ความเสี่ยงต่ำสุด |
| **YOLO26** | ตัวโมเดลดีกว่าจริง (NMS-free, เร็วกว่า YOLO11 ถึง 43% บน CPU, AP สูงกว่า 1.6-2.8 จุด) แต่ toolchain บน Hailo-8 ยังอยู่ในสถานะ "ยังไม่ validate อย่างเป็นทางการ" มีแค่ community workaround pipeline | ไม่ใช้เป็น pipeline หลัก แต่เก็บไว้เป็นหัวข้อทดลองเปรียบเทียบในเล่มได้ |
| **DAMO-YOLO** (Alibaba) | อยู่ใน official coverage ของ Hailo Model Zoo | ตัวเลือก "ต่างค่าย" ที่น่าสนใจที่สุดถ้าอยากเปลี่ยนจาก Ultralytics YOLO |
| **YOLOX** (Megvii, Apache 2.0) | อยู่ใน official list ของ Hailo, มีงานวิจัยใช้งานร่วมกับ Hailo จริงมาก่อน | ปลอดภัยสุดถ้าต้องการ license ต่างจาก AGPL |
| **SSD (MobileNet-SSD)** | รองรับ native เต็มรูปแบบบน OAK-D (มี decode node สำเร็จรูป) และ Hailo | ความแม่นยำต่ำกว่าโมเดลยุคใหม่ โดยเฉพาะวัตถุขนาดเล็ก |
| **RT-DETR** | Ultralytics ระบุชัดว่า**ไม่รองรับ**ผ่านเส้นทาง export ไป Hailo | ตัดออก |
| **RF-DETR** (Roboflow) | ไม่มีในรายการ Hailo ที่ยืนยันได้ ต้องรันบน host ไม่ใช่บนกล้อง | ไม่เหมาะกับสถาปัตยกรรมที่วางไว้ |

---

## 5. การตัด DINOv3 ออก และแนวคิดทดแทน

### 5.1 ปัญหาของ DINOv3 ในบริบทนี้
- ไม่ได้ช่วยลด false alarm ณ เวลาที่เกิดเหตุจริง (มันทำงาน offline/asynchronous)
- ไม่มีอยู่ใน scope เดิมของเล่ม (ข้อ 4), ไม่มีเวลาในแผนงาน (Gantt เทอม 1-2)
- DINOv3 เป็น self-supervised model ที่ให้ feature representation ไม่ใช่ label ตรงๆ — ยังต้องมี logic แยกมาตัดสินใจ label จริงอยู่ดี

### 5.2 แนวคิด/เครื่องมือทดแทน (ไม่ต้องมี foundation model หนักบน edge)
1. **Uncertainty-based Active Learning** — ใช้ confidence score จาก YOLO เอง (ไม่ต้องมีโมเดลแยก) flag เคสที่ความมั่นใจต่ำไว้ทบทวน
2. **Query-by-Committee ผ่าน sensor-vision mismatch** — การเทียบผล YOLO กับผล Sensor Fusion แล้ว flag เมื่อไม่ตรงกัน (ซึ่งมีอยู่แล้วใน diagram เดิม) คือหลักการ query-by-committee พอดี ไม่ต้องเพิ่มโมเดลใหม่
3. **FiftyOne (Voxel51, open-source)** — ใช้ดู confusion matrix, คลิกดูภาพที่ทำนายผิด, มี Active Learning Plugin (ใช้ framework modAL) integrate กับ CVAT/Label Studio สำหรับ human-in-the-loop labeling
4. **ใช้ embedding จาก backbone ของ YOLO เอง** แทนที่จะ deploy DINOv3 แยก — clustering หา hard cases โดยไม่เพิ่ม compute

### 5.3 Workflow ที่แนะนำ
```
เก็บเฟรมที่ mismatch (vision vs sensor) → prelabel ด้วยโมเดลปัจจุบัน
→ Active Learning เลือกภาพที่ informative ที่สุด → คนตรวจแก้ label (CVAT/FiftyOne)
→ รวม dataset → fine-tune YOLO11-seg ตัวเดิม → deploy → ทำซ้ำ
```

---

## 6. เทคนิคแยกแยะ "ฝุ่น" จาก "ควัน/ไฟจริง" (หัวใจสำคัญของโครงงาน)

ใช้หลักการ **Multi-modal Feature Fusion** โดยดึง feature เพิ่มเติมจาก hardware ที่มีอยู่แล้ว ไม่ต้องเพิ่ม sensor ใหม่:

### 6.1 PM Ratio Analysis (จาก PMS5003 ที่มีอยู่แล้ว)
PMS5003 ให้ค่า PM1.0, PM2.5, PM10 แยกกัน คำนวณอัตราส่วนได้:

$$DTPR = \frac{PM_{2.5}}{PM_{10}}$$

อ้างอิงจากงานวิจัยด้าน aerosol monitoring: DTPR ≤ 0.55 → **ฝุ่น (dust)**, 0.55–0.96 → **ควัน (smoke)**, > 0.96 → **หมอก (fog)**
*หลักการ*: อนุภาคฝุ่นมีขนาดใหญ่กว่า (coarse mode) มาจากกิจกรรมทางกล ส่วนอนุภาคควันมีขนาดเล็กกว่า (fine mode) มาจากการเผาไหม้/ควบแน่น

### 6.2 CO เป็น "ตัวยืนยันการเผาไหม้" ทางเคมี
CO เป็นผลผลิตเฉพาะจากการเผาไหม้ไม่สมบูรณ์ ฝุ่นไม่สร้าง CO — ใช้เป็น conditional gate ในสมการ Risk Score เช่น ถ้า PM สูงแต่ CO อยู่ baseline ปกติ → เอียงไปทาง Dust Event

### 6.3 Thermal Hotspot Correlation
ใช้ IoU (Intersection-over-Union) ระหว่าง "บริเวณจุดร้อนจาก MLX90640" กับ "บริเวณที่ vision ตรวจพบไฟ/ควัน" เป็น feature เสริม

### 6.4 Optical Flow — วิเคราะห์ลักษณะการเคลื่อนไหว (จากกล้อง RGB)
หลักการ: ไฟ/ควันมีการเคลื่อนไหวแบบปั่นป่วน (turbulent) และลอยตัวขึ้นจากแรงลอยตัว (buoyancy) ต่างจากฝุ่นที่มักถูกพัดด้วยลมในแนวราบ

เทคนิค: คำนวณ Dense Optical Flow (Horn-Schunck หรือ Lucas-Kanade) บน ROI แล้วสกัด feature:
- **Divergence ของ flow field** (ควัน/ไฟฟุ้งกระจาย → divergence เป็นบวก)
- **สัดส่วนเวกเตอร์ที่ชี้ขึ้น** (บ่งบอกการลอยตัวจากความร้อน)
- **พลังงานความปั่นป่วน** (variance ของ flow เมื่อเวลาผ่านไป — เชื่อมกับหลักการ flame flicker 10Hz ในหัวข้อ 2)

### 6.5 OAK-D Depth-based Discrimination (เทคนิคใหม่ล่าสุดที่คุยกัน)

**หลักการพื้นฐาน**: Stereo depth คำนวณจากสมการ

$$Z = \frac{f \times B}{d}$$

โดย $d$ (disparity) ต้องอาศัยการจับคู่ texture ระหว่างภาพซ้าย-ขวา พื้นผิวที่ไม่มี texture ชัดเจน (เช่น ควัน/ฝุ่นที่โปร่งแสงและเปลี่ยนรูปทรงตลอดเวลา) จะทำให้ได้ค่า depth ที่ noisy หรือคำนวณไม่ได้เลย (เกิด "หลุม" ในภาพ depth)

**หลักฐานสนับสนุน**: งานวิจัยจาก Carnegie Mellon University (FIReStereo) ยืนยันว่าสภาพแวดล้อมที่มีควัน/ฝุ่นส่งผลกระทบต่อกล้อง visual/stereo อย่างมีนัยสำคัญ ถึงขนาดต้องเปลี่ยนไปใช้กล้อง thermal แทนในงานวิจัยด้านหุ่นยนต์ที่ต้องทำงานในสภาพแวดล้อมที่มีควัน

**Feature ที่ดึงได้จริงจาก OAK-D** (ใช้ confidence map 0-255 ที่ OAK-D ให้มาต่อพิกเซล):

| Feature | สูตร | ความหมาย |
|---|---|---|
| Invalid/Low-Confidence Ratio | $R_{invalid} = \dfrac{\text{พิกเซลใน ROI ที่ confidence ต่ำกว่า threshold}}{\text{พิกเซลทั้งหมดใน ROI}}$ | ยิ่งสูง ยิ่งเอียงไปทางกลุ่มละอองโปร่งแสง |
| Mean Confidence Score | ค่าเฉลี่ย confidence ทั้งหมดใน ROI | สัญญาณต่อเนื่องแทน threshold ตายตัว |
| Temporal Depth Instability | $\sigma^2_{depth} = \text{Var}(\text{median}(Z_{ROI}))_{\text{sliding window}}$ | วัตถุแข็งทึบให้ depth เสถียร ควัน/ฝุ่นที่ปั่นป่วนให้ depth เต้นไปมา |

**การนำไปใช้**: รวมเป็นอีก evidence channel หนึ่งในสมการ Risk Score เช่น
$$P(\text{Dust/Smoke} \mid \text{Depth}) \propto R_{invalid} \times \sigma^2_{depth}$$

**ข้อควรระวัง**: งานวิจัยที่อ้างถึงยืนยันหลักการทั่วไป แต่ไม่ได้ทดสอบเฉพาะเจาะจงกับ use case นี้ — ควรทำการทดลองเก็บข้อมูลจริงเปรียบเทียบ 3 สถานการณ์ (ไฟจริง / ควัน / ฝุ่นจากพัดลม) ก่อนนำไปฝังเป็นส่วนสำคัญของระบบ

---

## 7. คณิตศาสตร์การรวมข้อมูลจากหลายแหล่ง (Sensor Fusion)

แทนที่จะใช้ weighted sum แบบง่ายๆ แนะนำ 2 แนวทางที่มีความรัดกุมทางทฤษฎีมากกว่า:

### 7.1 Bayesian Fusion
มอง sensor แต่ละตัวเป็น likelihood อิสระ $P(\text{fire}\mid \text{sensor}_i)$ รวมกันแบบ log-odds summation

### 7.2 Dempster-Shafer Evidence Theory (แนะนำ)
เหมาะกับปัญหานี้เพราะ**รองรับความไม่รู้/ความกำกวมได้ชัดเจน** — แต่ละ sensor "โหวต" มวลความเชื่อ (basic probability assignment) ให้สมมติฐานต่างๆ รวมถึงมวลของ "ไม่แน่ใจ" แล้วรวมหลักฐานด้วย Dempster's combination rule — ตรงกับสถานการณ์ที่ sensor ตัวเดียวฟันธงไม่ได้ (เช่น PM สูงอย่างเดียวบอกไม่ได้ว่าฝุ่นหรือควัน) เหมาะเป็นข้อสนับสนุนเชิงวิชาการเมื่อถูกถามเรื่อง "แบบจำลองทางคณิตศาสตร์" ตอน defend โครงงาน

---

## 8. สถาปัตยกรรมสุดท้าย (Final Architecture Summary)

```
┌─────────────────────────────────────────────────────────┐
│ Luxonis OAK-D (กล้อง RGB + Depth เท่านั้น ไม่มี AI)        │
│  - RGB stream                                             │
│  - Depth map + Confidence map (0-255 ต่อพิกเซล)           │
└───────────────────────┬─────────────────────────────────┘
                         │ ส่ง RGB + Depth ไป Pi
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Raspberry Pi 5 + AI HAT+ (Hailo-8, 26 TOPS)               │
│  1. YOLO11n/s-seg (detection + segmentation model เดียว)  │
│     → bbox, class, mask                                   │
│  2. Crop Depth map ด้วย bbox/mask → ระยะทาง + R_invalid   │
│     + σ²_depth                                             │
│  3. Optical Flow analysis บน ROI → divergence, turbulence │
│  4. รับข้อมูล Thermal (MLX90640), CO (ZE07-CO),           │
│     PM (PMS5003) จาก ESP32                                 │
│  5. คำนวณ PM Ratio (PM2.5/PM10)                            │
│  6. Sensor Fusion (Bayesian หรือ Dempster-Shafer)         │
│     → Risk Score                                            │
│  7. Finite State Machine → สถานะ/แจ้งเตือน                │
│  8. Database + Dashboard                                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ (Active Learning loop)
         Flag เคส mismatch → FiftyOne/CVAT (บน PC)
         → Human-in-the-loop label → Retrain YOLO11-seg
```

---

## 9. รายการสิ่งที่ต้องกลับไปแก้ในเล่ม (Checklist)

1. ข้อ 4 (ขอบเขตโครงงาน) — ปรับให้ตรงกับฮาร์ดแวร์จริง, ระบุ active learning loop เป็น future work
2. ข้อ 5.4/5.5/5.6 — ปรับ threshold/window เป็น time-based, เพิ่มสมการ PM ratio, Optical Flow feature, Depth-based feature, และ Dempster-Shafer fusion
3. ข้อ 5.9 (Risk table) — อัปเดต ZE03→ZE07-CO และ mitigation ที่เกี่ยวข้อง
4. ข้อ 6 (Gantt) — เพิ่มช่วงเวลาสำหรับ deploy YOLO11-seg ไป Hailo-8/OAK-D, งาน labeling mask dataset
5. ข้อ 7 (งบประมาณ) — ปรับราคาอุปกรณ์ใหม่ทั้งหมดตามของจริงที่ใช้

---

*เอกสารนี้สรุปจากการปรึกษาหารือทางเทคนิคหลายรอบ ใช้เป็นข้อมูลอ้างอิงประกอบการพัฒนาระบบและแก้ไขเล่มโครงงานเท่านั้น ควรตรวจสอบและทดลองจริงก่อนสรุปเป็นผลลัพธ์สุดท้ายในทุกหัวข้อ*
