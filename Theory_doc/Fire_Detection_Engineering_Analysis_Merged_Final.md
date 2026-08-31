# Fire Detection System — Engineering Analysis Notebook (Merged Final Version)
## Spatiotemporal Vision and Multi-Modal Sensor Fusion for Robust Fire Detection in Dusty Environments

> เอกสารนี้เป็นเวอร์ชันรวม (merged) ระหว่างรายละเอียดเทคนิคทั้งหมดที่พัฒนาร่วมกัน กับ framing เชิงวิศวกรรมที่ดีจากการสังเคราะห์ของ ChatGPT — ใช้เป็นเอกสารอ้างอิงหลักฉบับเดียวสำหรับแก้ไขเล่มโครงงานและพัฒนาระบบจริง

---

# 1. Problem Definition and Engineering Motivation

## 1.1 ทำไมการตรวจจับไฟในสภาพแวดล้อมที่มีฝุ่นถึงยาก

สภาพแวดล้อมอุตสาหกรรมมีสภาวะทางภาพจำนวนมากที่ทำให้ระบบสับสนกับไฟได้ เช่น กลุ่มฝุ่น, ไอน้ำ, ควัน, แสงสะท้อน, การเปลี่ยนแปลงของแสง, และวัตถุที่กำลังเคลื่อนที่

ระบบที่พึ่งพา vision อย่างเดียวสามารถตรวจจับ "ความผิดปกติทางภาพ" ได้ แต่ไม่สามารถระบุได้เสมอไปว่าเหตุการณ์นั้นเกิดจากการเผาไหม้จริงหรือสิ่งรบกวนจากสภาพแวดล้อม

**คำถามพื้นฐานของโครงงานนี้จึงไม่ใช่แค่**: "ระบบตรวจจับไฟได้ไหม" **แต่เป็น**: "ระบบแยกแยะการเผาไหม้จริงออกจากสิ่งรบกวนที่ไม่ใช่ไฟได้หรือไม่" — คำถามนี้เป็นแรงจูงใจหลักที่นำไปสู่การใช้ evidence จากหลายรูปแบบ (multi-modal evidence) แทนที่จะพึ่งพา sensor เดียว

## 1.2 ปรัชญาการออกแบบระบบ (System Design Philosophy)

ระบบยึดหลัก **Evidence-Based Decision Framework** โดยมีลำดับชั้นข้อมูลดังนี้:

1. **Observation** — มีอะไรปรากฏในสภาพแวดล้อม? (จากภาพ RGB)
2. **Behavior** — วัตถุนั้นเปลี่ยนแปลงอย่างไรตามเวลา? (จาก temporal/optical flow analysis)
3. **Physical Evidence** — เหตุการณ์นั้นก่อให้เกิดความร้อน, แก๊ส, หรือลักษณะอนุภาคหรือไม่? (จาก Thermal/CO/PM)
4. **Decision** — ระบบควรแจ้งเตือนหรือไม่? (จาก Sensor Fusion + FSM)

ปรัชญานี้ช่วยลดการพึ่งพา sensor ตัวเดียวในการตัดสินใจ และเป็นกรอบคิดหลักที่ควรใช้เปิดหัวข้อ "แนวทางการทำโครงงาน" ในเล่ม

---

# 2. Overall Architecture

## 2.1 Sensor Layer

### Engineering Decision: OAK-D เป็น Sensing Hardware ไม่ใช่ AI Processor หลัก

| หัวข้อ | รายละเอียด |
|---|---|
| **Decision** | OAK-D ใช้เป็นแหล่งข้อมูล RGB + Stereo Depth + Confidence เท่านั้น ไม่รัน AI on-device |
| **Reason** | การรวม compute ไว้ที่ Raspberry Pi 5 + Hailo-8 ศูนย์กลางเดียว ทำให้ deployment และ sensor fusion ง่ายกว่า การคำนวณ depth ของ OAK-D เป็นวงจรฮาร์ดแวร์เฉพาะ (stereo matching block) แยกจาก NN accelerator อยู่แล้ว จึงไม่เสียความสามารถด้าน depth แม้ไม่รัน AI |
| **Limitation** | ต้องส่งข้อมูล RGB+Depth ผ่าน USB ไป Pi ทุกเฟรม ต้องมั่นใจว่า bandwidth เพียงพอที่ 15-30 FPS |

### Environmental Sensors

**MLX90640 (Thermal)**
- Purpose: ให้หลักฐานด้านความร้อน — ความผิดปกติทางภาพจะน่าเชื่อถือมากขึ้นเมื่อมาพร้อมความผิดปกติของอุณหภูมิ
- Limitation: ความละเอียดเชิงพื้นที่ต่ำ (32×24) ต้องตีความอย่างระมัดระวัง โดยเฉพาะเมื่อจุดความร้อนมีขนาดเล็กหรืออยู่ไกล

**ZE07-CO**
- Purpose: หลักฐานการเผาไหม้ทางเคมี — CO บ่งชี้กิจกรรมทางเคมีที่เกี่ยวข้องกับการเผาไหม้ไม่สมบูรณ์
- **สำคัญ**: ไม่ควรใช้ CO เป็น hard gate เดี่ยวๆ เพราะการตอบสนองของแก๊สช้ากว่า vision มาก
- แนะนำใช้เป็น: CO trend (แนวโน้ม), baseline deviation (ค่าเบี่ยงเบนจากค่าปกติ), rate of increase (อัตราการเพิ่มขึ้น) แทนค่าดิบเพียงจุดเดียว

**PMS5003**
- Purpose: วิเคราะห์พฤติกรรมอนุภาค — ให้ PM1.0, PM2.5, PM10 แยกกัน
- Feature ที่เป็นไปได้: PM2.5/PM10 ratio (DTPR)

> **⚠️ ข้อควรระวังสำคัญ (Calibration Requirement)**: Threshold จากงานวิจัย (เช่น DTPR ≤ 0.55 → ฝุ่น, 0.55-0.96 → ควัน ที่อ้างอิงจากงานวิจัย aerosol monitoring) **ไม่ควรนำมาใช้ตรงๆ** เพราะ sensor คนละรุ่น สภาพแวดล้อมการติดตั้งคนละแบบ และแหล่งกำเนิดฝุ่น/ควันที่แตกต่างกัน (เช่น ฝุ่นไม้กับฝุ่นซีเมนต์มีขนาดอนุภาคต่างกัน) จะให้ threshold ที่ต่างจากในงานวิจัยได้มาก **จำเป็นต้องทำการทดลอง calibrate threshold ใหม่กับ PMS5003 ตัวจริงและสภาพแวดล้อมการใช้งานจริงของโครงงาน** ก่อนนำไปใช้ในสมการ Risk Score — ข้อนี้ควรระบุไว้ชัดเจนในหัวข้อ "ข้อจำกัดของระบบ" ของเล่ม

---

# 3. Vision Intelligence Layer

## 3.1 Engineering Decision: Segmentation แทน Detection อย่างเดียว

| หัวข้อ | รายละเอียด |
|---|---|
| **Decision** | ใช้ YOLO11-seg (detection + segmentation ในโมเดลเดียว) เป็นโมเดลตั้งต้น |
| **Reason** | ไฟและควันเป็นวัตถุที่ไม่มีรูปทรงตายตัว (non-rigid) — bounding box ให้แค่ตำแหน่งและขนาดโดยประมาณ แต่ segmentation ให้ pixel-level mask, พื้นที่ (area), ขอบเขต (boundary), และวิวัฒนาการรูปทรง (shape evolution) ซึ่งจำเป็นสำหรับการวิเคราะห์เชิงเวลา YOLOv8-seg/YOLO11-seg ไม่ใช่ 2 โมเดลแยกกัน แต่เป็นสถาปัตยกรรมเดียวที่มี dual head ทำงานใน forward pass เดียว |
| **Limitation** | ต้อง label mask (polygon) แทน bbox ธรรมดา ซึ่งใช้เวลามากกว่าในขั้นเตรียม dataset |

## 3.2 ตารางเปรียบเทียบทางเลือกโมเดล (Model Selection)

| โมเดล | สถานะการรองรับจริง | สรุป |
|---|---|---|
| **YOLO11n/s (-seg)** | รองรับเต็มรูปแบบทั้ง OAK-D และ Hailo-8 อย่างเป็นทางการ | **เลือกใช้เป็นหลัก** ความเสี่ยงต่ำสุด |
| **YOLO26** | เร็วกว่า YOLO11 ถึง 43% บน CPU, AP สูงกว่า 1.6-2.8 จุด แต่ toolchain บน Hailo-8 ยังไม่ validate อย่างเป็นทางการ มีแค่ community workaround | เก็บไว้เป็นหัวข้อทดลองเปรียบเทียบ ไม่ใช้เป็น pipeline หลัก |
| **DAMO-YOLO** (Alibaba) | อยู่ใน official coverage ของ Hailo Model Zoo | ตัวเลือก "ต่างค่าย" ที่น่าสนใจที่สุด |
| **YOLOX** (Megvii, Apache 2.0) | official list ของ Hailo + มีงานวิจัยใช้ร่วมกับ Hailo จริงมาก่อน | ปลอดภัยสุดถ้าต้องการ license ต่างจาก AGPL |
| **SSD (MobileNet-SSD)** | native เต็มรูปแบบบน OAK-D | ความแม่นยำต่ำกว่าโมเดลยุคใหม่ โดยเฉพาะวัตถุขนาดเล็ก |
| **RT-DETR / RF-DETR** | ไม่รองรับผ่าน Hailo export path / ต้องรันบน host ไม่ใช่บนกล้อง | ไม่เหมาะกับสถาปัตยกรรมที่วางไว้ |

## 3.3 Object Tracking (เติมช่องว่างสำคัญ)

| หัวข้อ | รายละเอียด |
|---|---|
| **Decision** | เพิ่ม Multi-Object Tracking (SORT/ByteTrack) ต่อจาก YOLO11-seg |
| **Reason** | Detection เพียงอย่างเดียวไม่มีกลไกจับคู่ "วัตถุเดียวกัน" ข้ามเฟรม ทำให้การวัดการขยายตัวของพื้นที่ (Area Evolution, หัวข้อ 4.3) ไม่น่าเชื่อถือถ้าไม่รู้ว่ากำลังเทียบพื้นที่ของจุดเดียวกัน ByteTrack/SORT ใช้ Kalman Filter ทำนายตำแหน่งถัดไปร่วมกับ IoU matching เพื่อผูก track ID ให้แต่ละวัตถุ มีงานวิจัยเฉพาะทางไฟป่า (AO-OCSORT: YOLOX + OC-SORT พร้อม similarity metric ที่ผสม optical flow และลักษณะทางกายภาพของเปลวไฟ) ยืนยันว่าแนวทางนี้ใช้ได้จริงในโดเมนนี้ |
| **Limitation** | เพิ่มความซับซ้อนของ pipeline อีก 1 ขั้น ต้องทดสอบว่า Kalman Filter tuning เหมาะกับความเร็วการเคลื่อนที่ของควัน/ไฟหรือไม่ |

---

# 4. Spatiotemporal Analysis

## 4.1 แรงจูงใจ

การตรวจจับแบบเฟรมเดียวไม่สามารถเข้าใจพฤติกรรมได้ — ตัวอย่างเช่น กลุ่มฝุ่นกับกลุ่มควันอาจมีลักษณะทางภาพคล้ายกันมากในเฟรมเดียว แต่พฤติกรรมเชิงเวลาต่างกัน ระบบจึงวิเคราะห์ 2 มิติคู่กัน:

- **Spatial**: รูปทรง (shape), พื้นที่ (area), ตำแหน่ง (position)
- **Temporal**: ความต่อเนื่อง (persistence), การเคลื่อนไหว (motion), วิวัฒนาการ (evolution)

## 4.2 Persistence Analysis

ประเมินว่าเหตุการณ์ถูกตรวจพบต่อเนื่องนานแค่ไหน:

$$P_{persistence} = \frac{\text{จำนวนเฟรมที่ตรวจพบ}}{\text{จำนวนเฟรมทั้งหมดในหน้าต่างเวลา}}$$

ใช้ลดการแจ้งเตือนผิดพลาดจากความผิดปกติทางภาพชั่วคราว (เช่น แสงสะท้อนแวบเดียว)

## 4.3 Area Evolution (ต้องใช้ Track ID จากหัวข้อ 3.3)

ใช้ segmentation mask หาพื้นที่ที่ตรวจพบ $A(t)$ แล้วคำนวณอัตราการเปลี่ยนแปลง:

$$r_A = \frac{A(t) - A(t-1)}{A(t-1)}$$

ใช้สำหรับ: การขยายตัวของควัน, ความผันผวนของเปลวไฟ, การเติบโตที่ผิดปกติ

## 4.4 Motion Analysis

ติดตามจุดศูนย์กลาง (centroid) ของวัตถุที่ track ได้: $C(t) = (x,y)$

$$\Delta C = C(t) - C(t-1)$$

ใช้สำหรับ: การลอยตัวของควัน, การเคลื่อนที่ตามกระแสอากาศ, การเคลื่อนที่ของวัตถุ

> **หมายเหตุ**: เพื่อให้ 4.3 และ 4.4 คำนวณได้ถูกต้อง จำเป็นต้องมี track ID ที่เสถียรจากหัวข้อ 3.3 ก่อน — นี่คือเหตุผลที่ Object Tracking ต้องมาก่อนขั้นตอนนี้ใน pipeline

---

# 5. Optical Flow Analysis

## 5.1 หลักการพื้นฐาน

Optical flow ประมาณการเคลื่อนที่ของพิกเซลระหว่างเฟรมที่ต่อเนื่องกัน บนสมมติฐาน **brightness constancy**:

$$I(x,y,t) = I(x+\Delta x, y+\Delta y, t+\Delta t)$$

ขยายด้วย Taylor series ได้ **Optical Flow Constraint Equation**:

$$I_x u + I_y v + I_t = 0$$

โดย $u, v$ คือ image velocity, $I_x, I_y$ คือ spatial gradient, $I_t$ คือการเปลี่ยนแปลงเชิงเวลา — สมการนี้มี 1 สมการ 2 ตัวแปรไม่ทราบค่า เรียกว่า **Aperture Problem** ต้องมีเงื่อนไขเพิ่มเติมจึงจะแก้ได้

## 5.2 วิธีที่พิจารณา

**Lucas-Kanade** (Local, Sparse)
- ข้อดี: เร็ว, simple
- ข้อจำกัด: ติดตามเฉพาะจุดที่เลือก (feature points) เท่านั้น, ต้องการ texture ในหน้าต่างท้องถิ่น (เชื่อมโยงกับปัญหาเดียวกันที่ทำให้ stereo depth มีปัญหากับควัน/ฝุ่นไร้ texture)
- คำนวณผ่าน least squares: $[u,v]^T = (A^TA)^{-1}A^Tb$

**Horn-Schunck** (Global, Dense)
- ข้อดี: ให้ flow ครอบคลุมทั้งภาพ
- ข้อจำกัด: ต้นทุนการคำนวณสูงกว่า
- คำนวณผ่านการหาค่าต่ำสุดของ energy functional: $E(u,v) = \iint[(I_xu+I_yv+I_t)^2 + \alpha^2(|\nabla u|^2+|\nabla v|^2)]dxdy$

**Farneback (เลือกใช้เป็นวิธีเริ่มต้น)**
- เหตุผล: ให้ dense optical flow, มี implementation สำเร็จรูปใน OpenCV (`cv2.calcOpticalFlowFarneback`), เหมาะกับการประมวลผลเฉพาะ ROI บน embedded device
- ใช้ polynomial expansion ระดับ 2 ประมาณค่าความสว่างในหน้าต่างท้องถิ่น แล้วเปรียบเทียบสัมประสิทธิ์ระหว่างเฟรม

**CLG Method (Combined Local-Global, ขั้นสูง)**
- งานของ Bruhn & Weickert รวมจุดแข็งของ LK (ทนต่อ noise) กับ HS (dense, เติมเต็มพื้นที่ไร้ texture) — เหมาะเป็นหัวข้อศึกษาต่อถ้า Farneback ไม่พอ

## 5.3 Feature ที่สกัดได้ (พร้อมสูตร)

กำหนด $V(x,y)=(u(x,y),v(x,y))$ เป็น flow field ใน ROI:

| Feature | สูตร | ความหมาย |
|---|---|---|
| **Motion Magnitude** | $\|V\| = \sqrt{u^2+v^2}$ | ปริมาณการเคลื่อนไหว |
| **Direction Distribution** | Circular histogram ของ $\theta = \text{atan2}(v,u)$ | แนวโน้มทิศทางการเคลื่อนที่ |
| **Circular Mean** | $\bar\theta = \text{atan2}(\sum\sin\theta_i, \sum\cos\theta_i)$ | ทิศทางเฉลี่ย (ใช้สถิติเชิงวงกลม เพราะมุมไม่ใช่ข้อมูลเชิงเส้น) |
| **Mean Resultant Length** | $R=\frac{1}{n}\sqrt{(\sum\cos\theta_i)^2+(\sum\sin\theta_i)^2}$ | ความเกาะกลุ่มของทิศทาง — $R$ สูง = พุ่งทิศเดียวชัดเจน (บ่งชี้การลอยตัวจากความร้อน) |
| **Divergence** | $\nabla\cdot V = \partial u/\partial x + \partial v/\partial y$ | พฤติกรรมขยาย/หดตัว |
| **Curl (Vorticity)** | $\nabla\times V = \partial v/\partial x - \partial u/\partial y$ | การหมุนวนที่ขอบ plume ตามหลักพลศาสตร์ของ buoyant plume |
| **Temporal Variance / Turbulence Energy** | $\sigma^2_{flow}=\text{Var}(\bar{\|V\|}(t))$ + PSD ผ่าน FFT | ความไม่เสถียรของการเคลื่อนไหว เชื่อมกับความถี่ flame flicker ~10Hz |

## 5.4 ข้อจำกัดสำคัญ

Optical flow แบบดั้งเดิมทุกวิธี (LK, HS, Farneback) ตั้งอยู่บนสมมติฐาน brightness constancy — **เปลวไฟละเมิดสมมติฐานนี้โดยตรง** เพราะเป็นวัตถุเรืองแสงในตัวเอง (self-luminous) ความสว่างภายในเปลี่ยนแปลงเองไม่ได้มาจากการเคลื่อนที่เพียงอย่างเดียว งานวิจัยเฉพาะทาง (Optical Flow Estimation for Flame Detection) เสนอวิธี Optimal Mass Transport (OMT) แก้ปัญหานี้โดยเฉพาะ

**ข้อสรุปเชิงปฏิบัติ**: Optical Flow ควรถือเป็น **supporting evidence** โดยเฉพาะสำหรับพฤติกรรมของควันและฝุ่น (ซึ่งเป็น passive tracer ที่สอดคล้องกับ brightness constancy ได้ดีกว่า) ไม่ควรใช้เป็นตัวตรวจจับเปลวไฟแบบ standalone

---

# 6. Depth Analysis

Stereo depth คำนวณจากการจับคู่ feature ระหว่างภาพ:

$$Z = \frac{fB}{d}$$

โดย $Z$=ระยะทาง, $f$=focal length, $B$=baseline, $d$=disparity

## 6.1 Feature ที่เป็นไปได้

| Feature | สูตร | ความหมาย |
|---|---|---|
| Invalid Depth Ratio | $R_{invalid}$ = สัดส่วนพิกเซลใน ROI ที่ confidence ต่ำกว่า threshold | ยิ่งสูง ยิ่งเอียงไปทางวัตถุโปร่งแสง/ไร้ texture |
| Confidence Variation | ค่าเฉลี่ย confidence map (0-255) ใน ROI | สัญญาณต่อเนื่องแทน threshold ตายตัว |
| Temporal Depth Variance | $\sigma^2_{depth}=\text{Var}(\text{median}(Z_{ROI}))_{\text{sliding window}}$ | วัตถุแข็งทึบให้ depth เสถียร กลุ่มละอองที่ปั่นป่วนให้ depth เต้นไปมา |

## 6.2 สมมติฐานและข้อจำกัด

**สมมติฐาน**: วัตถุคล้าย aerosol (ฝุ่น/ควัน) อาจสร้าง pattern ของ depth ที่ไม่เสถียร เพราะ stereo matching ต้องพึ่ง texture ที่ชัดเจนในการจับคู่ พื้นผิวโปร่งแสง/ไม่มี texture มักให้ depth ที่ noisy หรือคำนวณไม่ได้ — งานวิจัยจาก CMU (FIReStereo) ยืนยันหลักการทั่วไปว่าควัน/ฝุ่นส่งผลกระทบต่อกล้อง visual/stereo อย่างมีนัยสำคัญ

**ข้อควรระวัง**: depth ยังล้มเหลวได้จากสาเหตุอื่นเช่นกัน (texture ต่ำของพื้นผิวทั่วไป, การสะท้อนแสง, สภาพแสง) จึงต้อง**ถือเป็น supporting evidence** ไม่ใช่หลักฐานชี้ขาดเดี่ยวๆ ต้องทดลองเก็บข้อมูลเปรียบเทียบ (ไฟจริง/ควัน/ฝุ่น) เพื่อยืนยันสมมติฐานนี้กับอุปกรณ์จริงก่อนนำไปใช้งานจริง

---

# 7. Multi-Modal Sensor Fusion

## 7.1 ทำไมต้อง Fusion

แต่ละ sensor มีจุดอ่อนต่างกัน:

| Sensor | จุดแข็ง | จุดอ่อน |
|---|---|---|
| Vision | เร็ว | sensitive ต่อความกำกวมทางภาพ (ฝุ่น/แสง/สะท้อนคล้ายไฟ) |
| Thermal | หลักฐานความร้อนที่ชัดเจน | ความละเอียดเชิงพื้นที่จำกัด |
| CO | หลักฐานการเผาไหม้ที่เฉพาะเจาะจง | ตอบสนองช้า |
| PM | ให้ข้อมูลอนุภาค | กำกวมระหว่างฝุ่น/ควันด้วยตัวเอง |

Fusion คือการรวม "หลักฐานที่ไม่สมบูรณ์" จากหลายแหล่งเข้าด้วยกันเพื่อให้ได้ข้อสรุปที่น่าเชื่อถือกว่าการใช้แหล่งเดียว

## 7.2 Dempster-Shafer Evidence Theory (DST)

### เหตุผลที่เลือก DST

DST ถูกพิจารณาเพราะสามารถ**แทนความไม่แน่นอนได้โดยตรง** — ต่างจากการ classification แบบดั้งเดิมที่ sensor แต่ละตัวไม่จำเป็นต้องฟันธงเป็นคำตอบเดียว หลักฐานสามารถสนับสนุนกลุ่มสมมติฐานรวมกันได้ เช่น "Fire" หรือ "Dust" อย่างใดอย่างหนึ่ง (ยังแยกไม่ออก) หรือแม้แต่ "Unknown" (ไม่มีข้อมูลพอตัดสิน)

### นิยามทางคณิตศาสตร์

**Frame of Discernment**: $\Theta = \{Fire, Dust, Normal\}$, Power set $2^\Theta$ คือเซตย่อยทั้งหมด

**Basic Probability Assignment (BPA)**: $m: 2^\Theta \rightarrow [0,1]$, โดย $m(\emptyset)=0$, $\sum_{A\subseteq\Theta}m(A)=1$

**Belief**: $Bel(A) = \sum_{B\subseteq A}m(B)$ (ขอบเขตล่างของความเชื่อ)

**Plausibility**: $Pl(A) = \sum_{B\cap A\neq\emptyset}m(B) = 1-Bel(\bar A)$ (ขอบเขตบน)

**Dempster's Combination Rule**:
$$m_{12}(A) = \frac{1}{1-K}\sum_{B\cap C=A}m_1(B)\cdot m_2(C), \quad A\neq\emptyset$$
$$K = \sum_{B\cap C=\emptyset}m_1(B)\cdot m_2(C) \quad \text{(conflict mass)}$$

### ⚠️ Main Engineering Challenge (จุดสำคัญที่สุดของหัวข้อนี้)

**ส่วนที่ยากไม่ใช่สมการรวมหลักฐาน (Combination Rule) แต่คือการนิยาม BPA** — Basic Probability Assignment **ต้องสร้างจาก**:
- ข้อมูลที่เก็บได้จริง (collected dataset)
- การ calibrate sensor จริง (sensor calibration)
- ผลการทดลองจริง (experimental results)

**ไม่ใช่การกำหนดตัวเลขขึ้นเองแบบ manual guess** — วิธีที่แนะนำในทางปฏิบัติ:
1. เก็บ dataset สถานการณ์ต่างๆ (Fire/Dust/Normal) พร้อม ground truth label
2. คำนวณค่าความถี่สัมพัทธ์ (relative frequency) ของแต่ละ feature range ต่อแต่ละ class จาก dataset จริง เพื่อสร้างเป็น BPA
3. หรือใช้วิธีทางการจากงานวิจัย เช่น แปลง softmax/confidence output ของโมเดลเป็น BPA อย่างมีหลักการ (มีวิธี "novel method to determine BPA" ในงานวิจัยที่แนบลิงก์ไว้ท้ายเอกสาร)
4. ทดสอบ/ปรับ BPA ซ้ำด้วยข้อมูล validation set แยกจาก training

> เอกสารตัวอย่างที่แสดงวิธีคำนวณ (worked example) ด้วยตัวเลขสมมติในภาคผนวกท้ายเอกสารนี้ **ใช้เพื่อสาธิตกลไกการคำนวณเท่านั้น** ตัวเลข BPA จริงต้องมาจาก 4 ขั้นตอนข้างต้น

### Conflict Paradox (Zadeh's Paradox) — ปัญหาที่ต้องเตรียมรับมือ

งานวิจัยเรื่องการตรวจจับไฟในอุโมงค์ (ใช้ Temperature+CO+Smoke sensor เหมือนโครงงานนี้) พบว่าเมื่อ sensor ตัวหนึ่งให้ค่า mass เป็น 0 สำหรับสมมติฐานหนึ่ง (เช่น CO ตอบสนองช้าจนยังไม่ขึ้น ทั้งที่ thermal/smoke ยืนยันไฟแล้ว) Dempster's rule แบบดั้งเดิมจะให้ผลลัพธ์ที่ผิดเพี้ยน (paradox) — ทางแก้ที่เสนอคือ **Two-Level Fusion Framework**: level แรกรวมข้อมูลจาก sensor ประเภทเดียวกันก่อน (กรองความกำกวมออก), level สองคำนวณ relative conflict แล้วปรับ BPA ด้วย trust coefficient ก่อนเข้า Dempster's rule

### เทคนิคขั้นสูงเพิ่มเติม

- **Pignistic Probability Transform**: $BetP(A)=\sum_{B\supseteq A}m(B)/|B|$ — แปลง BPA สุดท้ายเป็นค่าความน่าจะเป็นเดี่ยวสำหรับป้อนเข้า FSM
- **Yager's Rule / PCR5-PCR6 (DSmT)**: ทางเลือกสำหรับสถานการณ์ conflict สูง กระจาย conflict mass กลับไปยัง focal element แทนที่จะทิ้งไปเฉยๆ
- **Reliability Discounting**: $m'(A)=\alpha\cdot m(A)$ สำหรับ $A\neq\Theta$ — ถ่วงน้ำหนัก BPA ตาม reliability ของ sensor (เช่น discount CO sensor ที่มีปัญหา drift)

---

# 8. Decision Layer

## 8.1 Temporal Smoothing

Risk Score อาจผันผวนเพราะ sensor มี noise:

**วิธีเริ่มต้น — Exponential Moving Average**:
$$R(t) = \alpha\cdot x(t) + (1-\alpha)\cdot R(t-1)$$

**ทางเลือกขั้นสูง — Kalman Filter**: โมเดล Risk Score เป็น state ที่มี process noise/measurement noise แยกกัน ให้ผลนิ่งกว่าและมีทฤษฎีรองรับชัดเจน (ตรงกับพื้นฐาน Control Systems)

**คำแนะนำเชิงปฏิบัติ**: เริ่มจาก EMA ก่อนเพราะ implement ง่ายและปรับ $\alpha$ ได้เร็ว ค่อยขยับไป Kalman Filter ถ้าพบว่า noise มีลักษณะซับซ้อนกว่าที่ EMA รับมือได้

## 8.2 Finite State Machine

สถานะที่แนะนำ: **NORMAL → SUSPICIOUS → DUST EVENT → WARNING → CRITICAL → ALARM**

**เหตุผล**: สภาพแวดล้อมจริงมีความไม่แน่นอนสูง ระบบไม่ควรสลับจาก Normal ไป Alarm ทันที ต้องมีสถานะกลางรองรับความกำกวมและใช้ hysteresis logic ป้องกันการสลับสถานะถี่เกินไป

**ข้อควรระวัง**: threshold ที่ใช้กำหนดการเปลี่ยนสถานะต้องเป็นหน่วยเวลา (วินาที) ไม่ใช่หน่วยเฟรม เนื่องจาก FPS ของระบบ (15-30) ไม่คงที่ตายตัวเท่ากับที่เคยออกแบบไว้ที่ 5 FPS

---

# 9. Multi-Rate Architecture

Sensor แต่ละตัวทำงานที่ความถี่ต่างกันโดยธรรมชาติ — นี่คือประเด็นเชิงวิศวกรรมควบคุมที่สำคัญและควรออกแบบ pipeline ให้สอดคล้อง:

| Sensor/โมดูล | ความถี่ | หมวด |
|---|---|---|
| Vision (YOLO+Tracking+Optical Flow) | สูง (15-30 Hz) | **Fast Loop** |
| Depth Analysis | สูง (ตาม frame rate กล้อง) | **Fast Loop** |
| Thermal (MLX90640) | ปานกลาง (~8-16 Hz จริง) | **Fusion Loop** |
| CO Trend | ช้า (response time หลักวินาที) | **Fusion Loop** |
| PM Trend | ช้า (อัปเดตระดับวินาที) | **Fusion Loop** |
| Decision (DST + FSM) | ตาม Fusion Loop | **Fusion Loop** |

**การออกแบบที่แนะนำ**: แยก pipeline เป็น 2 loop ที่ทำงานคนละความถี่กัน — Fast Loop คำนวณ feature จาก vision/depth ทุกเฟรมและ buffer ผลไว้ ส่วน Fusion Loop จะดึงค่าเฉลี่ย/ค่าล่าสุดจาก Fast Loop buffer มารวมกับค่า sensor เคมี/ความร้อนที่อัปเดตช้ากว่า แล้วคำนวณ Risk Score ที่ความถี่ของ Fusion Loop เอง (ไม่จำเป็นต้องรอทุก sensor พร้อมกันทุกเฟรม) — แนวคิดนี้ตรงกับหลักการ **multi-rate control system** ในทฤษฎีระบบควบคุม และควรใช้เป็น framing หลักในการอธิบายสถาปัตยกรรมซอฟต์แวร์ในเล่ม (เชื่อมโยงกับ TSC "Control & Systems Theory" ที่ทีมเลือกไว้)

---

# 10. Recommended Validation Strategy

## 10.1 Scenario Dataset

1. Normal environment
2. Dust event
3. Smoke event
4. Fire event
5. Interference event (แสงสะท้อน/LED/วัตถุสีคล้ายไฟ)

## 10.2 Evaluation Metrics

Precision, Recall, F1-score, False Alarm Rate, Detection Delay (response time)

## 10.3 Fire Simulation Software สำหรับการทดสอบที่ปลอดภัยกว่า

งานวิจัยเรื่อง Tunnel Fire Detection (DST) ใช้ซอฟต์แวร์จำลองไฟ **PyroSim** (GUI ของ Fire Dynamics Simulator - FDS จาก NIST) สร้างสถานการณ์ไฟจำลองหลายแบบ (heat release rate, ความเร็วลมต่างกัน) แล้วดึงข้อมูล sensor reading จากผลจำลองมาทดสอบระบบ fusion ก่อนไปทดสอบไฟจริง — วิธีนี้ปลอดภัยกว่าและประหยัดกว่าการจุดไฟจริงซ้ำๆ และช่วยทดสอบสถานการณ์ที่อันตรายเกินกว่าจะทำจริงได้ (ไฟขนาดใหญ่, ลมแรง) แนะนำใช้ FDS/PyroSim (ฟรี, จาก NIST) สร้างข้อมูลสังเคราะห์เพิ่มเติมสำหรับ threshold tuning ก่อนทดสอบภาคสนามจริง

---

# 11. สรุป Engineering Decisions ทั้งหมด (ADR Format)

| # | Decision | Reason | Limitation |
|---|---|---|---|
| 1 | OAK-D เป็นกล้อง RGB+Depth เท่านั้น | รวม compute ที่ Pi+Hailo-8 ศูนย์กลางเดียว ง่ายต่อ deployment/fusion | ต้องพึ่ง bandwidth USB ส่งข้อมูลทุกเฟรม |
| 2 | YOLO11-seg แทน detection อย่างเดียว | Segmentation ให้ pixel-level mask/area/shape ที่จำเป็นต่อ temporal analysis | ต้อง label mask ซึ่งใช้เวลานานกว่า bbox |
| 3 | เพิ่ม SORT/ByteTrack | แก้ปัญหาไม่มี identity ข้ามเฟรม จำเป็นต่อการวัด area growth | เพิ่มความซับซ้อนของ pipeline |
| 4 | Optical Flow เป็น supporting evidence | ให้ข้อมูลพฤติกรรมเสริม แยกควัน/ฝุ่นจากการเคลื่อนไหว | ไม่น่าเชื่อถือเป็นตัวตรวจจับเปลวไฟแบบ standalone (ละเมิด brightness constancy) |
| 5 | CO เป็น trend/evidence ไม่ใช่ hard gate | หลักฐานการเผาไหม้มีค่า แต่ response ช้าเกินกว่าจะใช้ตัดสินทันที | ต้องออกแบบ logic รองรับความล่าช้านี้โดยเฉพาะ |
| 6 | DST หลังเก็บ dataset เท่านั้น | BPA ต้องมาจากการสร้างแบบจำลองหลักฐานที่เชื่อถือได้ ไม่ใช่กำหนดเอง | ต้องมีขั้นตอนเก็บข้อมูล+calibrate ก่อนใช้งานจริงได้ |
| 7 | Multi-rate architecture (Fast/Fusion loop) | Sensor ทำงานคนละความถี่ ไม่ควรบังคับให้ sync กันทุกเฟรม | ต้องออกแบบ buffer/interface ระหว่าง 2 loop ให้ดี |
| 8 | DINOv3/Active Learning เป็น Future Work | ไม่ช่วยลด false alarm แบบ real-time, ไม่อยู่ใน scope/timeline เดิม | ต้องปรับข้อ 4 ของเล่มให้สะท้อนตรงนี้ |

---

# 12. Research Contribution Direction

แนวทาง contribution ที่เป็นไปได้: การพัฒนาระบบตรวจจับไฟแบบ edge-based ที่ทนทาน (robust) ต่อการแจ้งเตือนผิดพลาดในสภาพแวดล้อมที่มีฝุ่นหนาแน่น ผ่าน:
- Spatiotemporal vision analysis (segmentation + tracking + optical flow)
- Multi-modal sensing (vision + thermal + gas + particulate)
- Uncertainty-aware evidence fusion (Dempster-Shafer พร้อม BPA ที่ calibrate จากข้อมูลจริง)

---

# 13. เอกสารอ้างอิง (รวมทั้งหมด)

## รากฐานทางทฤษฎี
- Horn, B.K.P., Schunck, B.G. (1981). *Determining Optical Flow*. Artificial Intelligence 17:185-304
- Lucas, B.D., Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision*
- Bruhn, A., Weickert, J. (2005). *Lucas/Kanade Meets Horn/Schunck* — https://www.mia.uni-saarland.de/Publications/bruhn-ijcv05c.pdf
- Shafer, G. (1976). *A Mathematical Theory of Evidence*

## งานวิจัยเฉพาะทางที่ตรงกับโครงงาน
- A Tunnel Fire Detection Method Based on an Improved Dempster-Shafer Evidence Theory (Sensors, 2024) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11479314/ (DOI: 10.3390/s24196455)
- Multisensor data fusion for fire detection (SCIER system) — https://www.sciencedirect.com/science/article/abs/pii/S1566253509001006
- A Data Fusion-Based Fire Detection System (IEICE, Arduino-based) — https://search.ieice.org/bin/summary.php?id=e101-d_4_977
- Combination of Evidence in Dempster-Shafer Theory (Sandia National Labs) — https://www.stat.berkeley.edu/~aldous/Real_World/dempster_shafer.pdf
- A novel method to determine BPA in DST — https://journals.sagepub.com/doi/10.1177/1550147719865876
- Time-Domain Data Fusion Using Weighted Evidence and DS Combination Rule — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928930/
- Optical Flow Estimation for Flame Detection in Videos — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4000537/
- Unsupervised Segmentation of Fire and Smoke from Infra-Red Videos — https://arxiv.org/pdf/1909.12937
- Video Fire Smoke Detection Using Motion and Color Features (Fire Technology, Springer) — https://link.springer.com/article/10.1007/s10694-009-0110-z
- Optical Flow Feature Based for Fire Detection on Video Data (IEEE) — https://ieeexplore.ieee.org/document/8970957/
- Time-Series Correlation Optimization for Forest Fire Tracking (AO-OCSORT) — https://doi.org/10.3390/f16071101
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box — https://www.alphaxiv.org/overview/2110.06864v3

## เอกสารทางเทคนิค/Datasheet
- Luxonis DepthAI documentation — https://docs.luxonis.com
- Hailo Model Zoo documentation
- Ultralytics YOLO11/YOLO26 documentation — https://docs.ultralytics.com
- NIST Fire Dynamics Simulator (FDS) / PyroSim (Thunderhead Engineering)
- Sensor datasheets: MLX90640, ZE07-CO, PMS5003, Arduino UNO Q

---

*เอกสารนี้เป็นเวอร์ชันรวม (merged) ที่ผสาน framing เชิงวิศวกรรมจากการสังเคราะห์ของ ChatGPT (ปรัชญาการออกแบบ, multi-rate architecture, ADR format, คำเตือนเรื่อง calibration) เข้ากับรายละเอียดเทคนิคเชิงลึกทั้งหมดที่พัฒนาร่วมกัน (model comparison, object tracking, สูตรคณิตศาสตร์แบบเต็ม, ลิงก์งานวิจัยเฉพาะทาง) ใช้เป็นเอกสารอ้างอิงหลักฉบับเดียวสำหรับการแก้ไขเล่มโครงงานและพัฒนาระบบจริงต่อไป*
