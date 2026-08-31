# Workflow สุดท้ายของระบบตรวจจับอัคคีภัย (ฉบับสมบูรณ์)
### รวมทุกเทคนิคที่ใช้ในแต่ละขั้นตอน พร้อมข้อเสนอแนะเพิ่มเติมล่าสุด

---

## แผนภาพ Flow การทำงาน

```
┌─────────────────────────────────────────────────────────────┐
│  1. รับข้อมูล (Sensing Layer)                                  │
│     OAK-D: RGB + Depth map + Confidence map (ไม่มี AI)         │
│     ESP32: MLX90640 (Thermal) + ZE07-CO + PMS5003 (PM)        │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. AI ตรวจจับ + ติดตาม (Raspberry Pi 5 + Hailo-8)             │
│     YOLO11n/s-seg → bbox + class + mask                       │
│     + SORT/ByteTrack (Kalman Filter + IoU) → track ID          │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. สกัด Feature (ขนานกัน 5 กลุ่ม)                             │
│     • Optical Flow: divergence, curl, directional histogram,  │
│       turbulence energy (PSD/FFT)                              │
│     • Depth: invalid ratio, temporal variance, ระยะทาง          │
│     • PM Ratio: DTPR = PM2.5/PM10                               │
│     • CO: conditional combustion gate                          │
│     • Thermal: IoU กับตำแหน่งที่ vision ตรวจพบ                  │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. รวมหลักฐาน (Sensor Fusion)                                 │
│     Dempster-Shafer Combination Rule → BPA รวม                 │
│     → Pignistic Probability Transform → Risk Score ดิบ          │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4.5 [เพิ่มใหม่] Temporal Smoothing                             │
│      Exponential Moving Average หรือ Kalman Filter              │
│      กรอง noise ของ Risk Score ก่อนตัดสินใจ                     │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. ตัดสินสถานะ (Finite State Machine)                          │
│     Normal → Dust Event → Suspicious → Warning → Critical      │
│     → Alarm  (พร้อม Hysteresis Logic)                            │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. แสดงผล/แจ้งเตือน                                            │
│     Dashboard, Database logging, LED/Buzzer                     │
└───────────────────────────┬───────────────────────────────────┘
                            │  (async, ไม่ real-time)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  7. [Feedback Loop] Active Learning                             │
│     Flag mismatch → FiftyOne/CVAT (PC) → Human label            │
│     → Retrain YOLO11-seg → Redeploy                              │
└─────────────────────────────────────────────────────────────┘
```

---

## รายละเอียดเทคนิคในแต่ละขั้นตอน

### ขั้นที่ 1 — Sensing Layer

| อุปกรณ์ | ข้อมูลที่ให้ | หมายเหตุ |
|---|---|---|
| OAK-D | RGB (12MP), Depth map, Confidence map (0-255/พิกเซล) | ไม่รัน AI on-device — ใช้แค่ stereo-matching hardware block |
| MLX90640 | Thermal array 32×24 px | Refresh rate จริงมักอยู่ 8-16Hz แม้ spec สูงสุด 64Hz |
| ZE07-CO | ความเข้มข้น CO (ppm) | มี temperature compensation ในตัว |
| PMS5003 | PM1.0, PM2.5, PM10 แยกกัน | ใช้คำนวณ ratio ได้โดยตรง |

**FPS เป้าหมาย 15-30** — เพียงพอตาม Nyquist theorem สำหรับจับสัญญาณ flame flicker ~10Hz

### ขั้นที่ 2 — AI ตรวจจับ + ติดตาม

- **YOLO11n-seg หรือ YOLO11s-seg** รันบน Hailo-8 (26 TOPS) — โมเดลเดียวให้ทั้ง bbox, class, mask ในการ forward pass เดียว
- **[เพิ่มใหม่] Multi-Object Tracking** — ใช้ตระกูล SORT/ByteTrack ต่อจาก detection:
  - ByteTrack ใช้ Kalman Filter ทำนายตำแหน่งถัดไป + IoU matching จับคู่กับ detection ใหม่
  - แก้ปัญหา "ไม่รู้ว่าจุดที่ตรวจพบในเฟรมต่างกันเป็นไฟจุดเดียวกันหรือไม่" ซึ่งจำเป็นสำหรับวัดการขยายตัวของพื้นที่ (5.4.3)
  - มีงานวิจัยเฉพาะทางไฟป่า (AO-OCSORT) ที่ใช้ YOLOX + OC-SORT พร้อมเสริม similarity metric จาก optical flow และ contour ของเปลวไฟโดยเฉพาะ — ยืนยันว่าแนวทางนี้ใช้ได้จริงในโดเมนนี้

### ขั้นที่ 3 — สกัด Feature (5 กลุ่มขนาน)

**Optical Flow** (จาก mask/bbox ที่ track ได้):
- Divergence: $\nabla \cdot V = \partial u/\partial x + \partial v/\partial y$ — วัดการขยายตัว
- Curl: $\nabla \times V = \partial v/\partial x - \partial u/\partial y$ — วัดการหมุนวนที่ขอบ plume
- Directional histogram ด้วย circular mean และ mean resultant length $R$
- Turbulence energy: variance ของขนาด flow + PSD ผ่าน FFT (เชื่อมกับ flame flicker ~10Hz)
- **ข้อควรระวัง**: เปลวไฟละเมิด brightness constancy assumption (เรืองแสงในตัวเอง) — optical flow แม่นกับควันมากกว่าตัวไฟโดยตรง

**Depth-based** (จาก OAK-D confidence map):
- Invalid/low-confidence ratio: $R_{invalid}$ ในพื้นที่ ROI
- Temporal depth instability: $\sigma^2_{depth} = \text{Var}(\text{median}(Z_{ROI}))$
- ใช้หลักการที่ stereo matching ต้องพึ่ง texture — ควัน/ฝุ่นโปร่งแสงไม่มี texture ชัดเจน มักให้ depth ไม่แม่นยำ (มีงานวิจัย CMU ยืนยันว่าควันรบกวนกล้อง visual/stereo อย่างมีนัยสำคัญ)

**PM Ratio**: $DTPR = PM_{2.5}/PM_{10}$ — DTPR ≤ 0.55 → ฝุ่น, 0.55-0.96 → ควัน, > 0.96 → หมอก

**CO Gate**: ใช้เป็นเงื่อนไข (ไม่ใช่ threshold เดี่ยว) — ยืนยันการเผาไหม้ทางเคมี

**Thermal IoU**: เทียบ overlap ระหว่างจุดร้อนจาก MLX90640 กับตำแหน่ง vision ตรวจพบ

### ขั้นที่ 4 — Sensor Fusion (Dempster-Shafer)

- Frame of Discernment: $\Theta = \{Fire, Dust, Normal\}$
- แต่ละ feature จากขั้นที่ 3 แปลงเป็น Basic Probability Assignment (BPA)
- รวมด้วย Dempster's Combination Rule: $m_{12}(A) = \frac{1}{1-K}\sum_{B\cap C=A} m_1(B)m_2(C)$
- ระวัง Conflict Paradox เมื่อ sensor ตัวใดตัวหนึ่งตอบสนองช้า/ผิดปกติ (เช่น CO drift) — แก้ด้วย two-level fusion framework หรือ reliability discounting
- แปลงผลสุดท้ายเป็นค่าเดี่ยวด้วย Pignistic Probability Transform: $BetP(A) = \sum_{B\supseteq A} m(B)/|B|$

### ขั้นที่ 4.5 — [เพิ่มใหม่] Temporal Smoothing

Risk Score ดิบจาก Dempster-Shafer คำนวณอิสระทุกเฟรม อาจกระโดดตาม noise ชั่วขณะ แนะนำกรองก่อนเข้า FSM:

- **แบบง่าย** — Exponential Moving Average: $\hat{R}_t = \alpha R_t + (1-\alpha)\hat{R}_{t-1}$
- **แบบรัดกุมกว่า** — Kalman Filter: โมเดล Risk Score เป็น state ที่มี process noise/measurement noise แยกกัน ให้ผลนิ่งกว่าและมีทฤษฎีรองรับชัดเจน (ตรงกับพื้นฐาน Control Systems ที่ทีมมีอยู่แล้ว)

### ขั้นที่ 5 — Finite State Machine

- สถานะ: Normal → Dust Event → Light Interference → Suspicious → Warning → Critical → Alarm
- Threshold ต้องเป็น **หน่วยเวลา (วินาที)** ไม่ใช่หน่วยเฟรม เพราะ FPS เปลี่ยนจาก 5 เป็น 15-30 แล้ว
- มี Hysteresis Logic ป้องกันการสลับสถานะถี่เกินไป

### ขั้นที่ 6 — แสดงผล/แจ้งเตือน

- Dashboard แสดงภาพ, ผลตรวจจับ, ค่า sensor, Risk Score, สถานะปัจจุบัน
- บันทึก Database (เวลา, snapshot, ค่า sensor, risk score)
- เปิด LED/Buzzer เมื่อถึงระดับความเสี่ยงสูง

### ขั้นที่ 7 — [Feedback Loop] Active Learning (Async, ไม่ใช่ real-time)

- Flag เคส mismatch ระหว่างผล Vision กับ Sensor Fusion (query-by-committee)
- ใช้ **FiftyOne + CVAT** (บน PC) สำหรับ human-in-the-loop labeling แทน DINOv3
- Retrain YOLO11-seg ตัวเดิมเป็นระยะ แล้ว redeploy

---

## ข้อเสนอแนะเพิ่มเติมล่าสุด (สรุปสั้น)

| ข้อเสนอ | ปัญหาที่แก้ | เทคนิค |
|---|---|---|
| **Object Tracking** | ไม่มีกลไกจับคู่วัตถุเดียวกันข้ามเฟรม ทำให้วัดการขยายตัวไม่แม่น | SORT/ByteTrack (Kalman Filter + IoU) |
| **Temporal Smoothing** | Risk Score กระโดดตาม noise ชั่วขณะ | Exponential Moving Average หรือ Kalman Filter |
| **Fire Simulation Software** | การทดสอบไฟจริงซ้ำๆ อันตรายและมีต้นทุนสูง | PyroSim/FDS (NIST, ฟรี) จำลองสถานการณ์ก่อนทดสอบจริง |

---

## เอกสารอ้างอิงที่เกี่ยวข้อง (เพิ่มจากไฟล์ก่อนหน้า)

| หัวข้อ | ลิงก์ |
|---|---|
| Time-Series Correlation Optimization for Forest Fire Tracking (AO-OCSORT: YOLOX + OC-SORT สำหรับไฟป่าโดยเฉพาะ) | https://doi.org/10.3390/f16071101 |
| ByteTrack: Multi-Object Tracking by Associating Every Detection Box | https://www.alphaxiv.org/overview/2110.06864v3 |
| A Tunnel Fire Detection Method Based on an Improved Dempster-Shafer Evidence Theory (มีการใช้ PyroSim จำลองสถานการณ์ไฟ) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11479314/ |
| Fire Dynamics Simulator (FDS) และ PyroSim — ซอฟต์แวร์จำลองไฟจาก NIST | ค้นหา "NIST Fire Dynamics Simulator" หรือ "PyroSim Thunderhead Engineering" |

---

*เอกสารนี้เป็นเวอร์ชันล่าสุดของ workflow ระบบ ต่อยอดจากไฟล์ `สรุปแนวคิดเชิงเทคนิค_Fire_Detection_System.md` และ `Dempster-Shafer_และ_OpticalFlow_เทคนิคขั้นสูง.md` ที่สร้างไว้ก่อนหน้า ใช้ประกอบกันทั้ง 3 ไฟล์สำหรับแก้ไขเล่มโครงงานและพัฒนาระบบจริง*
