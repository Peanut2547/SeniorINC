# เอกสารเชิงลึก: Dempster-Shafer Evidence Theory และ Optical Flow Feature Extraction
### สำหรับระบบ Sensor Fusion ตรวจจับอัคคีภัยในสภาพแวดล้อมที่มีฝุ่นหนาแนน่

---

# ส่วนที่ 1: Dempster-Shafer Evidence Theory (DST)

## 1.1 ที่มาและแนวคิดหลัก

Dempster-Shafer Theory ถูกวางรากฐานโดย Arthur P. Dempster ในปี 1967 ผ่านแนวคิดเรื่อง upper/lower probability สำหรับจัดการ multi-valued mapping จากนั้นลูกศิษย์ของเขา Glenn Shafer ได้พัฒนาต่อในหนังสือ *A Mathematical Theory of Evidence* (1976) ทำให้กลายเป็นทฤษฎีที่สมบูรณ์สำหรับการวัดปริมาณความไม่แน่นอน (uncertainty quantification)

**ทำไมถึงเหมาะกับปัญหาของพี่มากกว่า Bayesian แบบดั้งเดิม**: Bayesian ต้องการ prior probability ที่ชัดเจนสำหรับทุกสมมติฐาน และไม่มีกลไกแยก "ไม่มีข้อมูลพอจะตัดสิน" ออกจาก "มีข้อมูลว่าไม่ใช่" อย่างชัดเจน ในขณะที่ DST มีแนวคิด **frame of discernment** ที่อนุญาตให้ sensor แต่ละตัว "โหวต" ความเชื่อให้กับกลุ่มสมมติฐานรวมกันได้ (ไม่ต้องฟันธงเป็นสมมติฐานเดียว) ซึ่งตรงกับสถานการณ์จริงมาก เช่น PM sensor รู้แค่ว่า "น่าจะเป็นฝุ่นหรือควันอย่างใดอย่างหนึ่ง" แต่แยกไม่ออกด้วยตัวเอง

## 1.2 นิยามทางคณิตศาสตร์

### Frame of Discernment
กำหนดสมมติฐานทั้งหมดที่เป็นไปได้และแยกจากกันโดยสมบูรณ์ (mutually exclusive and exhaustive):

$$\Theta = \{\text{Fire}, \text{Dust}, \text{Normal}\}$$

Power set ของ Θ คือเซตย่อยทั้งหมด รวม ∅ (empty set) และ Θ เอง:

$$2^\Theta = \{\emptyset, \{Fire\}, \{Dust\}, \{Normal\}, \{Fire,Dust\}, \{Fire,Normal\}, \{Dust,Normal\}, \Theta\}$$

### Basic Probability Assignment (BPA) / Mass Function
ฟังก์ชัน mass เป็นตัวแทนของ "หลักฐาน" จาก sensor แต่ละตัว:

$$m: 2^\Theta \rightarrow [0,1]$$

โดยต้องสอดคล้องเงื่อนไข:

$$m(\emptyset) = 0, \qquad \sum_{A \subseteq \Theta} m(A) = 1$$

เซตย่อย $A$ ที่ $m(A) > 0$ เรียกว่า **focal element** — จุดสำคัญคือ mass สามารถกำหนดให้กับเซตรวม เช่น $m(\{Fire, Dust\}) = 0.3$ ได้ ซึ่งหมายถึง "หลักฐานสนับสนุนว่าเป็นอย่างใดอย่างหนึ่งใน 2 อย่างนี้ แต่ยังแยกไม่ออก" — สิ่งนี้ทำไม่ได้ใน Bayesian แบบมาตรฐาน

### Belief และ Plausibility Function
Belief คือขอบเขตล่างของความเชื่อ (หลักฐานที่สนับสนุน A โดยตรง):

$$Bel(A) = \sum_{B \subseteq A} m(B)$$

Plausibility คือขอบเขตบน (หลักฐานที่ไม่ขัดแย้งกับ A):

$$Pl(A) = \sum_{B \cap A \neq \emptyset} m(B) = 1 - Bel(\bar{A})$$

ช่วง $[Bel(A), Pl(A)]$ คือ "ช่วงความเชื่อ" ที่สะท้อนความไม่แน่นอนได้ตรงกว่าตัวเลขเดี่ยวแบบ Bayesian probability

## 1.3 Dempster's Rule of Combination (สมการรวมหลักฐาน)

เมื่อมี 2 แหล่งข้อมูล (เช่น Vision model กับ Sensor Fusion) แต่ละแหล่งให้ BPA เป็น $m_1$ และ $m_2$ รวมกันด้วย orthogonal sum ($\oplus$):

$$m_{12}(A) = \frac{1}{1-K} \sum_{B \cap C = A} m_1(B) \cdot m_2(C), \quad A \neq \emptyset$$

โดย $K$ คือ **conflict mass** ที่วัดว่าหลักฐาน 2 แหล่งขัดแย้งกันมากแค่ไหน:

$$K = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)$$

ถ้ามี sensor มากกว่า 2 ตัว (Vision, Thermal, CO, PM, Depth) สามารถรวมทีละคู่ต่อเนื่องได้ เพราะ Dempster's rule มีคุณสมบัติ **commutative** (สลับลำดับได้) แม้จะไม่ใช่ associative ทุกกรณีก็ตาม

## 1.4 ตัวอย่างการคำนวณจริง (Worked Example) สำหรับโครงงานของพี่

สมมติ Θ = {Fire, Dust, Normal} และในเฟรมหนึ่งได้ค่า BPA จาก 2 แหล่งข้อมูล:

**Vision model (YOLO11-seg + Optical Flow)**:
- m₁(Fire) = 0.6, m₁(Dust) = 0.1, m₁({Fire,Dust}) = 0.2, m₁(Θ) = 0.1 *(ความไม่แน่นอนที่เหลือ)*

**Sensor Fusion (Thermal + CO + PM ratio)**:
- m₂(Fire) = 0.5, m₂(Dust) = 0.2, m₂({Fire,Dust}) = 0.1, m₂(Θ) = 0.2

คำนวณ conflict mass $K$ จากคู่ที่ intersection เป็น ∅ (เช่น m₁(Fire)×m₂(Dust) และ m₁(Dust)×m₂(Fire)):

$$K = (0.6)(0.2) + (0.1)(0.5) = 0.12 + 0.05 = 0.17$$

คำนวณ $m_{12}(Fire)$ จากทุกคู่ที่ intersection = {Fire}:

$$m_{12}(Fire) = \frac{(0.6)(0.5) + (0.6)(0.2) + (0.2)(0.5) + (0.6)(0.1) + (0.1)(0.5)}{1-0.17} = \frac{0.3+0.12+0.1+0.06+0.05}{0.83} \approx 0.75$$

จะเห็นว่าหลังรวม 2 แหล่งข้อมูล ความเชื่อมั่นใน "Fire" เพิ่มขึ้นจาก 0.5-0.6 เป็น ~0.75 เพราะทั้ง 2 แหล่งเห็นตรงกัน — นี่คือพลังของ evidence fusion เทียบกับการใช้ sensor เดียวเดี่ยวๆ

## 1.5 ปัญหาสำคัญที่ต้องรู้: Conflict Paradox (Zadeh's Paradox)

**นี่คือจุดที่สำคัญที่สุดสำหรับโครงงานของพี่**: งานวิจัยเรื่องการตรวจจับไฟในอุโมงค์ที่ใช้ DS theory กับ sensor วัดอุณหภูมิ, CO, และควัน พบปัญหาโดยตรง — เมื่อ temperature sensor กับ smoke sensor บ่งชี้ความน่าจะเป็นไฟสูง แต่ CO sensor กลับให้ค่าปกติ (เช่น m(fire)=0 จาก CO) จะเกิด "paradox" ในการรวมข้อมูลด้วย Dempster's combination rule แบบดั้งเดิม เพราะเมื่อ mass ของสมมติฐานหนึ่งเป็น 0 จาก sensor ใดสensor หนึ่ง ผลการรวมจะถูกบังคับให้เป็น 0 ไปด้วย ทั้งที่ sensor อื่นๆ ยืนยันตรงกันหมด — สถานการณ์นี้ **ตรงกับเคสของพี่เป๊ะ** เช่น กรณี PM สูงมาก + Thermal จับความร้อนได้ แต่ CO sensor ตอบสนองช้า (ตามที่ระบุใน risk table ข้อ 5.9) หากใช้ Dempster's rule แบบดั้งเดิมตรงๆ อาจได้ผลลัพธ์ที่ผิดเพี้ยนได้

**ทางแก้ที่งานวิจัยเสนอ**: ใช้ **two-level fusion framework** — level แรกรวมข้อมูลจาก sensor ประเภทเดียวกันก่อน (feature-level fusion) เพื่อกรองข้อมูลกำกวมออกก่อน แล้วคำนวณ BPA จาก feature ที่ผ่านการกรองแล้ว จากนั้น level สองคำนวณระดับ conflict ระหว่างหลักฐาน normalize ค่า relative conflict แล้วปรับ BPA ด้วย trust coefficient ก่อนจะนำเข้า Dempster's rule แบบดั้งเดิมอีกที — วิธีนี้ช่วยลดผลกระทบจาก sensor ตัวเดียวที่ทำงานผิดปกติ/ตอบสนองช้าได้ดีขึ้นมาก

## 1.6 เทคนิคขั้นสูงเพิ่มเติมที่ควรรู้ (สำหรับต่อยอด)

1. **Pignistic Probability Transformation (BetP)** — ใช้แปลงผลลัพธ์ BPA สุดท้ายให้เป็นค่าความน่าจะเป็นเดี่ยว (scalar) สำหรับป้อนเข้า Finite State Machine:
   $$BetP(A) = \sum_{B \supseteq A} \frac{m(B)}{|B|}, \quad \forall A \in \Theta$$
   วิธีนี้กระจาย mass ของเซตรวม (เช่น {Fire,Dust}) ไปยังสมาชิกแต่ละตัวอย่างเท่าเทียม เหมาะกับการตัดสินใจขั้นสุดท้าย

2. **Yager's Rule / PCR5-PCR6 (จาก Dezert-Smarandache Theory, DSmT)** — เป็นกฎรวมหลักฐานทางเลือกที่ออกแบบมาสำหรับสถานการณ์ conflict สูงโดยเฉพาะ แทนที่จะทิ้ง conflict mass ($K$) ไปเฉยๆ แบบ Dempster's rule ดั้งเดิม จะ**กระจาย (redistribute)** conflict mass กลับไปยัง focal element ที่เกี่ยวข้องตามสัดส่วน — เหมาะเป็นหัวข้อศึกษาต่อถ้าพบว่า conflict ในระบบสูงบ่อยครั้ง

3. **Reliability Discounting** — ถ่วงน้ำหนัก BPA ของแต่ละ sensor ตาม reliability score ก่อนรวม (เช่น CO sensor ที่มีปัญหา drift ตาม risk table ควรถูก discount ลง) ด้วยสูตร:
   $$m'(A) = \alpha \cdot m(A) \text{ สำหรับ } A \neq \Theta, \qquad m'(\Theta) = 1-\alpha(1-m(\Theta))$$
   โดย $\alpha \in [0,1]$ คือค่า reliability ของ sensor นั้น (0 = ไม่น่าเชื่อถือเลย, 1 = เชื่อถือเต็มที่)

## 1.7 บทความ/งานวิจัยแนะนำสำหรับศึกษาต่อ

| หัวข้อ | ลิงก์ |
|---|---|
| **A Tunnel Fire Detection Method Based on an Improved Dempster-Shafer Evidence Theory** (Sensors, 2024) — ตรงกับปัญหาของพี่มากที่สุด ใช้ Temperature+CO+Smoke sensor พร้อมแก้ปัญหา conflict paradox | https://pmc.ncbi.nlm.nih.gov/articles/PMC11479314/ (DOI: 10.3390/s24196455) |
| **Multisensor data fusion for fire detection (SCIER system)** — รวม temperature/humidity sensor กับ vision sensor ผ่าน 2-level fusion (cumulative sum + DS theory) | https://www.sciencedirect.com/science/article/abs/pii/S1566253509001006 |
| **A Data Fusion-Based Fire Detection System** (IEICE) — ใช้ Arduino + DS theory กับ sensor 3 ประเภท ใกล้เคียงกับ scope โครงงานระดับนักศึกษา | https://search.ieice.org/bin/summary.php?id=e101-d_4_977 |
| **Combination of Evidence in Dempster-Shafer Theory** (Sandia National Labs) — เอกสารพื้นฐานที่อธิบาย Dempster's rule, Yager's rule และกฎทางเลือกอื่นๆ อย่างละเอียด | https://www.stat.berkeley.edu/~aldous/Real_World/dempster_shafer.pdf |
| **A novel method to determine basic probability assignment in Dempster–Shafer theory and its application in multi-sensor information fusion** — สอนวิธีสร้าง BPA จากข้อมูลตัวเลขจริง (ไม่ใช่แค่กำหนดเอง) | https://journals.sagepub.com/doi/10.1177/1550147719865876 |
| **Time-Domain Data Fusion Using Weighted Evidence and DS Combination Rule** — ตัวอย่างการใช้ DS theory ร่วมกับผลจาก CNN classifier (ใกล้เคียงกับที่พี่จะใช้ YOLO) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928930/ |

---

# ส่วนที่ 2: Optical Flow Feature Extraction

## 2.1 หลักการพื้นฐาน: Brightness Constancy Assumption

Optical flow ตั้งอยู่บนสมมติฐานว่าความสว่างของจุดภาพเดียวกันจะคงที่ระหว่างเฟรม แค่ตำแหน่งเปลี่ยนไป:

$$I(x, y, t) = I(x+\Delta x, y+\Delta y, t+\Delta t)$$

ขยายด้วย Taylor series แล้วตัดพจน์อันดับสูงทิ้ง (first-order approximation) จะได้ **Optical Flow Constraint Equation**:

$$I_x u + I_y v + I_t = 0$$

โดย $I_x, I_y, I_t$ คือ partial derivative ของความสว่างตามแกน x, y, และเวลา ส่วน $u = dx/dt$, $v = dy/dt$ คือ velocity ที่ต้องการหา

**ปัญหา**: สมการนี้มี 1 สมการ 2 ตัวแปรไม่ทราบค่า ($u, v$) → แก้ไม่ได้โดยตรง เรียกว่า **Aperture Problem** ต้องมีสมมติฐานเพิ่มเติมเพื่อปิดระบบสมการ (นี่คือจุดที่ Horn-Schunck กับ Lucas-Kanade แยกทางกัน)

## 2.2 Horn-Schunck Method (Global Method)

เพิ่มเงื่อนไข **smoothness constraint** ทั่วทั้งภาพ แล้วแปลงปัญหาเป็นการหาค่าต่ำสุดของ energy functional:

$$E(u,v) = \iint \Big[ (I_x u + I_y v + I_t)^2 + \alpha^2 \big(|\nabla u|^2 + |\nabla v|^2\big) \Big] \, dx\, dy$$

พจน์แรกคือ data term (จากสมการ optical flow) พจน์ที่สองคือ regularization term ที่บังคับให้ flow field เปลี่ยนแปลงอย่างนุ่มนวล ($\alpha$ ควบคุมน้ำหนักของความนุ่มนวล) แก้โดยใช้ Euler-Lagrange equation แล้ว iterate ด้วยวิธี Gauss-Seidel

**ข้อดี**: ให้ flow field แบบ dense (ทุกพิกเซล) เหมาะกับการวิเคราะห์รูปทรง/พื้นที่ของควันที่ไม่มีขอบชัดเจน
**ข้อเสีย**: sensitive ต่อ noise เพราะ smoothness assumption อาจ "เกลี่ย" edge ที่แท้จริงออกไป

## 2.3 Lucas-Kanade Method (Local Method)

สมมติว่า flow คงที่ภายในหน้าต่างเล็กๆ (local window) รอบแต่ละพิกเซล ทำให้ได้ระบบสมการเกินกำหนด (overdetermined) แก้ด้วย least squares:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T b$$

โดย $A$ คือเมทริกซ์ของ $[I_x, I_y]$ ทุกพิกเซลในหน้าต่าง และ $b$ คือเวกเตอร์ของ $-I_t$

**เงื่อนไขสำคัญ**: $A^TA$ ต้อง invertible ได้ (ไม่ singular) ซึ่งหมายความว่าต้องมี gradient ในหลายทิศทางภายในหน้าต่างนั้น (ไม่ใช่พื้นผิวเรียบ) — นี่คือเหตุผลที่ Lucas-Kanade มักใช้ร่วมกับ **corner detector** (เช่น Shi-Tomasi "Good Features to Track") เพื่อเลือกจุดที่มี texture ชัดเจนมาติดตามก่อน

**ข้อสังเกตสำคัญสำหรับพี่**: นี่คือเหตุผลเดียวกับที่ทำให้ stereo depth ของ OAK-D มีปัญหากับควัน/ฝุ่นที่ไม่มี texture (ตามที่คุยกันไปก่อนหน้านี้) — Lucas-Kanade ก็มีจุดอ่อนแบบเดียวกันกับพื้นที่ที่ texture ต่ำ ควันจึงอาจ track ด้วย LK ได้ไม่แม่นยำเท่าวัตถุทึบเช่นกัน

## 2.4 Farneback Method (Dense, ใช้ใน OpenCV)

ประมาณค่าความสว่างในแต่ละหน้าต่างท้องถิ่นด้วย **polynomial expansion** ระดับ 2 (quadratic polynomial) แล้วเปรียบเทียบสัมประสิทธิ์ของ polynomial ระหว่าง 2 เฟรมเพื่อประมาณ displacement — ให้ dense flow field ครอบคลุมทุกพิกเซลเหมือน Horn-Schunck แต่คำนวณเร็วกว่า เป็นตัวเลือกที่ใช้งานได้จริงบน embedded (`cv2.calcOpticalFlowFarneback` ใน OpenCV) เหมาะกับการรันบน Raspberry Pi 5 CPU แบบ real-time

## 2.5 Combined Local-Global (CLG) Method — เทคนิคขั้นสูง

งานของ Bruhn และ Weickert (2005) เสนอการรวมจุดแข็งของทั้ง 2 วิธี: ใช้ structure tensor แบบ Lucas-Kanade (ทนต่อ noise ดีกว่า) ผสมกับ global smoothness ของ Horn-Schunck (ให้ผลแบบ dense และเติมเต็มบริเวณที่ไม่มี texture ได้) งานวิจัยนี้แสดงให้เห็นว่า Horn-Schunck ทำงานได้ดีกว่า Lucas-Kanade ในการ "เติมเต็ม" (filling-in effect) พื้นที่ไร้ texture — นี่คือหลักฐานเชิงวิชาการที่สนับสนุนว่าสำหรับ ควัน/ฝุ่นที่มักไร้ texture ชัดเจน วิธี Horn-Schunck หรือ CLG อาจให้ผลลัพธ์ที่น่าเชื่อถือกว่า Lucas-Kanade ล้วนๆ

## 2.6 ข้อควรระวังสำคัญ: เปลวไฟละเมิด Brightness Constancy Assumption

**นี่คือประเด็นที่ลึกและสำคัญมาก**: งานวิจัยเรื่อง Optical Flow Estimation for Flame Detection ชี้ให้เห็นปัญหาพื้นฐานว่า classical optical flow (LK, HS, Farneback ทั้งหมด) มีปัญหาเมื่อนำมาใช้กับเปลวไฟโดยตรง เพราะเปลวไฟเป็นวัตถุ**เรืองแสงในตัวเอง (self-luminous)** และมีลักษณะโปร่งแสง ทำให้ความสว่างของ "จุดเดียวกัน" ไม่คงที่ระหว่างเฟรมจริงๆ (ขัดกับสมมติฐาน brightness constancy ที่ optical flow ทุกวิธีต้องพึ่งพา) งานวิจัยนี้จึงเสนอวิธีใหม่ที่ออกแบบมาเฉพาะสำหรับไฟ ได้แก่ Optimal Mass Transport (OMT) ซึ่งจำลองการเคลื่อนที่ของ "มวลแสง" แทนที่จะสมมติว่าความสว่างคงที่

**ผลกระทบเชิงปฏิบัติต่อโครงงานของพี่**: 
- สำหรับ **ควัน** (ซึ่งเป็นอนุภาคที่สะท้อน/บดบังแสงแบบ passive มากกว่าจะเรืองแสงเอง) → optical flow แบบดั้งเดิม (LK/HS/Farneback) ยังใช้ได้ผลดีพอสมควร ตามที่งานวิจัยส่วนใหญ่ (Springer, IEEE ที่อ้างถึงด้านล่าง) ใช้กันจริง
- สำหรับ **ตัวเปลวไฟโดยตรง** → ผลลัพธ์จาก optical flow แบบดั้งเดิมอาจมี noise/ความคลาดเคลื่อนสูงกว่าที่คาด ควรตีความผลลัพธ์อย่างระมัดระวัง หรือพิจารณาใช้ optical flow เป็น feature เสริมสำหรับพื้นที่ควันเป็นหลัก มากกว่าพึ่งพาเป็นหลักฐานหลักสำหรับตัวเปลวไฟ

## 2.7 Feature ที่สกัดได้จริงสำหรับแยกฝุ่น/ควัน/ไฟ (พร้อมสูตร)

กำหนดให้ $V(x,y) = (u(x,y), v(x,y))$ คือ dense flow field ที่คำนวณได้ภายใน ROI ที่ YOLO ตรวจพบ

### Feature 1 — Divergence (การขยาย/หดตัว)
$$\nabla \cdot V = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$
ค่า divergence เป็นบวกแสดงว่าพื้นที่กำลัง "ขยายตัวออก" (source-like) ซึ่งสอดคล้องกับพฤติกรรมของกลุ่มควัน/ไฟที่ฟุ้งกระจายขยายตัวตามเวลา

### Feature 2 — Curl / Vorticity (การหมุนวน)
$$\nabla \times V = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$
วัดองค์ประกอบการหมุนของ flow — เกี่ยวข้องกับพฤติกรรม vortex ring ที่ขอบของ buoyant plume (กลุ่มควันที่ลอยตัวขึ้นจากความร้อนมักเกิดการหมุนวนที่ขอบ ตามหลักพลศาสตร์ของไหล)

### Feature 3 — Directional Histogram ด้วย Circular Statistics
คำนวณมุมของแต่ละ flow vector: $\theta(x,y) = \text{atan2}(v,u)$ แล้วสร้าง histogram แบบวงกลม เพราะมุมเป็นข้อมูลเชิงวงกลม (circular data) การหาค่าเฉลี่ยแบบเส้นตรงธรรมดาใช้ไม่ได้ ต้องใช้ **circular mean**:

$$\bar{\theta} = \text{atan2}\left(\sum_i \sin\theta_i, \sum_i \cos\theta_i\right)$$

และ **mean resultant length** ($R \in [0,1]$) เพื่อวัดความ "เกาะกลุ่ม" ของทิศทาง:

$$R = \frac{1}{n}\sqrt{\left(\sum_i \cos\theta_i\right)^2 + \left(\sum_i \sin\theta_i\right)^2}$$

ถ้า $R$ สูง (ใกล้ 1) แปลว่าทิศทางการเคลื่อนที่เกาะกลุ่มชัดเจน (เช่น พุ่งขึ้นทิศเดียว → บ่งชี้การลอยตัวจากความร้อนของไฟ/ควัน) ถ้า $R$ ต่ำ แปลว่าทิศทางกระจัดกระจายไม่มีทิศเด่น (อาจเป็นลมพัดฝุ่นแบบสุ่ม หรือความปั่นป่วนสูง)

### Feature 4 — Turbulence Energy และความเชื่อมโยงกับ Flame Flicker Frequency
คำนวณขนาด flow เฉลี่ยของ ROI ในแต่ละเฟรม $\bar{|V|}(t)$ แล้วเก็บเป็น time series ตลอด sliding window จากนั้นคำนวณ:
- **Variance**: $\sigma^2_{flow} = \text{Var}(\bar{|V|}(t))$ — ยิ่งสูงยิ่งปั่นป่วน
- **Power Spectral Density (PSD)** ผ่าน FFT ของ time series นี้ — เพื่อตรวจสอบว่ามี peak ความถี่ที่ใกล้เคียง ~10 Hz หรือไม่ (เชื่อมโยงโดยตรงกับที่คุยกันเรื่อง Nyquist และ flame flicker frequency ก่อนหน้านี้ — นี่คือวิธีพิสูจน์เชิงปริมาณว่าที่ 15-30 FPS ระบบจับสัญญาณ flicker นี้ได้จริงหรือไม่)

## 2.8 ตัวอย่าง Pipeline ที่งานวิจัยใช้จริง (สำหรับอ้างอิง)
งานวิจัยที่ตรงกับแนวทางของพี่ที่สุด ใช้ Lucas-Kanade optical flow คำนวณบน "candidate region" ที่คัดกรองมาแล้วด้วย color-based rule (HSV) ก่อน จากนั้นสกัด motion feature จากผล optical flow ไปใช้แยกควันออกจากวัตถุเคลื่อนไหวอื่นๆ ด้วย neural network — เป็นรูปแบบ pipeline "กรองก่อนด้วยกฎง่ายๆ (สี) แล้วค่อยคำนวณ feature ที่แพงกว่า (optical flow) เฉพาะจุดที่น่าสงสัย" ซึ่งเหมาะกับข้อจำกัด compute บน embedded system อย่างมาก และสอดคล้องกับแนวทาง 2-tier ที่เราคุยกันไปก่อนหน้านี้ (ตรวจจับหยาบก่อน แล้วค่อยวิเคราะห์ละเอียด)

อีกงานวิจัยหนึ่งเสนอการรวม optical flow feature (จากทั้ง Farneback แบบ dense และ Lucas-Kanade แบบ sparse) เข้ากับ Local Binary Pattern (LBP) ซึ่งเป็น texture feature แบบคลาสสิก แล้วป้อนเข้า SVM classifier — แสดงให้เห็นว่าการรวม motion feature กับ texture feature ให้ผลดีกว่าการใช้ motion หรือ texture เพียงอย่างเดียว ซึ่งเป็นแนวทางที่พี่สามารถผนวกเข้ากับ mask จาก YOLO11-seg ได้ (คำนวณ LBP หรือ texture feature อื่นบนพื้นที่ mask ที่ตรวจพบ)

## 2.9 บทความ/งานวิจัยแนะนำสำหรับศึกษาต่อ

| หัวข้อ | ลิงก์ |
|---|---|
| **Optical Flow Estimation for Flame Detection in Videos** — อธิบายปัญหา brightness constancy ของเปลวไฟและเสนอ Optimal Mass Transport method โดยเฉพาะ | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4000537/ |
| **Unsupervised Segmentation of Fire and Smoke from Infra-Red Videos** — ใช้ Horn-Schunck optical flow เป็น feature หลักสำหรับ segment ไฟ/ควัน | https://arxiv.org/pdf/1909.12937 |
| **Video Fire Smoke Detection Using Motion and Color Features** (Fire Technology, Springer) — pipeline ที่ใกล้เคียงกับแนวทางของพี่มากที่สุด (HSV candidate region + Lucas-Kanade + BPNN classifier) | https://link.springer.com/article/10.1007/s10694-009-0110-z |
| **Optical Flow Feature Based for Fire Detection on Video Data** (IEEE) — รวม Optical Flow (Farneback/LK) กับ LBP texture feature ผ่าน SVM | https://ieeexplore.ieee.org/document/8970957/ |
| **Lucas/Kanade Meets Horn/Schunck: Combining Local and Global Optic Flow Methods** (Bruhn & Weickert) — งานวิจัยขั้นสูงเรื่อง CLG method ที่รวมจุดแข็งของทั้ง 2 วิธีคลาสสิก | https://www.mia.uni-saarland.de/Publications/bruhn-ijcv05c.pdf |
| **Determining Optical Flow** (Horn & Schunck, 1981) — ต้นฉบับ paper คลาสสิกที่เป็นรากฐานของวิธี Horn-Schunck | ค้นหาชื่อ "Horn BKP, Schunck BG (1981) Determining optical flow. Artificial Intelligence 17:185-304" ในฐานข้อมูลมหาวิทยาลัย |

---

## บันทึกท้ายเอกสาร

ทั้ง 2 เทคนิคนี้เป็น "ชั้นการวิเคราะห์เพิ่มเติม" ที่ต่อยอดจากสถาปัตยกรรมหลักที่สรุปไว้ในเอกสารก่อนหน้า (`สรุปแนวคิดเชิงเทคนิค_Fire_Detection_System.md`) — Dempster-Shafer ใช้ในขั้นตอน **Sensor Fusion / Risk Score calculation** (แทนที่ weighted sum แบบง่าย) ส่วน Optical Flow ใช้ในขั้นตอน **Spatial-Temporal Analysis** (ข้อ 5.4 ในเล่ม) เป็น feature เสริมจากภาพ ควบคู่กับผลตรวจจับของ YOLO11-seg

แนะนำให้เริ่มจากการ implement เวอร์ชันง่ายที่สุดก่อน (Dempster's rule ดั้งเดิม + Farneback optical flow ผ่าน OpenCV) แล้วค่อยเพิ่มความซับซ้อน (two-level fusion, CLG method) เมื่อพบว่าเวอร์ชันง่ายมีข้อจำกัดจริงจากการทดลอง — ตรงกับหลักการ iterative design ที่ดีสำหรับโครงงานที่มีกรอบเวลาจำกัด
