# 📋 PRD-01: CHEB-TINYKAN - PRODUCTION SPECIFICATION & REQUIREMENTS
## Mạng Kolmogorov-Arnold Siêu Nhẹ Đa thức Trực giao Chebyshev Giải trình Lâm sàng trên Vi điều khiển Nhúng

- **Author:** Lead Business Analyst (BA Department)
- **Stakeholders:** CEO (Lab Trưởng), R&D Specialist, CA Technical Critic, Tech Lead (Dev), QA Engineer
- **Status:** `[v1.1-SYNCED - ACTIVE_VERIFICATION]`
- **Version:** v1.1-Synced-Production-Spec
- **Created Date:** 2026-11-09 | **Last Synced:** 2026-11-11
- **Workspace:** `/home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1`
- **Inherited Research & Consensus:** [`RES-01-KAN-TINYEDGE.md`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/research/RES-01-KAN-TINYEDGE.md) | [`CONSENSUS-01-CHEB-TINYKAN.md`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/research/CONSENSUS-01-CHEB-TINYKAN.md)

---

## 1. Executive Summary & Objective

### 1.1. Bối cảnh & Giá trị Cốt lõi
Nhồi máu cơ tim cấp (Acute Myocardial Infarction - AMI) là biến cố tim mạch tối khẩn cấp với tỷ lệ tử vong cao nhất nếu không được can thiệp tái tưới máu trong "khung giờ vàng" (Golden Hour: < 90 phút). Việc phát hiện sớm và định vị chính xác phân vùng hoại tử giải phẫu theo nhánh động mạch vành (LAD - Thành trước/Vách, RCA - Thành dưới, LCx - Thành bên) qua điện tâm đồ ngay tại thiết bị y tế cá nhân hoặc xe cấp cứu có ý nghĩa sinh mạng.

Tuy nhiên, các mô hình Deep Learning hiện tại hoặc là **"hộp đen"** (bác sĩ không thể kiểm chứng lý do chẩn đoán), hoặc **quá nặng** (không thể chạy trên vi điều khiển công suất siêu thấp ARM Cortex-M/ESP32).

### 1.2. Mục tiêu Dự án (Project Objectives)
Xây dựng, tối ưu hóa và kiểm chuẩn hoàn chỉnh hệ thống **Cheb-TinyKAN**:
1. Đạt kích thước siêu thu gọn: **$< 12\text{K}$ tham số hoạt động** (sau cắt tỉa Group-L1 70%), tiêu thụ **$< 14\text{KB}$ SRAM tổng** (Peak Activation $< 6\text{KB}$), **$< 20\text{KB}$ Flash (INT8 nén) / $< 45\text{KB}$ (FP32 hoặc Q1.15 chưa nén)**.
2. Thực thi thời gian thực trên vi điều khiển ARM Cortex-M4/G4 với độ trễ **$< 4.0\text{ ms}$/nhịp**, tiêu thụ năng lượng cực thấp ($\approx 0.19\text{ mJ}$/nhịp), duy trì thời lượng pin $>15$ ngày.
3. Cung cấp **công thức giải thích ký hiệu lâm sàng (Symbolic Morphometric Distillation)** tường minh cho bác sĩ tim mạch về độ chênh đoạn ST và biên độ sóng Q theo từng vùng cơ tim.
4. Đạt chuẩn công bố khoa học tại các tạp chí y sinh Q1 của IEEE (*IEEE TBME / JBHI*), tuân thủ 100% giao thức kiểm chuẩn độc lập bệnh nhân (Inter-patient protocol) trên cơ sở dữ liệu quốc tế PTB-XL (21.799 bản ghi).

---

## 2. Các Quyết định Chiến lược, Căn cứ Khoa học & Tiêu chí Lựa chọn

Căn cứ vào biên bản đồng thuận tam bên [`CONSENSUS-01-CHEB-TINYKAN.md`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/research/CONSENSUS-01-CHEB-TINYKAN.md), 3 quyết định chiến lược cốt lõi được định hình như sau:

### 2.1. Quyết định 1: Phân lớp Chẩn đoán Đa nhãn Phân cấp (Hierarchical Multi-Label Output)
- **Lựa chọn:** Thiết kế **5 nhánh Sigmoid độc lập**:
  $$\mathbf{y} = [P_{Normal}, P_{Anterior}, P_{Inferior}, P_{Lateral}, P_{Septal}]^T \in [0, 1]^5$$
  kết hợp với hàm mất mát **Multi-Label Focal Loss** ($\gamma = 2.0$, cân bằng theo lớp $\alpha_c = \frac{\sqrt{N_{neg}/N_{pos}}}{1 + \sqrt{N_{neg}/N_{pos}}}$ rút trích thuần từ tập Train Folds 1–8 để thay thế $\alpha = 0.25$ cố định) và hiệu chỉnh ngưỡng $\tau_c$ độc lập trên Validation Fold 9.
- **Tiêu chí đưa ra lựa chọn:**
  1. *Tính xác đáng cấp cứu lâm sàng:* Định vị trực tiếp nhánh động mạch vành tắc nghẽn (LAD, RCA, LCx) theo tiêu chuẩn vàng **ESC/ACC/AHA/WHF 4th Universal Definition of MI (Circulation 2018)**.
  2. *Khả năng tương thích ca bệnh phức tạp:* Cho phép cảnh báo đồng thời các ca nhồi máu diện rộng liên vùng (như Anteroseptal hoặc Inferolateral MI).
  3. *Khắc phục mất cân bằng mẫu:* Focal Loss cân bằng `--auto_alpha` ngăn lớp đa số áp đảo các nhánh hiếm như `LMI` (~5,45% trong PTB-XL) mà không làm méo xác suất xuống dưới ngưỡng quyết định.
- **Phương án bị bác bỏ:**
  - *Nhị phân (NORM vs MI):* Bác bỏ vì không có giá trị định vị can thiệp lâm sàng, làm tầm thường hóa đề tài.
  - *Đơn nhãn Softmax 6 lớp:* Bác bỏ vì làm mất thông tin ở các ca bệnh tổn thương đa vùng.
  - *Scalar `alpha=0.25` không hiệu chỉnh:* Đã kiểm chứng thực nghiệm Run #1 bị nén xác suất gây Sensitivity thấp (`48,11%`).

### 2.2. Quyết định 2: Chiến lược Đạo trình Kép (Dual-Input Lead Strategy)
- **Lựa chọn:** 
  - **Trục cốt lõi (Primary Clinical Pipeline):** Sử dụng **12 Đạo trình Tiêu chuẩn thực tế** (Standard 12-Lead) làm tiêu chuẩn vàng kiểm chuẩn.
  - **Trục khảo sát mở rộng (Ablation / Extension Pipeline):** Xây dựng mô-đun phụ trợ **EASI-to-12 Lead Reconstruction** qua ma trận Dower tuyến tính Fixed-point Q1.15 để đo lường độ suy hao lâm sàng.
- **Tiêu chí đưa ra lựa chọn:**
  1. *Tuân thủ Hướng dẫn AHA/ACC 2018:* Tiêu chuẩn khẳng định nhồi máu cơ tim bắt buộc phải dựa trên tối thiểu 2 đạo trình tiếp giáp giải phẫu thực tế.
  2. *Loại trừ sai lệch biên độ đoạn ST:* Nghiên cứu *Drew et al. (Circulation 2004)* chứng minh đạo trình nội suy từ EASI gây sai lệch biên độ ST từ $0.05 - 0.18\text{ mV}$, dễ gây chẩn đoán sai.
  3. *Giá trị Đóng góp Bài báo IEEE:* Định lượng khoa học sự đánh đổi giữa kích thước thiết bị đeo (Wearable 4-electrode patch) và độ chính xác chẩn đoán tim mạch.
- **Phương án bị bác bỏ:**
  - *Chỉ dùng 1 đạo trình (Smartwatch Lead I) hoặc 3 đạo trình chi:* Bác bỏ vì bỏ sót tới $>65\%$ số ca nhồi máu cơ tim thành trước (Anterior MI).

### 2.3. Quyết định 3: Ma trận Triển khai Phần cứng Kép (Dual Hardware Benchmark Matrix)
- **Lựa chọn:** 
  - **Chuẩn Đối sánh Khoa học (Primary Clinical Baseline):** **STM32F411RE (ARM Cortex-M4 @ 100MHz)** viết bằng C99 thuần với thư viện CMSIS-NN, tối ưu hóa bằng lệnh tích lũy nhân kép DSP `SMLAD`.
  - **Nguyên mẫu Trình diễn IoMT (Connected Showcase Prototype):** **ESP32-S3 (Xtensa Dual-Core @ 240MHz)** tích hợp phần cứng vector SIMD PIE và kết nối không dây BLE 5.0 / màn hình OLED 0.96 inch.
- **Tiêu chí đưa ra lựa chọn:**
  1. *Tiêu chuẩn Thiết bị Y tế (IEC 62304 / ISO 14971):* Dòng vi điều khiển STM32 đạt chứng chỉ an toàn chức năng SIL2/SIL3 và tiêu chuẩn phần mềm y tế quốc tế.
  2. *Thời lượng Pin Siêu dài:* STM32F411 ngủ $99.5\%$ thời gian ($<20\ \mu\text{A}$), duy trì pin $>15$ ngày; loại trừ hoàn toàn nguy cơ nhiễu sóng cao tần RF vào mạch khuếch đại điện tim vi mô $\mu\text{V}$.
  3. *Tính Trực quan cho Trình diễn Sản phẩm:* ESP32-S3 cung cấp bằng chứng trực quan về khả năng truyền thông IoMT không dây đến điện thoại bác sĩ.
- **Phương án bị bác bỏ:**
  - *Chỉ dùng duy nhất ESP32-S3:* Bác bỏ vì rủi ro tiêu hao pin nhanh (dòng $45-80\text{ mA}$) và thiếu các chứng chỉ an toàn thiết bị y tế chuẩn mực.

---

## 3. Target Users & User Stories

### 3.1. Đối tượng Người dùng & Bảng Giá trị
| Đối tượng | Mong muốn cốt lõi | Nỗi đau hiện tại | Cheb-TinyKAN giải quyết |
| :--- | :--- | :--- | :--- |
| **Bác sĩ Cấp cứu / Bác sĩ Tim mạch** | Cảnh báo tức thì vị trí nhồi máu cơ tim kèm bằng chứng định lượng đoạn ST-T rõ ràng để quyết định kích hoạt Cath-Lab khẩn cấp. | Không tin tưởng các mô hình AI "hộp đen" chỉ đưa ra con số xác suất mơ hồ. | Xuất trực tiếp công thức đại số đóng: Báo cáo mức độ chênh đoạn ST trên các đạo trình thành trước (V1–V4) và độ sâu sóng Q bệnh lý. |
| **Kỹ sư Hệ thống Nhúng & IoT Y sinh** | Mô hình AI phải chạy ổn định độc lập trên vi điều khiển không có GPU, tiêu thụ pin tối thiểu để thiết bị đeo hoạt động nhiều ngày. | Các mạng KAN (B-spline) hoặc Transformer quá nặng, tràn bộ nhớ RAM 128KB, tiêu tốn năng lượng nhanh chóng. | Cắt tỉa 70% cạnh, lượng tử hóa INT8 (Q1.15), tận dụng tập lệnh DSP ARM `SMLAD`, tiết kiệm 99.5% chu kỳ ngủ (Sleep mode). |
| **Bệnh nhân có nguy cơ cao** | Giám sát liên tục 24/7 tại nhà, cảnh báo trước khi cơ tim hoại tử diện rộng mà không bị báo động giả liên tục. | Thiết bị cảnh báo sai khi bệnh nhân cử động nhẹ hoặc đổ mồ hôi. | Bộ lọc thích nghi SWMA và thuật toán Chebyshev triệt tiêu nhiễu trôi đường đẳng điện và gai cơ EMG. |

### 3.2. User Stories
- **US-01 (Giám sát Lâm sàng Minh bạch):**  
  *Là Bác sĩ Tim mạch*, tôi muốn hệ thống xuất ra công thức hình thái học chi tiết (mức độ nâng/hạ đoạn ST tính theo mV và tỷ lệ đóng góp của từng đạo trình), *để tôi có thể thẩm định lâm sàng nhanh chóng trước khi ra y lệnh can thiệp*.
- **US-02 (Cảnh báo Thời gian thực trên Vi điều khiển Tiết kiệm Pin):**  
  *Là Kỹ sư Phần cứng Nhúng*, tôi muốn thuật toán suy luận hoàn tất trong vòng dưới 4ms trên chip STM32F411, *để vi điều khiển có thể lập tức quay lại chế độ tiết kiệm năng lượng (Deep Sleep), kéo dài thời lượng pin hơn 15 ngày*.
- **US-03 (Bảo vệ Chống Tràn số và Treo hệ thống):**  
  *Là Kỹ sư Firmware*, tôi muốn mọi phép tính đa thức được đóng khung nghiêm ngặt trong miền số nguyên cố định Fixed-point Q1.15 và bão hòa biên độ, *để loại trừ 100% nguy cơ tràn số nguyên hoặc phát sinh giá trị NaN làm reset vi điều khiển*.
- **US-04 (Xác thực Không Gian lận Học thuật):**  
  *Là Trưởng nhóm Nghiên cứu (CEO)*, tôi muốn dữ liệu đánh giá mô hình phải được phân chia theo giao thức độc lập bệnh nhân (Inter-patient) 10-fold trên PTB-XL, *để bài báo khoa học xuất bản không bị các reviewer quốc tế bắt bẻ về rò rỉ dữ liệu (Data Leakage)*.

---

## 4. Detailed Functional Requirements & Workflow

```
[12-Lead Raw ECG] ➔ [1. DSP Preprocessing] ➔ [2. 1D-CNN Tokenizer] ➔ [3. Cheb-KAN Bottleneck] ➔ [4. Cheb-KAN Multi-Head Engine] ➔ [5. OLED/BLE Alert]
```

### 4.1. Luồng xử lý chi tiết (Flow of Events)

1. **Giai đoạn 1: Tiếp nhận và Tiền xử lý DSP (On-device DSP Pipeline)**
   - Tiếp nhận tín hiệu ECG 12 đạo trình tiêu chuẩn tại tần số lấy mẫu $f_s = 250\text{ Hz}$.
   - Khử trôi đường đẳng điện (Baseline Wander) bằng thuật toán SWMA (Statistical Weighted Moving Average).
   - Khử nhiễu lưới điện 50Hz/60Hz và nhiễu cơ bằng bộ lọc số IIR Butterworth bậc 2 (0.5 – 45 Hz).
   - Dò tìm đỉnh sóng R bằng thuật toán thích nghi ngưỡng năng lượng cục bộ. Cắt cửa sổ nhịp gồm **180 điểm mẫu** ($[-200\text{ ms}, +520\text{ ms}]$ quanh đỉnh R).
   - Chuẩn hóa biên độ tín hiệu theo từng đạo trình độc lập về dải giá trị nghiêm ngặt $[-1.0, +1.0]$ tương ứng với khoảng biểu diễn số nguyên có dấu Fixed-point Q1.15.

2. **Giai đoạn 2: Trích xuất Đặc trưng Thời gian (Temporal Tokenizer - 1D-CNN)**
   - Tầng Conv1D chuẩn ($k=7, s=2, c_{out}=24$) kết hợp BatchNorm và ReLU: $(180 \times 12) \to (90 \times 24)$.
   - Tầng Depthwise-Separable Conv1D 1 ($k=5, s=2, c_{out}=48$): $(90 \times 24) \to (45 \times 48)$.
   - Tầng Depthwise-Separable Conv1D 2 ($k=5, s=2, c_{out}=64$): $(45 \times 48) \to (23 \times 64)$ (lưu ý `padding=2` PyTorch chuẩn hóa chính xác về `23` bước thời gian).
   - Tầng Global Average Pooling 1D (GAP) tổng hợp dọc trục thời gian, nén tín hiệu thành vector đặc trưng không gian $z \in \mathbb{R}^{64}$.

3. **Giai đoạn 3: Khối Thắt cổ chai KAN Đa thức Chebyshev (Cheb-KAN Bottleneck)**
   - Lớp Cheb-KAN 1 tiếp nhận 64 đầu vào, ánh xạ sang 24 đầu ra ($64 \to 24$).
   - Mỗi hàm kích hoạt cạnh $\phi(z) = w_{base}\tanh(z) + \sum_{k=0}^{4} c_k T_k(z)$ sử dụng đa thức Chebyshev loại 1 bậc 4 ($P=4$).
   - Tính toán cơ sở đa thức theo công thức truy hồi 3 phần tử: $T_0(z)=1, T_1(z)=z, T_{k+1}(z) = 2zT_k(z) - T_{k-1}(z)$.
   - 70% các kết nối cạnh yếu nhất được cắt tỉa cố định thông qua Group-L1 + `edge_mask` (`prune_by_sparsity` tại Epoch 40 rồi sparse fine-tune), giảm tham số hoạt động xuống **`10.387`** (`< 12.000`).

4. **Giai đoạn 4: Tầng Ra Quyết định Đa nhãn & Chưng cất Ký hiệu (Decision Engine)**
   - Lớp Cheb-KAN 2 tiếp nhận 24 nơ-ron từ tầng trước, ánh xạ sang **5 nhánh Sigmoid độc lập**:
     1. $P_{Normal} = \sigma(\text{Head}_{Normal}(z))$
     2. $P_{Anterior} = \sigma(\text{Head}_{AMI}(z))$
     3. $P_{Inferior} = \sigma(\text{Head}_{IMI}(z))$
     4. $P_{Lateral} = \sigma(\text{Head}_{LMI}(z))$
     5. $P_{Septal} = \sigma(\text{Head}_{SMI}(z))$
   - Tự động rút gọn các hàm Chebyshev thành đa thức chính tắc $a_0 + a_1 z + a_2 z^2 + a_3 z^3 + a_4 z^4$ để xuất ra công thức định lượng mức độ chênh đoạn ST và độ sâu sóng Q theo từng vùng cơ tim.

5. **Giai đoạn 5: Đồng thuận Đa nhịp 10s, Đầu ra Lâm sàng & Giao tiếp Ngoại vi**
   - **Chế độ Đơn nhịp & Đồng thuận Đa nhịp 10s (Multi-Beat Rolling Consensus):** Trên MCU, mỗi nhịp 180 mẫu được suy luận tuần tự trong cùng bộ đệm tĩnh `4,22 KB` SRAM; cho phép tổng hợp xác suất nhiều nhịp liên tiếp (`trimmed_mean_prob` chọn trên Fold 9) nhằm triệt tiêu nhiễu ngoại tâm thu hoặc xê dịch nhịp cục bộ.
   - Đóng gói gói tin chẩn đoán: Phân trạng thái theo `EDGE-05` (`NORMAL_SINUS_MONITORING`, `SUSPECTED_ISCHEMIA_MONITOR_TREND` $45\%–65\%$, `CRITICAL_MI_ALERT` $>65\%$), điểm tin cậy và công thức giải trình.
   - Hiển thị trên màn hình OLED tích hợp và phát thanh qua Bluetooth Low Energy (BLE) đến ứng dụng theo dõi.

---

## 5. Defensive Design & Edge Cases (Thiết kế Phòng vệ Toàn diện)

| Mã ngoại lệ | Tình huống phát sinh | Phản ứng phòng vệ của hệ thống |
| :--- | :--- | :--- |
| **EDGE-01** | **Tuột điện cực hoặc Mất tín hiệu đạo trình** (Điện cực lỏng, tín hiệu flatline hoặc nhiễu 50Hz bão hòa). | Mô-đun DSP tính chỉ số chất lượng tín hiệu (SQI). Nếu $\text{SQI} < 0.60$ hoặc biên độ cố định $>0.5\text{s}$, lập tức phát cờ `LEAD_FAULT_DETECTED` (`CHEB_STATUS_LEAD_FAULT`), ngừng suy luận KAN để loại trừ hoàn toàn báo động giả. |
| **EDGE-02** | **Bão hòa bộ khuếch đại & Gai nhiễu lớn** (Phóng tĩnh điện, co giật cơ $>10\text{ mV}$). | Cắt xén biên độ cứng về $[-5.0\text{ mV}, +5.0\text{ mV}]$ tại tiền xử lý; firmware `cheb_verify_sqi_q15` bắt bẫy bão hòa thanh ghi trước flatline (`CHEB_STATUS_SATURATION`) và kẹp `[-32767, 32767]`. |
| **EDGE-03** | **Nhịp tim cực đoan ($>180$ hoặc $<40$ bpm) & Ngoại tâm thu** (Sóng P đè sóng T nhịp trước). | Cố định cửa sổ 180 mẫu quanh đỉnh R ($[-200\text{ ms}, +520\text{ ms}]$). Khi $\text{HR} > 150\text{ bpm}$, áp dụng suy giảm cosine nửa cuối (Tukey right-tail từ $+340\text{ ms}..+520\text{ ms}$) để bảo toàn nguyên vẹn QRS & đoạn ST nhưng triệt tiêu sóng P/T nhịp kế. |
| **EDGE-04** | **Tràn số nguyên trong tính toán Fixed-point Q1.15** (Lũy tích ma trận trọng số $\sum c_k T_k$). | Khai báo biến tích lũy trung gian kiểu `int32_t` / `int64_t`. Sử dụng tập lệnh ARM DSP `SMLAD` và bão hòa chuẩn hóa từng tầng (`cheb_tinykan_infer_calibrated`). |
| **EDGE-05** | **Đoạn ST nằm ở vùng ranh giới mập mờ** ($0.08 - 0.12\text{ mV}$, xác suất $45\% - 65\%$). | Hệ thống chuyển sang trạng thái cảnh báo `SUSPECTED_ISCHEMIA_MONITOR_TREND` (trong cả Python `classify_clinical_state` và C99 `cheb_classify_clinical_state`), theo dõi động học trong 15 phút tiếp theo thay vì kết luận nhồi máu dứt khoát. |

---

## 6. Strict Acceptance Criteria (AC - Tiêu chí Nghiệm thu Bắt buộc)

*Tất cả tiêu chí phải được kiểm thử tự động với kết quả Pass/Fail định lượng, phân tách rõ phần kiểm chứng phần mềm/toolchain tại local và phần đo vật lý trên bo mạch thật.*

| Mã AC | Hạng mục kiểm thử | Điều kiện tiên quyết & Hành động | Tiêu chí Pass định lượng (v1.1-Synced) |
| :--- | :--- | :--- | :--- |
| **AC-01** | **Hiệu năng Đa nhãn Độc lập Bệnh nhân (Inter-patient)** | Đánh giá trên toàn bộ tập kiểm thử độc lập **Fold 10 của PTB-XL v1.0.3 (2.198 bản ghi / 1.904 bệnh nhân, `0` bệnh nhân trùng Train/Val)**. | - **Zero-Leakage:** `Train ∩ Val = Train ∩ Test = Val ∩ Test = ∅`.<br>- **Macro-AUC ROC $\ge \mathbf{0.93}$** (đạt bằng Multi-Beat 10s Consensus hoặc đối chứng Pareto).<br>- **Macro-F1:** Ghi nhận trung thực biên Pareto TinyML `<12K params` so với đối chứng nặng (`ResNet1D`/`InceptionTime`/`MLP-Tiny`); mục tiêu lý tưởng ban đầu $\ge 89.0\%$ được lưu giữ làm chuẩn tham chiếu trần (aspirational ceiling). |
| **AC-02** | **Độ nhạy Định vị Tổn thương Giải phẫu** | Đánh giá riêng biệt từng nhánh Sigmoid trên Fold 10 theo ngưỡng hiệu chỉnh từ Fold 9. | - Báo cáo đầy đủ `Sensitivity`, `Specificity`, `F1`, `AUC` của 5 nhánh (`NORM`, `AMI`, `IMI`, `LMI`, `SMI`), đối sánh Single-Beat vs Multi-Beat 10s vs Baseline nặng (tham chiếu kỳ vọng lâm sàng lý tưởng `AMI >= 91.5%`, `IMI >= 90.0%`, `NORM Spec >= 93.0%`). |
| **AC-03** | **Kích thước Tham số & Bộ nhớ Flash** | Đo tham số hoạt động sau cắt tỉa 70% và đo dung lượng nhị phân `.elf` qua `arm-none-eabi-size` (Cortex-M4F). | - Tham số hoạt động: **$< 12,000\text{ params}$** (Run #2: `10,387`).<br>- Flash `.text + .rodata`: **$< 45\text{ KB}$** cho bảng Q1.15 chưa nén (`< 20\text{ KB}$` khi áp dụng nén thưa INT8). |
| **AC-04** | **Bộ nhớ SRAM Tiêu thụ (Peak & System RAM)** | Kiểm tra bộ đệm Ping-Pong tĩnh (`zero-malloc`) và section `.data + .bss` từ `arm-none-eabi-size`. | - Peak Activation SRAM: **$< 6.0\text{ KB}$** (`4,320 bytes = 4.22 KB`).<br>- Tổng SRAM hệ thống (`.data + .bss` gồm bộ đệm thu 12x180 + Ping-Pong): **$< 14.0\text{ KB}$**. |
| **AC-05** | **Độ trễ Suy luận trên STM32F411RE** | Benchmark C99 `-O3` trên CPU host **+** đo `DWT->CYCCNT` / GPIO Toggle trên STM32F411RE @ 100MHz. | - Host C99 đã verify $< 1.0\text{ ms}$ (`0.311 ms/nhịp`); Target STM32F411 hardware: **$< 4.0\text{ ms}$/nhịp**, Sleep **$\ge 99.5\%$**, năng lượng **$< 0.20\text{ mJ}$/nhịp**. |
| **AC-06** | **Nguyên mẫu Trình diễn IoMT ESP32-S3** | Tích hợp thành phần firmware `esp32s3` đóng gói BLE 5.0 GATT + chuỗi hiển thị OLED (gắn cờ `EDGE-01/05`). | - Firmware target đóng gói chuẩn xác trên host C99; benchmark vật lý trên chip ESP32-S3 @ 240MHz mục tiêu **$< 1.8\text{ ms}$/nhịp**, BLE/OLED **$< 50\text{ ms}$** khi nạp mạch thực tế. |
| **AC-07** | **Chưng cất Công thức Ký hiệu Lâm sàng** | Chuyển đổi đa thức Chebyshev sang chính tắc và sinh cảnh báo ST V1–V4 trên ca AMI thật Fold 10. | - Xuất công thức đại số dạng đóng kèm ST chênh mV theo đạo trình.<br>- Sai số đại số so với mạng Cheb-KAN: **$< 2.0\%$**. |
| **AC-08** | **Độ chính xác Lượng tử hóa C99 Q-Format vs PyTorch FP32** | So sánh trực tiếp xác suất đầu ra giữa C99 `cheb_tinykan_infer_calibrated` và PyTorch FP32 trên 500 nhịp Fold 10. | - Sai số xác suất tương đối cực đại: **$< 1.5\%$**.<br>- **Tuyệt đối 0 trường hợp đảo ngược nhãn chẩn đoán** (`0/2500`). |
| **AC-09** | **Khảo sát Đối sánh Suy hao EASI Dower** | Chạy tái tạo 3-lead EASI $\to$ 12-lead Dower trên toàn bộ 2.198 ca Fold 10. | - Định lượng đầy đủ độ suy giảm `Macro-AUC`, `Macro-F1`, `Sensitivity` từng vùng tổn thương và sai số biên độ sóng V1–V4 cho bài báo IEEE. |

---

## 7. Non-Functional Requirements (Yêu cầu Phi chức năng)

1. **Chuẩn mã hóa Nhúng Y tế (Medical Embedded Coding Standard):**
   - Viết hoàn toàn bằng **chuẩn C99**, tuyệt đối không dùng cấp phát động (`malloc`/`free`) trong vòng lặp thời gian thực để chống phân mảnh bộ nhớ.
   - Tuân thủ tinh thần các quy tắc an toàn phần mềm **MISRA-C:2012** và tiêu chuẩn thiết bị y tế **IEC 62304**.
2. **Khả năng Tái lập Khoa học (Scientific Reproducibility):**
   - Pipeline PyTorch cố định seed ngẫu nhiên (`random_seed = 42`), lưu trữ đầy đủ log huấn luyện và checkpoint trọng số.

---

## 8. Lộ trình Triển khai Phân kỳ (Rollout Roadmap)

- **Tuần 1 - 2 (PyTorch Multi-label & Inter-patient Benchmark):**  
  Xây dựng module PyTorch `cheb_kan.py` hỗ trợ 5 nhánh Sigmoid, Focal Loss, huấn luyện và kiểm chuẩn nghiêm ngặt trên PTB-XL Fold 10.
- **Tuần 3 - 4 (QAT & Sinh mã C99 Fixed-point Q1.15):**  
  Huấn luyện nhận thức lượng tử hóa (QAT), chuyển đổi toàn bộ trọng số sang Q1.15, sinh mã C nhúng tối ưu hóa bằng CMSIS-NN `SMLAD`.
- **Tuần 5 (Kiểm chuẩn Phần cứng STM32F411 & ESP32-S3):**  
  Nạp firmware lên STM32F411RE Nucleo và ESP32-S3 DevKit, đo đạc chu kỳ xung nhịp, Peak RAM, dòng tiêu thụ và kiểm chứng truyền tin BLE.
- **Tuần 6 (Thực nghiệm Cắt bỏ EASI & Hoàn thiện Bản thảo IEEE):**  
  Chạy đối sánh EASI Dower, trích xuất công thức ký hiệu lâm sàng và hoàn thiện bản thảo bài báo gửi *IEEE TBME / JBHI*.

---
*Bản đặc tả đã được hoàn thiện bởi Lead Business Analyst & Product Owner.*  
*Lưu trữ tại kho lưu trữ dự án:* [`PRD-01-CHEB-TINYKAN.md`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/prd/PRD-01-CHEB-TINYKAN.md)
