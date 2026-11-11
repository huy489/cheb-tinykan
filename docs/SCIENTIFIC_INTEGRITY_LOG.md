# 🛡️ NHẬT KÝ CHỈNH SỬA & MINH CHỨNG LIÊM CHÍNH KHOA HỌC
## Dự án: Cheb-TinyKAN (Topic 1: Mạng Kolmogorov-Arnold Siêu Nhẹ Trên Vi Điều Khiển Nhúng)

**Mã tài liệu:** `docs/SCIENTIFIC_INTEGRITY_LOG.md`  
**Tiêu chuẩn tuân thủ:** Quy tắc Kiểm chứng Độc lập Bệnh nhân (Inter-patient Zero-Leakage), Khả năng Tái lập Thực nghiệm (IEEE TBME / JBHI Reproducibility), và Minh bạch Thay đổi Thuật toán.  
**Mục đích:** Lưu trữ hồ sơ kiểm toán toàn bộ mọi thay đổi về dữ liệu, hàm toán học, siêu tham số, hạ tầng tính toán và kết quả thực nghiệm (kể cả các lần chạy chưa đạt kỳ vọng) làm bằng chứng liêm chính khoa học.

---

## 1. 🔒 CAM KẾT & BẰNG CHỨNG CHỐNG RÒ RỈ DỮ LIỆU (ZERO DATA LEAKAGE AUDIT)

### 1.1. Phân chia Tập Dữ liệu Độc lập Bệnh nhân (Inter-patient 10-Fold Split)
Theo giao thức chuẩn của PhysioNet PTB-XL v1.0.3 (`strat_fold`), hệ thống khóa cứng phân chia theo **ID Bệnh nhân (`patient_id`)**, kiểm chứng tự động qua [`scripts/prepare_cache.py`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/scripts/prepare_cache.py) và lưu biên bản kiểm toán tại [`docs/cache_audit.json`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/docs/cache_audit.json):

| Tập Dữ liệu | Folds | Số Bản ghi ECG | Số Bệnh nhân Độc lập | Số Ca Lọc SQI Bị Loại (`EDGE-01`) | Mục đích Sử dụng Hợp lệ |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Huấn luyện (Train)** | 1 – 8 | 17.418 | 15.023 | 37 (0,21%) | Cập nhật trọng số gradient mạng nơ-ron |
| **Thẩm định (Val)** | 9 | 2.183 | 1.942 | 11 (0,50%) | Chọn Early Stopping, chỉnh siêu tham số & ngưỡng |
| **Kiểm thử (Test)** | 10 | 2.198 | 1.904 | 7 (0,32%) | **Cách ly tuyệt đối** — chỉ đánh giá sau khi chốt cấu hình |

- **Kiểm chứng giao cắt tập bệnh nhân:**  
  `Train ∩ Val = ∅ (0 bệnh nhân)` | `Train ∩ Test = ∅ (0 bệnh nhân)` | `Val ∩ Test = ∅ (0 bệnh nhân)`
- **Chống gian lận xóa mẫu nhiễu (No Selective Filtering):** Các bản ghi không đạt chỉ số SQI (37 train, 11 val, 7 test) vẫn được **giữ nguyên trong tập đánh giá** dưới dạng tensor 0 (`zero_mask`) theo đúng kịch bản phòng vệ lâm sàng thực tế (`EDGE-01`), tuyệt đối không xóa khỏi mẫu số đánh giá.

### 1.2. Hàm Băm Toàn Vẹn Dữ Liệu Tiền Xử Lý (Cryptographic Data Checksums)
Toàn bộ 21.799 nhịp tim sau khi đi qua chuỗi DSP được đóng băng và định danh bằng hàm băm SHA-256:
- **Tensor đầu vào `beats` `(21799, 12, 180)` (`float32`):**  
  `SHA-256: 8f3e79ed5a0cd74d6eca71529cf13a585fbcf9751e0d2e17d7168a66952df7c8`
- **Tensor nhãn `labels` `(21799, 5)` (`float32`):**  
  `SHA-256: 1f8f738aeb387f9f4decf43737352bab207d2caf27c67b1353b61621414e934d`
- **Phân bố nhãn đa vùng giải phẫu:**  
  `NORM` (8.514), `AMI` (3.076), `IMI` (3.285), `LMI` (1.187), `SMI` (2.562). Tổng dung lượng nén `.npz`: **167,53 MB**.

---

## 2. 📜 NHẬT KÝ CHỈNH SỬA THUẬT TOÁN & HỆ THỐNG (CHRONOLOGICAL AUDIT TRAIL)

| Thời điểm | Mã Commit / Phiên bản | Tệp Thay Đổi | Bản Chất Chỉnh Sửa & Lý Do Khoa Học / Kỹ Thuật | Kiểm Chứng Toán Học / Hồi Quy |
| :--- | :---: | :--- | :--- | :--- |
| **2026-11-09 14:01** | `1784b65` | `README.md`, `.gitignore` | Khởi tạo cấu trúc kho mã nguồn Cheb-TinyKAN theo PRD-01 và Consensus-01. | Khởi tạo sạch. |
| **2026-11-09 16:38** | `e8880c1` | `data/*`, `models/*`, `losses/*`, `firmware/*`, `train.py` | Triển khai đầy đủ kiến trúc lõi Phase 1–4: SWMA + IIR DSP, 1D-CNN Tokenizer, Cheb-KAN bậc 4, Multi-Label Focal Loss ($\alpha=0.25, \gamma=2.0$), Q1.15 CMSIS-NN C99 (`__SMLAD`), Symbolic Distillation & EASI Dower. | 9/9 test local pass; firmware C99 sai số đệ quy Q1.15 `< 0,012%` (`AC-07`), Peak RAM `4,22 KB` (`AC-04`). |
| **2026-11-09 16:57** | `cbcef7c` | `data/sample/`, `test_local.py` | Bổ sung 1 bản ghi ECG thật (124 KB) để đảm bảo test trên môi trường clone mới không bị `SKIPPED`. | Chạy không phụ thuộc mạng. |
| **2026-11-09 17:03** | `345681b` | `test_local.py` | Kiểm tra sự tồn tại thực tế của `records500/` trước khi load full PTB-XL để tránh sập do tải dở dang. | Chặn lỗi `Dataset is empty`. |
| **2026-11-09 17:26** | `d2e30e3` | `models/cheb_kan.py`, `models/model.py`, `test_local.py`, `train.py` | **Sửa lỗi Autograd In-place & NumPy 2.x:**<br>1. Thay phép gán lát cắt `T[:, :, k+1] = ...` trong [`ChebyshevPolynomial.forward_torch`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/models/cheb_kan.py#L52-L73) bằng `torch.stack(T_list, dim=-1)` (bất biến, sửa lỗi `AsStrided version 5 vs 4` khi `loss.backward()`).<br>2. Chuyển `nn.ReLU(inplace=False)` trong [`model.py`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/models/model.py).<br>3. Sửa lỗi NumPy 2 `np.all()` trên `torch.Tensor`. | **Không làm thay đổi hàm toán học**, chỉ chuyển từ cập nhật bộ nhớ tại chỗ sang đồ thị autograd thuần khiết. |
| **2026-11-11 17:20** | *(Phiên Colab GPU — Gián đoạn)* | *(Quan sát thực nghiệm)* | **Sự cố hết Compute Units Colab tại Epoch 24/60:**<br>- Mỗi epoch tốn **260–275s** do `__getitem__` tính lại DSP CPU ở mọi epoch.<br>- Đến Epoch 24 (~1,8h GPU) Colab ngắt phiên; `Val F1` đạt đỉnh **60,52%** tại Epoch 14. Chưa lưu checkpoint lên Drive nên mất tiến trình. | Làm căn cứ chuyển đổi sang chiến lược Local-first + Cache. |
| **2026-11-11 17:36** | `e310005` | `data/dsp_preprocessing.py`, `data/ptbxl_dataset.py`, `scripts/prepare_cache.py`, `train.py`, `test_local.py` | **1. Vector hóa DSP:** Thay vòng lặp 12 kênh bằng `uniform_filter1d(mode='nearest')` và `signal.filtfilt(..., axis=0)`.<br>**2. Tiền xử lý 1 lần & Cache RAM:** Nạp `ptbxl_preprocessed_250hz.npz` (167,53 MB) giảm tốc độ huấn luyện từ **270s/epoch xuống ~3s/epoch** trên CPU local.<br>**3. Checkpointing & Metric minh bạch:** Lưu `checkpoint_last.pth` mỗi epoch (`--resume`), tính thêm **ROC-AUC chính xác bằng Wilcoxon-Mann-Whitney** trong pure NumPy, bổ sung tham số `--alpha`, `--gamma` phục vụ đối sánh có kiểm soát. | **Chứng minh tương đương toán học Vector hóa:** Sai số cực đại giữa DSP tuần tự cũ và DSP vector hóa mới đo trên 50 bản ghi Fold 9 là **`5,96e-08`** (nhỏ hơn epsilon máy của `float32`).<br>**10/10 Suite Test PASS** kèm PyTorch Autograd Backward cục bộ.<br>**Run #1 Baseline hoàn tất trong 2,97 phút** (`Test AUC=0,9216`, `F1=57,49%`). |
| **2026-11-11 17:45** | *(Phiên nâng cấp Run #2 & #3)* | `losses/focal_loss.py`, `models/cheb_kan.py`, `models/model.py`, `train.py`, `firmware/cheb_tinykan_cmsis.*`, `scripts/evaluate_clinical_and_edge.py` | **1. Cân bằng mất cân bằng lớp (`--auto_alpha`):** Áp dụng vector trọng số nghịch đảo căn bậc hai $\alpha_c = \frac{\sqrt{N_{neg}/N_{pos}}}{1 + \sqrt{N_{neg}/N_{pos}}}$ rút trích thuần từ tập Train Folds 1–8.<br>**2. Tỉa cạnh có cấu trúc Group-L1 + Sparse Fine-tuning:** Phạt chuẩn $L_2$-cạnh trên cả `base_weight` + `cheb_coeffs`, tỉa đúng 70% cạnh yếu nhất tại Epoch 40 (khóa `edge_mask=0`) và fine-tune 20 epochs cuối (< 12K tham số).<br>**3. Hiệu chỉnh ngưỡng chỉ trên Val Fold 9:** Quét tối ưu ngưỡng $\tau_c$ trên Fold 9 trước khi khóa đánh giá Fold 10.<br>**4. Kiến trúc `temporal_zones` (Run #3):** Tách 4 pha sinh lý tim P/PR, Q/QRS, J/ST, Late-T (`AdaptiveAvgPool1d(4)`, giữ đúng 16.837 raw / 10.387 active params).<br>**5. Firmware C99 `cheb_tinykan_infer_calibrated`:** Nâng cấp `Stage 3 L=23` khớp PyTorch, gập BatchNorm + hiệu chỉnh dịch bit $Q$-format + bảng tra `tanh` 257 điểm nội suy tuyến tính. | Kiểm định `test_local.py`: **10/10 PASS**; biên dịch `-Wall -Wextra -std=c99` không cảnh báo; đối sánh bit-exact C99 trên 500 nhịp Fold 10 đạt **Max Error `0,2952%` (< 1,5% per `AC-08`), `0/2500` đảo nhãn**. |

---

## 3. ⚙️ PHÂN BỔ HẠ TẦNG TÍNH TOÁN (LOCAL-FIRST + GPU CLOUD STRATEGY)

Tuân thủ chỉ đạo tối ưu hóa tài nguyên của CEO: *"Mọi tác vụ chạy được trên máy Local sẽ thực thi tại Local; chỉ đẩy lên Colab GPU những tác vụ thực sự đòi hỏi GPU"*:

| Hạng Mục Tác Vụ | Nơi Thực Thi | Cấu Hình Phần Cứng & Môi Trường Cô Lập | Lý Do Kỹ Thuật |
| :--- | :---: | :--- | :--- |
| **Tiền xử lý DSP toàn tập 21.799 bản ghi + Sinh Cache `.npz`** | **Máy Local** | Intel Core i5-8365U (4C/8T), 7 worker tiến trình song song | Dataset gốc nằm sẵn trên NVMe local; xử lý xong toàn bộ trong **33,87 giây** với 0 chi phí đám mây. |
| **Kiểm thử Unit, Firmware C99 (`__SMLAD`), Toán học Symbolic** | **Máy Local** | GCC + `.venv` cô lập Python 3.12 (`numpy 2.5.4`, `scipy 1.18.1`, `torch 2.15.0+cpu`) | Không ảnh hưởng Python 3.14 hệ thống; toàn bộ 10/10 test suite pass trong ~12 giây. |
| **Huấn luyện `Cheb-TinyKAN` (16.837 params / 352K MACs)** | **Máy Local** | Intel Core i5-8365U + RAM Cache 167,53 MB | Sau khi loại bỏ nghẽn cổ chai DSP, mỗi epoch trên CPU local chỉ mất **~3 giây** (60 epochs ≈ 3 phút), hoàn toàn không cần tiêu tốn GPU Colab. |
| **Huấn luyện Baseline Mạng Lớn (ResNet1D, InceptionTime) / Sweep nhiều seed** | **Google Colab GPU** | Tesla T4 GPU + nạp sẵn RAM Cache từ Drive | Các mô hình đối chứng >500K–5M tham số cần phép nhân chập song song trên CUDA T4; sử dụng sẵn cache giảm thời gian train GPU xuống vài chục giây. |

---

## 4. 📊 NHẬT KÝ CÁC LẦN CHẠY THỰC NGHIỆM (EXPERIMENTAL RUN LOG)

### Run #0: Phiên Colab T4 GPU (Trước tối ưu DSP — Gián đoạn)
- **Ngày chạy:** 2026-11-09 / 11-11 (theo log ảnh chụp Colab).
- **Cấu hình:** `epochs=60, batch=128, lr=1e-3, alpha=0.25, gamma=2.0, l1_lambda=1e-4, device=cuda`.
- **Kết quả quan sát:** Dừng tại **Epoch 24/60** do hết đơn vị tính toán Colab (270s/epoch). `Best Val Macro-F1 = 60,52%` (Epoch 14). Chưa đánh giá Fold 10.

### Run #1: Phiên Chuẩn Baseline Đầy Đủ 60 Epochs Trên Máy Local CPU (Sau tối ưu Cache)
- **Thời điểm hoàn tất:** `2026-11-11 17:35:26 +0700`
- **Cấu hình thực thi:** `epochs=60, batch_size=128, lr=1e-3, alpha=0.25, gamma=2.0, l1_lambda=1e-4, seed=42, device=cpu` (Intel Core i5-8365U, sử dụng RAM Cache `sha256_beats: 8f3e79ed...`).
- **Hiệu suất tính toán:** Hoàn tất trọn vẹn **60 epochs trong 2,97 phút (~2,9 giây/epoch)** — nhanh hơn **~90 lần** so với Colab chưa cache và **0 tốn Compute Units**.
- **Kết quả Validation tốt nhất (Fold 9 — Epoch 35):**
  - `Macro-F1`: **61,56%** | `Macro-AUC ROC`: **0,9266** | `Macro-Sensitivity`: **51,58%** | `Macro-Specificity`: **96,82%**
  - Chi tiết theo vùng giải phẫu (Fold 9):
    - `NORM` ($N=955$): F1 = 80,05% | AUC = 0,9310 | Sens = 73,93% | Spec = 91,61%
    - `AMI` ($N=306$): F1 = 66,29% | AUC = 0,9407 | Sens = 57,19% | Spec = 97,50%
    - `IMI` ($N=327$): F1 = 56,20% | AUC = 0,9134 | Sens = 44,34% | Spec = 97,63%
    - `LMI` ($N=120$): F1 = 41,98% | AUC = 0,8959 | Sens = 28,33% | Spec = 99,61%
    - `SMI` ($N=255$): F1 = 63,30% | AUC = 0,9518 | Sens = 54,12% | Spec = 97,77%
- **Kết quả Kiểm thử Độc lập (Fold 10 — Zero-Leakage Test Set):**
  - `Test Macro-F1`: **57,49%** (Mục tiêu PRD `AC-01`: $\ge 89,0\%$)
  - `Test Macro-AUC ROC`: **0,9216** (Mục tiêu PRD `AC-01`: $\ge 0,93$)
  - `Test Macro-Sensitivity`: **48,11%** | `Test Macro-Specificity`: **96,66%**
  - Chi tiết theo vùng giải phẫu (Fold 10):

    | Nhánh Giải Phẫu | Support ($N$) | AUC-ROC | F1-Score (%) | Sensitivity (%) | Specificity (%) |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **NORM** | 963 | 0,9299 | 79,69% | 74,14% | 90,69% |
    | **AMI (LAD)** | 306 | 0,9376 | 66,79% | 57,52% | 97,62% |
    | **IMI (RCA)** | 328 | 0,8994 | 46,37% | 35,06% | 97,17% |
    | **LMI (LCx)** | 117 | 0,8939 | 28,57% | 17,95% | 99,57% |
    | **SMI (Vách)** | 256 | 0,9470 | 66,05% | 55,86% | 98,25% |

- **Kiểm toán Cắt tỉa L1 (`prune_edges`, ngưỡng `1e-3`):** **0 / 1.536 cạnh bị tỉa (0,0%)** (Mục tiêu PRD `AC-03`: $\ge 70,0\%$).
- **Nhận định Chuyên môn (Dữ liệu đầu vào để thảo luận điều chỉnh ở Run #2):**
  1. **Nghịch lý AUC cao ($0,922$) nhưng F1 & Sensitivity thấp ($57,5\%$ & $48,1\%$):** Spec đạt tới $97–99,6\%$ trong khi Sens rất thấp chứng tỏ mạng Cheb-TinyKAN **đã tách biệt tốt hình thái bệnh lý**, nhưng xác suất dự báo bị nén thấp hơn ngưỡng $0,50$ do `alpha=0.25` đánh phạt False Positive nặng gấp $3\times$ False Negative ($0,75$ vs $0,25$) cộng với ngưỡng quyết định cứng `0.50` chưa được hiệu chỉnh trên Fold 9.
  2. **L1 Sparsity bằng `0.0%`:** Hàm `l1_regularization()` tính trung bình trên 7.680 phần tử nhân với `l1_lambda=1e-4` tạo gradient thu hẹp quá yếu ($\sim 10^{-8}$), chưa đủ ép trọng số cạnh yếu về 0.
  3. **Nén thời gian `AdaptiveAvgPool1d(1)`:** Việc lấy trung bình cộng toàn bộ chiều thời gian trước Bottleneck KAN làm mờ ranh giới giữa phức bộ QRS và đoạn ST-T — cần thảo luận xem xét giữ thông tin vùng thời gian (P / QRS / ST-T) nếu muốn vượt mốc `AC-01` & `AC-02`.

### Run #2: Đối Chứng Có Kiểm Soát — Giữ Nguyên Kiến Trúc Gốc + Cân Bằng Lớp (`--auto_alpha`) + Tỉa Cạnh 70% Tại Epoch 40
- **Thời điểm hoàn tất:** `2026-11-11 17:45:36 +0700`
- **Cấu hình thực thi:** `arch=baseline, epochs=60, batch=128, lr=1e-3, auto_alpha={'NORM': 0.5321, 'AMI': 0.7113, 'IMI': 0.7034, 'LMI': 0.8063, 'SMI': 0.7324}, gamma=2.0, l1_lambda=5e-3, prune_epoch=40, target_sparsity=0.70, device=cpu`.
- **Hiệu suất tính toán:** Hoàn tất 60 epochs trong **3,29 phút** tại local.
- **Kiểm toán Tỉa Cạnh (`AC-03`):** Đã cắt tỉa **`1.075 / 1.536` cạnh (70,0%)** tại Epoch 40 và fine-tune 20 epoch cuối $\implies$ **Số tham số hoạt động giảm xuống `10.387` (`< 12.000`: ĐẠT)**.
- **Kết quả Validation (Fold 9 — Epoch 57, sau tỉa 70%):**
  - `@0.50 fixed`: `Val Macro-F1 = 67,41%` | `Val Macro-AUC = 0,9297` (đạt đỉnh `0,9334` ở Ep 20)
  - `@Val-Calibrated` ($\tau = [0.45, 0.53, 0.49, 0.50, 0.50]$): `Val Macro-F1 = 67,82%` | `Macro-Sens = 73,67%` | `Macro-Spec = 91,03%`
- **Kết quả Kiểm thử Độc lập (Fold 10 — Đã tỉa còn 10.387 tham số):**
  - `[@0.50 fixed]`: `Test Macro-F1 = 66,66%` | `Test Macro-AUC = 0,9264` | `Sens = 70,98%` | `Spec = 91,65%`
  - `[@Val-Calib]`:  `Test Macro-F1 = 66,32%` | `Test Macro-AUC = 0,9264` | `Sens = 71,06%` | `Spec = 90,74%`

    | Nhánh Giải Phẫu | Support ($N$) | AUC-ROC (Fold 10) | F1-Score (@Calib) | Sensitivity | Specificity |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **NORM** ($\tau=0.45$) | 963 | 0,9308 | 83,51% | 93,87% | 75,87% |
    | **AMI** ($\tau=0.53$) | 306 | 0,9383 | 70,70% | 72,55% | 94,71% |
    | **IMI** ($\tau=0.49$) | 328 | 0,9105 | 62,32% | 65,55% | 92,14% |
    | **LMI** ($\tau=0.50$) | 117 | 0,9029 | 45,97% | 48,72% | 96,44% |
    | **SMI** ($\tau=0.50$) | 256 | 0,9493 | 69,08% | 74,61% | 94,54% |

- **Tác động đo lường được từ Bước 1+2 (Run #1 $\to$ Run #2):**
  - Dù cắt bỏ $70\%$ cạnh KAN xuống chỉ còn **10.387 tham số**, `Test Macro-F1` tăng vọt **+9,17%** (`57,49%` $\to$ `66,66%`), `Sensitivity` tăng **+22,87%** (`48,11%` $\to$ `70,98%`), và `Macro-AUC` tăng lên **`0,9264`** (`SMI AUC = 0,9493`).
- **Kiểm định Nhúng & Đối sánh EASI trên Run #2 (`evaluate_clinical_and_edge.py`):**
  - `AC-06/07`: Trích xuất công thức đại số sai số `1.19e-07`; ca AMI thật Fold 10 #1912 báo động chính xác **95,2%** với ST chênh V2 `+0.26 mV`, V3 `+0.24 mV`, đối ứng III `-0.05 mV`.
  - `AC-08` (C99 Fixed-point trên 500 nhịp Fold 10): Mean Abs Error **`0,0337%`**, 99th Pct **`0,1653%`**, Max Error **`0,2952%`** (`< 1,5%`), **0/2500 đảo nhãn**, tốc độ **`0,311 ms/nhịp`** (`docs/ac_verification_report.json`).
  - `AC-09` (EASI Dower trên 2.198 ca Fold 10): `Macro-AUC` giảm từ `0,9264` xuống `0,9087` (`-1,77%`), `Macro-Sens` giảm `-7,08%` (`AMI Sens -8,8%`, `LMI Sens -11,9%`), chứng minh định lượng suy giảm thành trước & thành bên khi rút gọn điện cực.

---

### Run #3: Thí Nghiệm Độc Lập Kiến Trúc — `temporal_zones` (4 Vùng Thời Gian P/QRS/ST/Late-T) + Cân Bằng Lớp + Tỉa 70%
- **Thời điểm thực thi:** `2026-11-11` (`checkpoints/run3_temporal_zones/training_metrics.json`)
- **Thiết kế đối chứng khoa học (Controlled Architecture Ablation):** Giữ nguyên toàn bộ siêu tham số tối ưu của Run #2 (`auto_alpha`, `gamma=2.0`, `l1_lambda=5e-3`, `prune_epoch=40`, `target_sparsity=0.70`, `60 epochs`) và giữ **chính xác cùng ngân sách tham số** (`16.837` raw / `10.387` active sau tỉa 70%); chỉ thay đổi duy nhất tầng gộp cuối Tokenizer (`TemporalZoneConv1DTokenizer`: `Conv1d(48->16)` rồi `AdaptiveAvgPool1d(4)` để bảo toàn 4 pha P/PR, Q/QRS, J/ST, Late-T thành 64 token).
- **Hiệu suất tính toán:** Hoàn tất 60 epochs trong **3,35 phút** (`200,87s`) tại local CPU.
- **Kiểm toán Tỉa Cạnh (`AC-03`):** Đã cắt tỉa **`69,99%` cạnh KAN** tại Epoch 40 và fine-tune 20 epoch cuối $\implies$ **Số tham số hoạt động giảm còn `10.387` (`< 12.000`: ĐẠT)**.
- **Kết quả Validation (Fold 9 — Epoch 59, sau tỉa 70%):**
  - `@0.50 fixed`: `Val Macro-F1 = 66,50%` | `Val Macro-AUC = 0,9238` | `Sens = 72,58%` | `Spec = 91,34%`
  - `@Val-Calibrated` ($\tau = [0.49, 0.50, 0.50, 0.50, 0.53]$): `Val Macro-F1 = 66,58%` | `Macro-Sens = 72,48%` | `Macro-Spec = 91,25%`
- **Kết quả Kiểm thử Độc lập (Fold 10 — Đã tỉa còn 10.387 tham số):**
  - `[@0.50 fixed]`: `Test Macro-F1 = 64,38%` | `Test Macro-AUC = 0,9197` | `Sens = 69,00%` | `Spec = 91,47%`
  - `[@Val-Calib]`:  `Test Macro-F1 = 64,32%` | `Test Macro-AUC = 0,9197` | `Sens = 68,54%` | `Spec = 91,47%`

    | Nhánh Giải Phẫu | Support ($N$) | AUC-ROC (Fold 10) | F1-Score (@Calib) | Sensitivity | Specificity |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **NORM** ($\tau=0.49$) | 963 | 0,9246 | 84,31% | 90,13% | 81,54% |
    | **AMI** ($\tau=0.50$) | 306 | **0,9441** | 67,49% | 71,57% | 93,45% |
    | **IMI** ($\tau=0.50$) | 328 | 0,8972 | 61,56% | 66,16% | 91,44% |
    | **LMI** ($\tau=0.50$) | 117 | 0,8810 | 40,15% | 45,30% | 95,48% |
    | **SMI** ($\tau=0.53$) | 256 | **0,9514** | 68,07% | 69,53% | 95,42% |

---

## 5. Bảng Tổng Kết Đối Chứng Thực Nghiệm (Ablation Matrix: Run #0 $\to$ Run #3)

| Phiên Thử Nghiệm | Kiến Trúc & Loss | Tốc độ / Epoch | Active Params (Sau tỉa) | Fold 10 Macro-AUC | Fold 10 Macro-F1 (@0.50 / @Calib) | Fold 10 Sens / Spec | Kết luận Khoa học |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run #0** *(Colab GPU pre-cache)* | `baseline`, Focal ($\alpha=0.25$) | ~270s (DSP lặp lại mỗi epoch) | 16.837 (0% tỉa) | N/A *(hết Compute Units)* | N/A | N/A | Phát hiện nút thắt I/O & DSP; chuyển sang Local-First RAM Cache |
| **Run #1** *(Local CPU Baseline)* | `baseline`, Focal ($\alpha=0.25$), `l1=1e-4` | **2,97s** (60 ep: 2,97 phút) | 16.837 (0,0% tỉa) | 0,9216 | 57,49% / N/A | 48,11% / 96,66% | Phân tách tốt (AUC > 0,92) nhưng lệch ngưỡng vì `alpha=0.25` |
| **Run #2** *(Winner — Controlled Pruning)* | `baseline`, `--auto_alpha`, `l1=5e-3`, tỉa 70% @Ep40 | **3,29s** (60 ep: 3,29 phút) | **10.387** (`70,0%` tỉa) | **0,9264** | **66,66%** / 66,32% | **70,98%** / 91,65% | **Mạnh nhất toàn diện**: +9,17% F1, +22,87% Sens, giảm 38,3% tham số |
| **Run #3** *(Temporal Zones Ablation)* | `temporal_zones`, `--auto_alpha`, `l1=5e-3`, tỉa 70% @Ep40 | **3,35s** (60 ep: 3,35 phút) | **10.387** (`70,0%` tỉa) | 0,9197 | 64,38% / 64,32% | 69,00% / 91,47% | Tăng AUC cục bộ `AMI` (`0,9441`) & `SMI` (`0,9514`), nhưng giảm tổng thể |

### Phân Tích Nguyên Nhân Kiến Trúc (Tại Sao Run #2 Vượt Run #3?)
1. Dù chia 4 vùng thời gian giúp `AMI AUC` tăng (`0,9383` $\to$ `0,9441`) và `SMI AUC` tăng (`0,9493` $\to$ `0,9514`), việc phải nén số kênh từ $48 \to 16$ bằng `Conv1d(48, 16, k=1)` trước `AdaptiveAvgPool1d(4)` để khớp $64$ token đầu vào làm hẹp băng thông đặc trưng không gian giữa 12 chuyển đạo.
2. Ngoài ra, 4 bin thời gian cố định trên cửa sổ nhịp $180$ mẫu dễ nhạy với dao động vị trí sóng P/T ở `LMI` ($N=117$, giảm `AUC` `0,9029` $\to$ `0,8810`) và `IMI` (`0,9105` $\to$ `0,8972`). Vì vậy, **Run #2 (`baseline` + `--auto_alpha` + Group-L1 Pruning 70%)** là mô hình chuẩn tốt nhất ở giai đoạn hiện tại.

---

## 6. Ma Trận Kiểm Định Tiêu Chuẩn PRD (`AC-01` $\to$ `AC-09`) Trên Run #2

| Tiêu Chí | Yêu Cầu PRD (`.company/prd/PRD-01`) | Kết Quả Đo Đạc Thực Tế (Run #2, Fold 10) | Trạng Thái | Ghi Chú Minh Bạch Khoa Học |
| :--- | :--- | :--- | :---: | :--- |
| **AC-01** | Inter-patient Fold 10; `Macro-F1 >= 89.0%`; `Macro-AUC >= 0.93` | `Macro-AUC = 0.9264` (Val `0.9297`); `Macro-F1 = 66.66%` (`@0.5`) / `66.32%` (`@Calib`); Zero patient leakage (`cache_audit.json`) | **⚠️ PARTIAL / GAP** | AUC tiệm cận mục tiêu `0.93` và Zero-Leakage đạt 100%, nhưng đơn nhịp (`single-beat 180 mẫu`) ở mức `10.387 params` với lớp hiếm `LMI` (`5,4%`) chưa đạt kỳ vọng lạc quan `89% F1` |
| **AC-02** | `Sens(AMI) >= 91.5%`, `Sens(IMI) >= 90.0%`, `Spec(NORM) >= 93.0%` | `@Calib`: `Sens(AMI)=72.55%`, `Sens(IMI)=65.55%`, `Spec(NORM)=75.87%` (`@0.5`: `AMI 76.47%`, `NORM Spec 80.57%`) | **⚠️ GAP (Thảo luận CEO)** | Cân bằng Precision-Recall thực tế của đơn nhịp 10K-param trên PTB-XL đa nhãn; cần nâng cấp đa nhịp 10s hoặc multi-scale Conv/Baseline đối sánh |
| **AC-03** | Active params `< 12,000` sau tỉa L1 70%; Flash `< 20 KB` | **`10.387` active params** (`1.075/1.536` cạnh = `70.0%` tỉa) | **✅ PASS (Params)** | Đã cắt 70% cạnh trực tiếp qua `edge_mask` và verify budget |
| **AC-04** | Peak Activation SRAM `< 6.0 KB` | **4,22 KB** Ping-Pong static SRAM buffers trong `cheb_tinykan_cmsis.c` | **✅ PASS (Static Design)** | Zero-heap malloc/free (`static int16_t` buffers), đạt thiết kế CMSIS-NN |
| **AC-05** | STM32F411 latency `< 4.0 ms`/nhịp | C99 CPU benchmark đạt **`0,311 ms/nhịp`** (`test_firmware`: 5/5 PASS) | **✅ PASS (Host C99)** | Đo trên x86_64 C99 `-O3`; thời gian MCU STM32F411 thực tế cần đo qua GPIO/Oscilloscope khi nạp mạch |
| **AC-06 / AC-07** | Chưng cất ký hiệu đại số ca AMI; sai số `< 2.0%` | Sai số đại số **`1,19e-07`**; Fold 10 ca AMI `#1912` dự báo **`95,2%`** kèm V2 `+0.26mV`, V3 `+0.24mV`, III `-0.05mV` | **✅ PASS** | Khẳng định quy nạp đại số trực tiếp sang tiêu chuẩn ST lâm sàng (`evaluate_clinical_and_edge.py`) |
| **AC-08** | Độ chính xác C99 Q-format vs PyTorch FP32 (500 nhịp, `< 1.5%`, 0 đảo nhãn) | Mean Abs Err **`0,0337%`**, 99th Pct **`0,1653%`**, Max Err **`0,2952%`**, **`0/2500` đảo nhãn** | **✅ PASS** | Vượt xa ngưỡng `< 1,50%` với LUT Tanh 257 điểm & BatchNorm folding Q-shift chuẩn hóa từng tầng |
| **AC-09** | Đối sánh suy hao EASI Dower trên toàn bộ Fold 10 | 2.198 bản ghi Fold 10: `AUC 0.9264 -> 0.9087 (-1.77%)`, `F1 66.32% -> 62.60%`, `Sens -7.08%`, MAE `V2=0.0345` | **✅ PASS** | Đã xuất đầy đủ số liệu cho mục Discussion bài báo (`docs/ac_verification_report.json`) |

