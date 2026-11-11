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
| **2026-11-11 17:28** | *(Tối ưu hóa Hybrid Local/Cloud)* | `data/dsp_preprocessing.py`, `data/ptbxl_dataset.py`, `scripts/prepare_cache.py`, `train.py`, `test_local.py` | **1. Vector hóa DSP:** Thay vòng lặp 12 kênh bằng `uniform_filter1d(mode='nearest')` và `signal.filtfilt(..., axis=0)`.<br>**2. Tiền xử lý 1 lần & Cache RAM:** Nạp `ptbxl_preprocessed_250hz.npz` (167,53 MB) giảm tốc độ huấn luyện từ **270s/epoch xuống ~3s/epoch** trên CPU local.<br>**3. Checkpointing & Metric minh bạch:** Lưu `checkpoint_last.pth` mỗi epoch (`--resume`), tính thêm **ROC-AUC chính xác bằng Wilcoxon-Mann-Whitney** trong pure NumPy, bổ sung tham số `--alpha`, `--gamma` phục vụ đối sánh có kiểm soát. | **Chứng minh tương đương toán học Vector hóa:** Sai số cực đại giữa DSP tuần tự cũ và DSP vector hóa mới đo trên 50 bản ghi Fold 9 là **`5,96e-08`** (nhỏ hơn epsilon máy của `float32`).<br>**10/10 Suite Test PASS** kèm PyTorch Autograd Backward cục bộ. |

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

