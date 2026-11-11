# 🩺 Workspace: Topic 1 - Cheb-TinyKAN

> **Đề tài:** Mạng Kolmogorov-Arnold Đa thức Trực giao Chebyshev Siêu nhẹ Chưng cất Hình thái Ký hiệu cho Định vị Nhồi máu Cơ tim Thời gian thực trên Vi điều khiển Nhúng  
> **Tên tiếng Anh (Chuẩn IEEE):** *Cheb-TinyKAN: An Ultra-Lightweight Orthogonal Polynomial Kolmogorov-Arnold Network with Symbolic Morphometric Distillation for Real-Time Myocardial Infarction Localization on Cortex-M Microcontrollers*

---

## 📂 Cấu trúc Workspace

```
exe/topic1/
├── .company/                      # Bộ máy điều hành dự án ảo
│   ├── BOARD.md                   # Bảng Kanban theo dõi tiến độ tổng thể (Status: IN_DEV)
│   ├── research/                  # Hồ sơ nghiên cứu chuyên sâu & đối kháng
│   │   ├── RES-01-KAN-TINYEDGE.md # Nghiên cứu 7 biến thể KAN & thiết kế kiến trúc
│   │   └── CONSENSUS-01-CHEB-TINYKAN.md # Biên bản đồng thuận R&D + CA Red Team
│   ├── prd/                       # Yêu cầu kỹ thuật & Nghiệp vụ (BA Department)
│   │   └── PRD-01-CHEB-TINYKAN.md # Bản đặc tả sản xuất v1.1-Synced (AC-01..09 & EDGE-01..05)
│   ├── bugs/                      # Nhật ký lỗi & phản hồi từ QA
│   ├── qa/                        # Kế hoạch kiểm thử & Biên bản nghiệm thu (Sign-off)
│   └── docs/                      # Tài liệu kỹ thuật & Giải phẫu mã nguồn
│       ├── ARCHITECTURE_DEEPDIVE.md # Bản vẽ kỹ thuật toàn cảnh: DSP, Layers & KAN Activations
│       └── PROJECT_DEEPDIVE.md    # Sách giải phẫu mã nguồn 4 tầng (/explain)
├── firmware/                      # Động cơ C99 nhúng tối ưu hóa ARM CMSIS-NN (__SMLAD)
│   ├── cheb_tinykan_cmsis.h       # Interface C99, inline ASM smlad, EDGE-05 & Multi-beat
│   ├── cheb_tinykan_cmsis.c       # C99 calibrated + Q1.15, Ping-Pong RAM 4.22 KB, LUT 257-pt
│   ├── cheb_tinykan_weights_run2.h# Trọng số Run #2 gập BatchNorm đóng băng cho Flash MCU
│   ├── test_firmware.c            # Bộ kiểm định 6/6 C99 harness (ASan/UBSan & -O3 -Werror)
│   └── targets/                   # Mã nguồn biên dịch mục tiêu phần cứng nhúng
│       ├── stm32f411/             # Bare-metal Cortex-M4F ELF (arm-none-eabi-gcc + DWT cycles)
│       └── esp32s3/               # ESP-IDF BLE GATT + OLED showcase component
├── models/                        # Kiến trúc mạng nơ-ron PyTorch & Định lượng KAN
│   ├── cheb_kan.py                # Tầng Cheb-KAN bậc 4, truy hồi 3 phần tử, Group-L1 & edge_mask
│   ├── model.py                   # 1D-CNN Tokenizer (`baseline` / `temporal_zones`) + ChebTinyKAN
│   ├── baselines.py               # Mô hình đối chứng IEEE (MLPTiny, ResNet1D, InceptionTime1D)
│   ├── quantization.py            # Mô phỏng Q1.15, FakeQuant, L1 Edge Pruning & C Header Export
│   └── symbolic_distillation.py   # Ma trận đại số M, cảnh báo lâm sàng & phân luồng EDGE-05
├── data/                          # Xử lý dữ liệu & Chuỗi DSP tại biên
│   ├── dsp_preprocessing.py       # SWMA (W=151), IIR filtfilt, SQI (EDGE-01/02), Multi-beat (EDGE-03)
│   ├── easi_dower.py              # Ma trận Dower 12x3 phục vụ Ablation Study AC-09
│   ├── ptbxl_dataset.py           # Native int16 loader, RAM Cache đơn nhịp & đa nhịp Q1.15
│   └── scp_mapping.py             # Ánh xạ SCP-ECG 5 phân vùng giải phẫu (NORM, AMI, IMI, LMI, SMI)
├── docs/                          # Tài liệu kỹ thuật, kiểm toán JSON & Liêm chính Khoa học
│   ├── PROJECT_DEEPDIVE.md        # Sách giải phẫu mã nguồn 4 tầng (/explain)
│   ├── ARCHITECTURE_DEEPDIVE.md   # Bản vẽ kỹ thuật toàn cảnh: DSP, Layers & KAN Activations
│   ├── DEV_JOURNAL.md             # Nhật ký thay đổi mã nguồn chi tiết của Dev
│   ├── SCIENTIFIC_INTEGRITY_LOG.md# Nhật ký minh chứng liêm chính khoa học & SHA-256 checksums
│   ├── ac_verification_report.json# Kiểm định tự động AC-06/07/08/09
│   └── cache_audit.json           # Biên bản kiểm toán Zero Data Leakage across Folds 1-10
├── losses/                        # Hàm mất mát tối ưu hóa biên
│   └── focal_loss.py              # Multi-Label Focal Loss (hỗ trợ scalar alpha & vector auto_alpha)
├── scripts/                       # Tự động hóa tiền xử lý, kiểm định & Colab GPU
│   ├── prepare_cache.py           # Tiền xử lý DSP đơn nhịp (33.9s cho toàn bộ 21,799 bản ghi)
│   ├── prepare_multibeat_cache.py # Tiền xử lý 8 nhịp/bản ghi Q1.15 (38.8s, 173,414 nhịp sạch)
│   ├── evaluate_clinical_and_edge.py # Chưng cất ký hiệu, C99 500 nhịp (AC-08) & EASI (AC-09)
│   ├── evaluate_multibeat_consensus.py # Đánh giá đồng thuận đa nhịp 10s trên Fold 9 -> Fold 10
│   ├── train_baselines.py         # Benchmark đối chứng MLP-Tiny / ResNet1D / InceptionTime
│   ├── export_c_weights.py        # Xuất trọng số Run #2 sang Flash header C99
│   └── train_colab.sh             # Script Colab T4 GPU cho các mô hình CNN nặng
├── train.py                       # Pipeline huấn luyện chuẩn hóa (Folds 1-8 / 9 / 10, --auto_alpha)
├── test_local.py                  # Bộ kiểm thử tự động toàn diện mở rộng (11/11 tests passed)
├── colab_sync.py                  # Cầu nối Service Account tự động kéo kết quả từ Drive
└── README.md                      # Giới thiệu tổng quan workspace
```

---

## 🎯 Trạng thái Hiện tại
- **Chiến lược thực thi:** **Hybrid Local-First + Cloud GPU** — Toàn bộ DSP (đơn nhịp 33,9s / đa nhịp 38,8s), kiểm thử PyTorch (`test_local.py`: **11/11 suites**), kiểm thử C99 (`test_firmware.c`: **6/6 suites** với ASan/UBSan & `-Wall -Wextra -Werror`), biên dịch ARM Cortex-M4 (`arm-none-eabi-gcc`: `.text=38,316 B`, `.bss=12,968 B = 12.66 KB`), và huấn luyện `Cheb-TinyKAN` / `MLP-Tiny` chạy trực tiếp tại local; chỉ định tuyến các mạng nặng như `InceptionTime` (~131s/epoch trên CPU) & `ResNet1D` lên Colab T4 GPU.
- **Thành tựu nổi bật trên Zero-Leakage Fold 10 (Run #2 — `10,387` active params):**
  - **Đơn nhịp (180 mẫu):** `Macro-AUC = 0.9264`, `Macro-F1@0.50 = 66.66%`, `Sens = 70.98%`.
  - **Đồng thuận Đa nhịp 10s (`trimmed_mean_prob` chọn trên Fold 9):** **`Macro-AUC = 0.9360`** (vượt mốc `0.93` của `AC-01`), `Macro-F1@0.50 = 68.77%`.
  - **Chính xác Lượng tử hóa C99 (`AC-08` trên 500 nhịp):** Max Error **`0.2952%`** (`< 1.5%`), **`0/2500` đảo nhãn**, **`0.311 ms/nhịp`**.

---

## 🔗 Liên kết Nhanh
- [📘 Sách Giải Phẫu Mã Nguồn 4 Tầng (Project Deep-Dive)](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/docs/PROJECT_DEEPDIVE.md)
- [🛡️ Nhật ký Chỉnh sửa & Liêm chính Khoa học](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/docs/SCIENTIFIC_INTEGRITY_LOG.md)
- [📜 Nhật ký Kỹ thuật Dev](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/docs/DEV_JOURNAL.md)
- [📘 Bản Vẽ Kiến Trúc Toàn Cảnh (DSP & Layers)](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/docs/ARCHITECTURE_DEEPDIVE.md)
- [📋 Kanban Board](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/BOARD.md)
- [📋 Bản Đặc tả Sản xuất PRD-01 (v1.1-Synced)](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/prd/PRD-01-CHEB-TINYKAN.md)
- [🤝 Biên bản Đồng thuận Khoa học Tam bên](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/research/CONSENSUS-01-CHEB-TINYKAN.md)
