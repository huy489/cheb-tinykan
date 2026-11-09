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
│   │   └── PRD-01-CHEB-TINYKAN.md # Bản đặc tả sản xuất v1.0, 9 ACs & 5 Edge Cases
│   ├── bugs/                      # Nhật ký lỗi & phản hồi từ QA
│   ├── qa/                        # Kế hoạch kiểm thử & Biên bản nghiệm thu (Sign-off)
│   └── docs/                      # Tài liệu kỹ thuật & Giải phẫu mã nguồn
│       └── ARCHITECTURE_DEEPDIVE.md # Bản vẽ kỹ thuật toàn cảnh: DSP, Layers & KAN Activations
├── firmware/                      # Động cơ C99 nhúng tối ưu hóa ARM CMSIS-NN (__SMLAD)
│   ├── cheb_tinykan_cmsis.h       # Khai báo cấu trúc, giao diện Q1.15 & bẫy phòng vệ SQI
│   ├── cheb_tinykan_cmsis.c       # C99 chuẩn, bộ đệm Ping-Pong tĩnh (Peak RAM 4.22 KB)
│   └── test_firmware.c            # Test harness C99 đo độ trễ vi giây & kiểm định bit-exact
├── models/                        # Kiến trúc mạng nơ-ron PyTorch & Định lượng KAN
│   ├── cheb_kan.py                # Tầng Cheb-KAN bậc 4, truy hồi 3 phần tử, L1 Sparsification
│   ├── model.py                   # 1D-CNN Tokenizer + Bottleneck KAN + 5 Sigmoid Heads
│   ├── quantization.py            # Mô phỏng Q1.15, FakeQuant, L1 Edge Pruning & C Header Export
│   └── symbolic_distillation.py   # Chuyển đổi đại số đa thức chính tắc & báo động lâm sàng
├── data/                          # Xử lý dữ liệu & Chuỗi DSP tại biên
│   ├── dsp_preprocessing.py       # Bộ lọc SWMA (W=151), Butterworth IIR filtfilt, SQI & R-Slicer
│   ├── easi_dower.py              # Ma trận Dower 12x3 phục vụ Ablation Study rút gọn đạo trình
│   ├── ptbxl_dataset.py           # Native int16 binary loader, downsampling 250Hz, Fold 1-10
│   └── scp_mapping.py             # Ánh xạ SCP-ECG 5 phân vùng giải phẫu (NORM, AMI, IMI, LMI, SMI)
├── losses/                        # Hàm mất mát tối ưu hóa biên
│   └── focal_loss.py              # Multi-Label Focal Loss (gamma=2.0, alpha=0.25)
├── train.py                       # Script huấn luyện & đối chuẩn Inter-patient 10-fold
├── test_local.py                  # Bộ kiểm thử tự động toàn diện mở rộng (9/9 tests passed)
├── colab_sync.py                  # Cầu nối Service Account tự động kéo kết quả từ Drive
└── README.md                      # Giới thiệu tổng quan workspace
```

---

## 🎯 Trạng thái Hiện tại
- **Giai đoạn:** `[IN_DEV]` (Đã hoàn thiện trọn vẹn Phase 1, Phase 2, Phase 3 và Phase 4 tại local, **tuân thủ chỉ thị chưa push lên Git**).
- **Kiểm định chất lượng:** Bộ kiểm định `test_local.py` đã vượt qua **9/9 bài kiểm tra** với độ chính xác 100%:
  1. Toán học truy hồi Chebyshev 3 phần tử (Sai số < 0.05% trên miền $[-1, 1]$).
  2. Ánh xạ đa nhãn lâm sàng SCP-ECG (Tách biệt 5 đầu ra độc lập).
  3. Đọc dữ liệu nhị phân PTB-XL Fold 10 & Slicer nhịp (180 mẫu/nhịp).
  4. Hạch toán ngân sách tham số TinyML (<12,000 tham số sau khi tỉa L1 70%).
  5. Ma trận đại số đóng chuyển đổi hệ đa thức Chebyshev sang đơn thức chính tắc.
  6. Bộ lượng tử hóa Q1.15 & thuật toán cắt tỉa L1 KAN Edge Pruning.
  7. Bộ tái tạo 12 đạo trình EASI Dower (Ablation Study AC-09).
  8. Mô-đun chưng cất ký hiệu & sinh chuỗi cảnh báo lâm sàng cho bác sĩ tim mạch.
  9. Biên dịch và thực thi nhị phân động cơ C99 CMSIS-NN (Độ trễ 0.20 ms/nhịp, Peak RAM 4.22 KB).

---

## 🔗 Liên kết Nhanh
- [📋 Kanban Board](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/BOARD.md)
- [📘 Bản Vẽ Kiến Trúc Toàn Cảnh (DSP & Layers)](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/docs/ARCHITECTURE_DEEPDIVE.md)
- [📋 Bản Đặc tả Sản xuất PRD-01](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/prd/PRD-01-CHEB-TINYKAN.md)
- [🤝 Biên bản Đồng thuận Khoa học Tam bên](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/research/CONSENSUS-01-CHEB-TINYKAN.md)
