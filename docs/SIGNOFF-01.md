# ✅ QA SIGNOFF: CHEB-TINYKAN (TOPIC 1 — SOFTWARE, ALGORITHM & CORTEX-M4 ELF)

- **Sign-off By:** QA/QC Lead Engineer
- **Reviewed PRD:** [`PRD-01-CHEB-TINYKAN.md`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/.company/prd/PRD-01-CHEB-TINYKAN.md) (`v1.1-Synced`)
- **Date:** 2026-11-11
- **Final Result:** `[CONDITIONAL SIGN-OFF — SOFTWARE & CORTEX-M4 ELF VERIFIED; PHYSICAL BOARD BENCH PENDING HARDWARE]`

---

## 1. Acceptance Criteria Verification Matrix (Run #4 + ARM Cortex-M4F ELF)

| Criteria ID | Mô tả tiêu chí | Kết quả thực đo | Trạng thái | Ghi chú kiểm định |
| :--- | :--- | :--- | :---: | :--- |
| **AC-01** | Inter-patient Fold 10 (`2.198` ca, `0` rò rỉ); `Macro-AUC >= 0.93`; biên Pareto F1 | Zero-leakage (`0` shared patients); **Run #4 Multi-Beat `Macro-AUC = 0,9447`** (`Single = 0,9302`); **`Macro-F1 = 69,73%`** (vượt `ResNet1D` 548K params: `68,37%`) | **✅ AUC & Pareto PASS** *(⚠️ Gap vs 89% F1 lý tưởng)* | Phản ánh trung thực giới hạn bài toán đa nhãn PTB-XL ở `<12K` params |
| **AC-02** | Sensitivity / Specificity 5 vùng giải phẫu trên Fold 10 | Multi-beat Run #4 `@Calib`: `AMI Sens=74,18%` (`@0.50: 77,45%`), `IMI Sens=72,56%`, `SMI Sens=72,66%`, `LMI Sens=47,01%`, **`Macro-Spec=92,96%`** | **⚠️ GAP vs trần lý tưởng** | Đạt độ đặc hiệu rất cao (`AMI 96,25%`, `SMI 97,22%`) |
| **AC-03** | Active `< 12,000` params; Flash `< 45 KB` (FP32/Q15) / `< 20 KB` (INT8 nén) | **`10.387` active params** (`70%` tỉa); `arm-none-eabi-size`: **`.text + .rodata = 38.316 B (37,42 KB < 45 KB)`** | **✅ PASS** | Đo trực tiếp bằng `arm-none-eabi-gcc 14.3.1 -Os -Werror` |
| **AC-04** | Peak Activation `< 6.0 KB`; System SRAM `< 14.0 KB` | Peak Ping-Pong **`4.320 B (4,22 KB)`**; ELF `.data + .bss` = **`12.968 B (12,66 KB < 14,0 KB)`** | **✅ PASS** | Zero `malloc`/`free`; `.data = 0 B` |
| **AC-05** | Host C99 `< 1 ms`; STM32F411 hardware `< 4.0 ms`/nhịp | Host C99 `cheb_tinykan_infer_calibrated`: **`0,199 ms/nhịp`** (Run #4) / `0,311 ms` (Run #2); binary có lệnh phần cứng **`smlad`** | **✅ Host & ELF** *(⏳ GPIO Bench)* | DWT `CYCCNT` đã gắn sẵn trong `main_stm32f411.c`; đo GPIO/mJ thực thực hiện khi nạp mạch |
| **AC-06** | ESP32-S3 BLE/OLED showcase (`< 1.8 ms`, `< 50 ms`) | Host GCC `-Werror` & logic test `cheb_esp32_format_ble_oled` (`0xCB` BLE + `EDGE-01/05` OLED) PASS | **✅ Host Dry-Run** *(⏳ Board Bench)* | Chưa có `xtensa-esp32s3-elf-gcc` / devkit ESP32-S3 cắm trên máy |
| **AC-07** | Chưng cất ký hiệu đại số & ST V1–V4 (`< 2.0%`) | Sai số đại số **`8,94e-08`** (Run #4) / `1,19e-07` (Run #2); ca `AMI #1912` báo động `93,1%–95,2%` | **✅ PASS** | [`docs/ac_verification_report_run4.json`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/docs/ac_verification_report_run4.json) |
| **AC-08** | Độ chính xác C99 Q-format vs PyTorch FP32 (500 nhịp Fold 10, `< 1.5%`, 0 đảo nhãn) | Run #4: Mean **`0,0251%`**, 99th **`0,1172%`**, Max **`0,1927%`**, **`0/2500` đảo nhãn** | **✅ PASS** | Vượt xa ngưỡng `< 1,50%` |
| **AC-09** | Đối sánh suy hao EASI Dower toàn bộ Fold 10 (`2.198` bản ghi) | Run #4: `AUC 0,9302 -> 0,9188 (-1,14%)`; Run #2: `AUC 0,9264 -> 0,9087 (-1,77%)` | **✅ PASS** | Đủ bảng số liệu & MAE V1–V4 cho bài báo |

---

## 2. Test Execution Summary
- **Python Verification (`test_local.py`):** **11 / 11 Suites PASSED** (Toán học Chebyshev, SCP-ECG, PTB-XL Fold 10 2.198 ca, TinyML Budget, Symbolic M, Q1.15/Pruner, EASI, Bedside Alerts, C99 Firmware + ESP32-S3 compile, PyTorch Forward/Backward, Multi-Beat `EDGE-03` Tukey + `EDGE-05` + Baselines).
- **C99 Firmware Harness (`test_firmware.c`):** **6 / 6 Suites PASSED** sạch sẽ dưới cả **`gcc -O2 -Wall -Wextra -Werror -fsanitize=address,undefined`** lẫn **`gcc -O3 -Wall -Wextra -Werror -std=c99`**.
- **ARM Cortex-M4F Cross-Build (`arm-none-eabi-gcc 14.3.1`):** Biên dịch thành công `stm32f411_cheb.elf` với 0 cảnh báo, kiểm chứng mã máy `smlad`, đo Flash `37,42 KB` & System RAM `12,66 KB` (`docs/embedded_build_report.json`).
- **Open Hardware / Cloud Items Recorded Transparently:**
  1. Đo `DWT->CYCCNT` / GPIO + điện năng thực tế trên **STM32F411RE Nucleo** & **ESP32-S3 DevKit**.
  2. Chạy `InceptionTime` (`457K` params, `131,45s/epoch` trên CPU) qua [`scripts/train_colab.sh`](file:///home/pd/hcmus_queue/lab_dientuysinh/project/exe/topic1/scripts/train_colab.sh) trên Colab T4 GPU khi có Compute Units.
