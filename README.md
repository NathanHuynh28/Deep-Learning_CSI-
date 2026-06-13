# Deep-Learning CSI (Channel State Information)

Dự án nghiên cứu sử dụng Deep Learning để phân loại tư thế người (pose estimation) từ dữ liệu CSI (Channel State Information) - thông tin trạng thái kênh Wi-Fi.

## 📋 Tổng Quan Project

Project này chia thành **2 phần chính**:

### **Phần 1: Tiền Xử Lí Dữ Liệu (Preprocessing)**
- Chuyển dữ liệu Wi-Fi CSI thô thành dữ liệu sẵn sàng cho model
- Pipeline gồm: tách train/val/test, cắt window, loại nhiễu, augmentation, chuẩn hóa
- Output: các file `.npz` chứa dữ liệu đã xử lí

### **Phần 2: Xây Dựng & Huấn Luyện Model (Modeling)**
- Xây dựng các model Deep Learning: GRU, Attention-GRU, Multi-output
- Thực hiện các thử nghiệm có kiểm soát với 35 training runs
- Output: model tốt nhất, metrics đánh giá, dự đoán

---

## 🗂️ Cấu Trúc Thư Mục

```
Deep-Learning_CSI-/
│
├── README.md                          # File này
│
├── Phần 1 tiền xử lí/                 # PHẦN 1: Preprocessing
│   ├── README.md                      # Hướng dẫn chi tiết phần 1
│   ├── workflow.ipynb                 # Notebook chính chạy toàn bộ pipeline
│   ├── split_and_window.py            # Script tách train/val/test và cắt window
│   │
│   ├── Denoise/
│   │   └── CSI_denoise.py             # Loại nhiễu từ CSI signal
│   │
│   ├── Augmentation/
│   │   └── CSI_augment.py             # Augmentation dữ liệu (temporal shift)
│   │
│   ├── Normalization/
│   │   └── CSI_normalize.py           # Chuẩn hóa dữ liệu (normalization)
│   │
│   ├── Dataset/
│   │   └── CSI_dataset.py             # PyTorch Dataset/DataLoader
│   │
│   └── processed/                     # Thư mục output
│       ├── sample_split/              # Danh sách chia train/val/test
│       │   ├── split.csv
│       │   └── report.csv
│       ├── denoised/                  # Dữ liệu đã loại nhiễu
│       │   └── phase_hampel/
│       ├── windows/                   # Dữ liệu cắt window chính
│       │   ├── raw/
│       │   ├── raw_shift/             # Raw + temporal augmentation
│       │   ├── phase_hampel/          # Denoised
│       │   └── phase_hampel_shift/    # Denoised + temporal augmentation
│       └── reports/                   # Báo cáo phân tích
│           └── denoise/
│               └── phase_hampel/
│
├── Phần 2 mô hình/                    # PHẦN 2: Modeling & Training
│   ├── README.md                      # Hướng dẫn chi tiết phần 2
│   ├── workflow_all.ipynb             # ⭐ Notebook chính: chạy 35 training runs
│   ├── workflow.ipynb                 # Reference GRU cơ bản
│   ├── workflow_attention_gru.ipynb   # Reference Attention-GRU
│   ├── workflow_multi_output_best_model.ipynb  # Reference Multi-output
│   │
│   ├── gru_baseline.py                # Model GRU baseline
│   ├── attention_gru.py               # Model Attention-GRU
│   ├── multi_output_attention_gru.py  # Model Multi-output (shared trunk + heads)
│   │
│   ├── train_gru.py                   # Trainer cho GRU
│   ├── train_attention_gru.py         # Trainer cho Attention-GRU
│   ├── train_multi_output_best_model.py  # Trainer cho Multi-output
│   │
│   └── runs/                          # Output sau khi train
│       ├── gru_baseline/
│       ├── attention_gru/
│       └── multi_output_best_model/
│           └── S1_G01_raw_h64_b32_lr001/  # Ví dụ run
│               ├── config.json        # Cấu hình của run
│               ├── metrics.json       # Tổng kết metrics
│               ├── history.csv        # Loss/metrics theo epoch
│               ├── predictions.csv    # Dự đoán trên val/test
│               ├── best_model.pt      # Checkpoint tốt nhất
│               ├── last_model.pt      # Checkpoint cuối cùng
│               └── selection.json     # Thông tin selector (multi-output)

```

---

## 🔄 Luồng Dữ Liệu Tổng Thể

### **Phần 1: Preprocessing**

```
Dataset_CSI_3D_v2/session1/
        ↓
split_and_window.py
        ↓
┌─────────────────────────────────────────┐
│ raw/data.npz                            │
│ train: 652, val: 140, test: 139         │
└─────────────────────────────────────────┘
        ↓
        ├─→ CSI_augment.py ──→ raw_shift/data.npz
        │                      train: 2608, val: 140, test: 139
        │
        └─→ CSI_denoise.py ──→ denoised/phase_hampel/
               ↓
        split_and_window.py
               ↓
        phase_hampel/data.npz
        train: 652, val: 140, test: 139
               ↓
        CSI_augment.py ──→ phase_hampel_shift/data.npz
                          train: 2608, val: 140, test: 139
                          ↓
                    CSI_normalize.py ──→ normalization_stats.npz
                          ↓
                    CSI_dataset.py ──→ DataLoader batches
                                       shape: (B, 192, 192)
```

### **Phần 2: Training**

```
4 artifacts chính:
├── raw (652/140/139)
├── raw_shift (2608/140/139)
├── phase_hampel (652/140/139)
└── phase_hampel_shift (2608/140/139)

                    ↓
        
Stage 1: GRU Screening (7 runs)
├─→ Top 3 chọn lọc
        ↓
Stage 2: Attention-GRU Narrowing (10 runs)
├─→ Top 4 chọn lọc
        ↓
Stage 3: Multi-output Cell Bottleneck (10 runs)
├─→ Top 3 chọn lọc
        ↓
Stage 4: Final Confirmation & Ablation (8 runs)
├─→ Xác nhận & ablation study
        ↓
Best Model + Metrics cho Report
```

---

## 📊 Định Dạng Dữ Liệu

### **Input (Phần 1)**
- CSI raw data từ Wi-Fi, shape: `(3, 64, 192)`
- Ý nghĩa:
  - `3` = RX antennas / channels
  - `64` = subcarriers
  - `192` = time frames

### **Output Phần 1**
- `X_train/val/test`: shape `(N, 3, 64, 192)`
- `y_train/val/test`: class labels (pose, cell, presence, center)

### **Input Phần 2 (vào GRU)**
Sau xử lí trong Dataset:
- `(3, 64, 192)` → chuẩn hóa → optional thêm Gaussian noise
- Reshape thành `(192, 192)` → batch: `(B, 192, 192)`
- `batch_first=True`, `sequence_length=192`, `input_size=192`

---

## 🎯 Mục Tiêu & Các Task

### **Phần 1: Mục Tiêu**
1. ✅ Tách dữ liệu thành train/val/test
2. ✅ Cắt window 192 frames
3. ✅ Loại nhiễu (Denoise)
4. ✅ Augmentation temporal shift
5. ✅ Fit normalization stats trên train
6. ✅ Chuẩn bị PyTorch Dataset/DataLoader
7. ✅ Kiểm tra class imbalance

**Lưu ý:** Không huấn luyện model trong Phần 1.

### **Phần 2: Mục Tiêu**
1. ✅ Xây dựng GRU baseline cho chuỗi CSI
2. ✅ Thêm Attention để học temporal attention weights
3. ✅ Kiểm tra multi-output cho các task: presence, cell, pose, center
4. ✅ Dùng validation metrics để chọn model tốt nhất
5. ✅ Chỉ dùng test metrics cho báo cáo cuối, không tune model

---

## 🏗️ Kiến Trúc Model

### **Stage 1: GRU Baseline**
```
CSI window (B, 192, 192)
        ↓
    GRU layers
        ↓
   hidden state cuối
        ↓
  Linear classifier
        ↓
  7 pose classes
```

### **Stage 2: Attention-GRU**
```
CSI window (B, 192, 192)
        ↓
GRU (hoặc BiGRU)
        ↓
Temporal Attention Pooling
        ↓
   Dropout
        ↓
  Linear classifier
        ↓
  7 pose classes
```

### **Stage 3 & 4: Multi-output Shared Trunk**
```
CSI window (B, 192, 192)
        ↓
Shared Attention-GRU Trunk
        ↓
┌──────────────────────────────────┐
│ presence head (binary)           │
│ cell head (25-class)             │
│ pose head (7-class)              │
│ center head (x/y regression)     │
└──────────────────────────────────┘
```

---

## 🔧 Kỹ Thuật Huấn Luyện

| Kỹ Thuật | Mục Đích |
|---------|---------|
| **AdamW** | Tối ưu với weight decay tách riêng |
| **ReduceLROnPlateau** | Giảm learning rate khi validation dừng cải thiện |
| **Early Stopping** | Dừng training khi validation không cải thiện |
| **Gradient Clipping** | Giảm nguy hiểm gradient quá lớn trong RNN |
| **Runtime Gaussian Noise** | Thêm noise nhẹ trên train (regularization) |
| **Focal Loss** | Tăng trọng số cho hard examples (cell) |
| **Effective-number Class Weighting** | Cân bằng lớp theo tần suất |
| **Label Smoothing** | Làm target bớt quá chắc chắn |
| **Top-k Validation Selector** | Chọn cấu hình tốt nhất qua stage |
| **Composite Score** | Gộp nhiều metrics cho multi-output |

---

## 📈 Thử Nghiệm Có Kiểm Soát

### **4 Stages với 35 Runs:**

| Stage | Prefix | Số Runs | Mục Đích | Selector |
|-------|--------|---------|---------|----------|
| **Stage 1** | `S1_G*` | 7 | GRU screening | Top 3 theo `val_macro_f1` |
| **Stage 2** | `S2_A*` | 10 | Attention-GRU narrowing | Top 4 theo `val_macro_f1` |
| **Stage 3** | `S3_M*` | 10 | Multi-output cell bottleneck | Top 3 theo `val_composite_score` |
| **Stage 4** | `S4_F*` | 8 | Final confirmation & ablation | Xác nhận configs tốt |
| **Tổng** | | **35** | | |

---

## 📝 Cách Chạy

### **Phần 1: Preprocessing**
```bash
1. Chuẩn bị dữ liệu gốc: Dataset_CSI_3D_v2/session1/
2. Mở Jupyter Notebook
3. Chạy: Phần 1 tiền xử lí/workflow.ipynb
4. Output: processed/windows/<profile>/data.npz
```

### **Phần 2: Training**
```bash
1. Hoàn tất Phần 1, giữ nguyên: Phần 1 tiền xử lí/processed/windows/
2. Mở Jupyter Notebook
3. Chạy: Phần 2 mô hình/workflow_all.ipynb
4. Bấm "Run All" để chạy toàn bộ 35 training runs
5. Output: runs/<model_type>/<experiment>/<files>
```

**Thời gian ước tính:** 25-46 phút (tùy early stopping & hardware)

---

## 📊 Kết Quả Hiện Tại

| Mục | Run | Metric Chính |
|-----|-----|-------------|
| **Best Pose Test Overall** | `S1_G02_raw_shift_h64_anchor` | `test_macro_f1 = 0.6877` |
| **Best GRU (Validation)** | `S1_G04_raw_shift_h128` | `val_macro_f1 = 0.7474` |
| **Best Attention (Validation)** | `S2_A01_top_gru_a32` | `val_macro_f1 = 0.7242` |
| **Best Multi-output Final** | `S4_F06_multi_linear_ablation` | `val_composite_score = 0.5682` |
| **Best Localization** | `S4_F06_multi_linear_ablation` | `within 1 cell = 0.6692` |

---

## 💾 Output Sau Khi Chạy

Mỗi training run tạo một thư mục với:

| File | Nội Dung |
|------|---------|
| `config.json` | Cấu hình của run |
| `metrics.json` | Tổng kết metrics |
| `history.csv` | Loss/metrics theo epoch |
| `predictions.csv` | Dự đoán trên val/test |
| `best_model.pt` | Checkpoint tốt nhất (validation) |
| `last_model.pt` | Checkpoint cuối cùng |
| `selection.json` | Thông tin selector (multi-output) |

---

## ⚙️ Các Config Chính

### **Phần 1:**
- `OVERWRITE_DENOISE`: True = rebuild denoise, False = giữ cũ
- `OVERWRITE_AUGMENT`: True = rebuild augmentation, False = giữ cũ
- `DENOISE_PROFILES`: `['phase_hampel']` (profile denoise chính)
- `AUGMENT_STRIDE`: 16 (khoảng cách giữa crops)
- `AUGMENT_MAX_CROPS`: 4 (số crop tối đa)
- `NOISE_SIGMA`: 0.01 (Gaussian noise runtime)
- `TARGET`: `'pose'` (hoặc 'cell', 'presence', 'center')

### **Phần 2:**
- `num_layers`: 1 hoặc 2 (RNN layers)
- `hidden_size`: 64, 96, 128 (GRU hidden dim)
- `attention_dim`: 32, 48, 64 (Attention dim)
- `dropout`: 0.0 - 0.3
- `batch_size`: 32, 64
- `learning_rate`: 1e-3, 5e-4
- `max_epochs`: 80-100

---

## 📚 Tài Liệu Chi Tiết

- **Phần 1 Chi Tiết:** `Phần 1 tiền xử lí/README.md`
- **Phần 2 Chi Tiết:** `Phần 2 mô hình/README.md`

---

## 🚀 Cách Viết Báo Cáo

1. **Stage 1 Ranking:** Tìm GRU backbone ổn định
2. **Stage 2 Ranking:** Xem Attention có cải thiện không
3. **Stage 3 Ranking:** Xem Multi-output & bottleneck cell
4. **Stage 4:** Xác nhận top model & đọc ablation
5. **Cuối Cùng:** Dùng test metrics của model được chọn

**Quy tắc:** 
- Validation → chọn model nào
- Test → model đã chọn đạt kết quả cuối ra sao

---

## 📝 Ghi Chú Hạn Chế

- Dataset nhỏ, val/test chỉ có 140/139 mẫu
- Cell classification là bottleneck (25 classes, support thấp)
- Kết quả có thể dao động theo seed
- Không nên tuyên bố model đã "giải quyết" bài toán hoàn toàn

---

## 📞 Liên Hệ & Support

Dự án dùng PyTorch, Jupyter Notebook. Xem các README trong từng phần để biết chi tiết.

**Last Updated:** June 13, 2026
