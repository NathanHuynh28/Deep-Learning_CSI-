# Deep-Learning CSI (Channel State Information)

Dự án nghiên cứu sử dụng Deep Learning để phân loại tư thế người (pose estimation) từ dữ liệu CSI (Channel State Information) - thông tin trạng thái kênh Wi-Fi.

## 📋 Tổng Quan Project

Project này chia thành **3 phần chính**:

### **Phần 0: Thu Thập Dữ Liệu CSI (Data Collection)**
- Thu thập dữ liệu CSI từ 4 ESP32-S3 modules (1 TX + 3 RX)
- Xây dựng perimeter layout với 25 zones (5x5 grid)
- Thu thập 1370 samples với các tư thế: Standing, Sitting, Arms_Out, Lying, Walk_in_place, Transition
- Output: CSV files chứa CSI metadata và I/Q values

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
├── code firmware v2/                  # PHẦN 0: Thu thập dữ liệu
│   ├── operation_guide.md             # Hướng dẫn chi tiết thu thập dữ liệu
│   ├── controller.py                  # Controller chính điều khiển 4 nodes
│   ├── firmware/                      # Code firmware cho TX/RX
│   │   ├── tx_firmware.ino
│   │   └── rx_firmware.ino
│   └── config/
│       └── calibration.json           # Calibration cho layout/nodes
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

├── Dataset_CSI_3D_v2/                 # Dữ liệu gốc (từ Phần 0)
│   └── session1/
│       ├── session.md
│       ├── calibration.json
│       ├── manifest.csv
│       ├── quality.csv
│       └── data_P*.csv                # Raw CSI data (6 files)
```

---

## 🔄 Luồng Dữ Liệu Tổng Thể

```
PHẦN 0: Thu Thập Dữ Liệu
│
├─ 4 ESP32-S3 Modules (1 TX + 3 RX)
│  ├─ TX: x=-0.50, y=1.50 (West)
│  ├─ RX1: x=1.50, y=-0.50 (North)
│  ├─ RX2: x=3.50, y=1.50 (East)
│  └─ RX3: x=1.50, y=3.50 (South)
│
├─ 1370 Samples (25 zones x 5x5 grid)
│  ├─ 80 Empty (checkpoints)
│  ├─ 390 Standing
│  ├─ 310 Sitting
│  ├─ 150 Arms_Out
│  ├─ 190 Lying
│  ├─ 160 Walk_in_place
│  └─ 90 Transition
│
└─ Dataset_CSI_3D_v2/session1/
   └─ data_P*.csv (438 columns: metadata + 3 RX × (9 metadata + 128 CSI))
        ↓
PHẦN 1: Preprocessing
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
                          ↓
PHẦN 2: Training
                          ↓
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

### **Input (Phần 0 - Thu Thập)**
- CSI data từ 3 RX modules
- Mỗi frame: 100 Hz, 128 I/Q values per RX, 13 metadata fields
- CSV format: 438 columns
- Tư thế: 7 classes (Standing, Sitting, Arms_Out, Lying, Walk_in_place, Transition, Empty)
- Vị trí: 25 zones (5x5 grid)

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

### **Phần 0: Mục Tiêu Thu Thập Dữ Liệu**
1. ✅ Xây dựng perimeter layout với 4 modules (1 TX + 3 RX)
2. ✅ Thu thập 1370 samples với các tư thế đa dạng
3. ✅ Đảm bảo chất lượng CSI (clean frames, valid sequences)
4. ✅ Tạo calibration metadata cho preprocessing
5. ✅ QC kiểm tra firmware counters, channel, baudrate

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

## 🎮 Phần 0: Hướng Dẫn Thu Thập Dữ Liệu CSI

### **1. Active Layout: layout1**

Quy ước tọa độ bắt buộc:
- Nhìn top-down từ trên xuống
- Gốc (0,0) là góc trên-trái của lưới 5 cột × 5 hàng
- `+x` tăng từ west sang east
- `+y` tăng từ north sang south
- Kích thước grid: 3.00m × 3.00m, mỗi zone 0.60m × 0.60m

**Node Positions:**

| Node | Side | Tọa Độ | Chiều Cao |
|---|---|---|---|
| TX | West | x=-0.50, y=1.50 | h=1.00 |
| RX1 | North | x=1.50, y=-0.50 | h=1.00 |
| RX2 | East | x=3.50, y=1.50 | h=1.00 |
| RX3 | South | x=1.50, y=3.50 | h=1.00 |

```
                     North / -y
                     RX1 (1.50, -0.50, h=1.00)
                           |
                           v
    +------+------+------+------+------+   East / +x
    | C01  | C02  | C03  | C04  | C05  |   RX2 (3.50, 1.50, h=1.00)
    +------+------+------+------+------+         <---
West| C06  | C07  | C08  | C09  | C10  |
TX --->+------+------+------+------+------+   origin top-left of grid
(-0.50,| C11  | C12  | C13  | C14  | C15  |   +x west->east
 1.50) +------+------+------+------+------+   +y north->south
h=1.00 | C16  | C17  | C18  | C19  | C20  |
    +------+------+------+------+------+
    | C21  | C22  | C23  | C24  | C25  |
    +------+------+------+------+------+
                           ^
                           |
                    RX3 (1.50, 3.50, h=1.00)
                    South / +y
```

### **2. Tư Thế (Pose/Orientation)**

| Label | Degree | Ý Nghĩa |
|---|---:|---|
| `Empty` | - | Không có người trong vùng thu |
| `Facing_RX` | 0 | +x, nhìn east về RX2 |
| `Facing_TX` | 180 | -x, nhìn west về TX |
| `Facing_Left` | 270 | -y, nhìn north về RX1 |
| `Facing_Right` | 90 | +y, nhìn south về RX3 |
| `Diagonal_RX1` | 315 | +x/-y, hướng northeast |
| `Diagonal_RX3` | 45 | +x/+y, hướng southeast |

### **3. Large Collection Plan: PART1_25ZONE_PERIMETER_V1**

**Tổng 1370 samples:**

| Block | Count | Mục Đích |
|---|---:|---|
| Empty checkpoints | 80 | 16 checkpoint × 5 trial |
| Standing | 390 | 78/person: interior 3×3 × 4 hướng, perimeter, diagonal |
| Sitting | 310 | 62/person: interior 3×3 × 4 hướng, inward, diagonal |
| Arms_Out | 150 | 30/person: cross5 × 4 hướng, corner inward |
| Lying | 190 | 38/person: cross5 × 4 hướng, corner/edge inward |
| Walk_in_place | 160 | 32/person: cross5 × 4 hướng, corner/edge inward |
| Transition | 90 | 18/person: cross5 × 2 axis, corner/edge inward |

**Core cross zones:** `C08, C12, C13, C14, C18`

### **4. QC Checklist**

**Trước Thu:**
- [ ] Dọn phòng, không có người khác đi qua
- [ ] Dán lưới C01..C25 đúng kích thước 3m × 3m
- [ ] Đặt TX west, RX1 north, RX2 east, RX3 south đúng tọa độ
- [ ] Anten cao 1.00m, cùng hướng, không chạm trong session
- [ ] Chụp ảnh setup phòng và ghi tọa độ nếu khác default
- [ ] Kiểm tra channel TX/RX/controller đều là 1
- [ ] Kiểm tra TX Serial Monitor `TX_MAC,...`
- [ ] Chạy: `python controller.py --plan-only`
- [ ] Chạy: `python controller.py --preflight-only`
- [ ] Preflight PASS: `first_word_invalid=0`, `malformed=0`, `no_csi/output_busy` trong ngưỡng

**Trong Thu:**
- [ ] Actor đứng center/footprint controller yêu cầu
- [ ] Actor quay đúng orientation theo map
- [ ] Empty sample `P0/NO_CELL` phải không có người
- [ ] Transition dùng pattern sit_down_then_stand_up
- [ ] FAIL thì thử lại sample đó, không skip

**Sau Thu:**
- [ ] `quality.csv` PASS: `clean_frames=240`
- [ ] Không có `first_word_invalid_*`
- [ ] `missing_seq <= 20` cho PASS
- [ ] Mỗi RX có `RX*_csi_len = 128`
- [ ] `calibration.json` đầy đủ metadata
- [ ] Không trộn session khác layout/channel

### **5. File Output (Phần 0)**

```
Dataset_CSI_3D_v2/session1/
├── session.md
├── calibration.json
├── manifest.csv
├── quality.csv
├── data_P0.csv  (Empty samples)
├── data_P1.csv  (Person 1)
├── data_P2.csv  (Person 2)
├── data_P3.csv  (Person 3)
├── data_P4.csv  (Person 4)
└── data_P5.csv  (Person 5)
```

Mỗi file: 438 columns = 27 base + 3 RX × (9 metadata + 128 CSI)

---

## 📝 Cách Chạy

### **Phần 0: Thu Thập Dữ Liệu**
```bash
1. Chuẩn bị 4 ESP32-S3 modules (firmware trong code firmware v2/)
2. Xây dựng perimeter layout theo layout1 (5x5 grid, 3m x 3m)
3. Cấp điện, flash firmware cho TX + 3 RX
4. Chạy: python controller.py --plan-only    # Kiểm tra plan
5. Chạy: python controller.py --preflight-only  # QC trước
6. Chạy: python controller.py  # Thu thập 1370 samples
7. Output: Dataset_CSI_3D_v2/session1/
```

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

**Thời gian ước tính:** 
- Phần 0 (Thu thập): 8-10 giờ (1370 samples)
- Phần 1 (Preprocessing): 5-10 phút
- Phần 2 (Training): 25-46 phút

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

### **Phần 0 (Controller):**
- `SESSION_ID`: "session1" (tên session thu thập)
- `ROOM_NAME`: "room1" (tên phòng)
- `LAYOUT_ID`: "layout1" (ID layout được dùng)
- `COLLECTION_PLAN_ID`: "PART1_25ZONE_PERIMETER_V1" (plan 1370 samples)
- `CALIBRATION_ID`: "calibration1" (ID calibration)
- `COLLECTION_MODE`: "part1_large" (mode thu chính)
- `COM_PORTS`: Các cổng COM của TX/RX (điều chỉnh theo máy)
- `CHANNEL`: 1 (Wi-Fi channel, giữ nguyên)

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

- **Phần 0 Chi Tiết:** `code firmware v2/operation_guide.md`
- **Phần 1 Chi Tiết:** `Phần 1 tiền xử lí/README.md`
- **Phần 2 Chi Tiết:** `Phần 2 mô hình/README.md`

---

## 🚀 Cách Viết Báo Cáo

1. **Phần 0:** Mô tả setup hardware, layout, số samples thu được, QC metrics
2. **Phần 1:** Mô tả preprocessing steps, data augmentation, normalization
3. **Stage 1 Ranking:** Tìm GRU backbone ổn định
4. **Stage 2 Ranking:** Xem Attention có cải thiện không
5. **Stage 3 Ranking:** Xem Multi-output & bottleneck cell
6. **Stage 4:** Xác nhận top model & đọc ablation
7. **Cuối Cùng:** Dùng test metrics của model được chọn

**Quy tắc:** 
- Validation → chọn model nào
- Test → model đã chọn đạt kết quả cuối ra sao

---

## 📝 Ghi Chú Hạn Chế

- Dataset nhỏ, val/test chỉ có 140/139 mẫu
- Cell classification là bottleneck (25 classes, support thấp)
- Kết quả có thể dao động theo seed
- Không nên tuyên bố model đã "giải quyết" bài toán hoàn toàn
- CSI là 2.5D occupancy/footprint, không phải true 3D body mesh hay 17 keypoints

---

## 🔧 Debug & Troubleshooting

### **Lỗi 3 RX Drop (Phần 0)**

Nếu 1 RX và 2 RX pass nhưng 3 RX cùng lúc bị drop rate:

```bash
# Flash lại firmware
python controller.py --preflight-only
python controller.py --soak-only --soak-seconds 30
```

**Kiểm tra firmware stats:**
- `csi_cb`: Tổng CSI callback firmware nhìn thấy
- `csi_seen`: CSI callback đã qua length filter
- `espnow_valid`: ESP-NOW packet đã qua validation
- `seq_gap`: Nếu tăng → RX đang mất packet trên radio
- `csi_mac_mismatch`: Kiểm tra TX_MAC_FILTER
- `output_busy`: Serial output bottleneck

### **Kiểm tra Preprocessing**

```python
# Check shape sau mỗi bước
import numpy as np
data = np.load('processed/windows/raw/data.npz')
print(data['X_train'].shape)  # Should be (652, 3, 64, 192)
print(data['y_train'].shape)  # Should be (652,)
```

---

## 📞 Liên Hệ & Support

Dự án dùng:
- **Phần 0:** ESP32-S3, Arduino IDE, Python controller
- **Phần 1 & 2:** PyTorch, Jupyter Notebook, NumPy/Pandas

Xem các README trong từng phần để biết chi tiết.

**Last Updated:** June 13, 2026
