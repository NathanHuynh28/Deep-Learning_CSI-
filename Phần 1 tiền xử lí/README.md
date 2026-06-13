Part 1 Preprocessing README
Mục tiêu
Part 1 biến raw Wi-Fi CSI thành dữ liệu sẵn sàng đưa vào model ở Part 2.
Pipeline hiện làm các việc:
1. Split sample train/val/test
2. Cắt window 192 frames
3. Denoise CSI
4. Augment temporal shift cho train
5. Fit normalization stats trên train
6. Chuẩn bị PyTorch Dataset/DataLoader
7. Inspect class imbalance
Không làm training trong Part 1.
Cây thư mục chính
Phần 1 tiền xử lí/
├── workflow.ipynb
├── split_and_window.py
├── Denoise/
│   └── CSI_denoise.py
├── Augmentation/
│   └── CSI_augment.py
├── Normalization/
│   └── CSI_normalize.py
├── Dataset/
│   └── CSI_dataset.py
├── processed/
│   ├── sample_split/
│   │   ├── split.csv
│   │   └── report.csv
│   ├── denoised/
│   │   └── phase_hampel/
│   └── windows/
│       ├── raw/
│       ├── raw_shift/
│       ├── phase_hampel/
│       └── phase_hampel_shift/
└── reports/
    └── denoise/
        └── phase_hampel/
Thư mục nên bỏ qua/xóa được:
CSI Denoising/     # legacy, rỗng, không dùng
__pycache__/       # cache Python
Luồng dữ liệu tổng thể
Dataset_CSI_3D_v2/session1/
        |
        v
split_and_window.py
        |
        +--> processed/sample_split/split.csv
        |
        +--> processed/windows/raw/data.npz
                    |
                    +--> Augmentation/CSI_augment.py
                    |           |
                    |           v
                    |   processed/windows/raw_shift/data.npz
                    |
                    v
Denoise/CSI_denoise.py
        |
        v
processed/denoised/phase_hampel/
        |
        v
split_and_window.py
        |
        v
processed/windows/phase_hampel/data.npz
        |
        v
Augmentation/CSI_augment.py
        |
        v
processed/windows/phase_hampel_shift/data.npz
Sau đó:
processed/windows/<profile>/data.npz
        |
        v
Normalization/CSI_normalize.py
        |
        v
processed/windows/<profile>/normalization_stats.npz
        |
        v
Dataset/CSI_dataset.py
        |
        v
x batch cho GRU: (B, 192, 192)
4 artifact chính
raw
raw_shift
phase_hampel
phase_hampel_shift
Shape hiện tại:
raw:
  train = 652
  val   = 140
  test  = 139

raw_shift:
  train = 2608
  val   = 140
  test  = 139

phase_hampel:
  train = 652
  val   = 140
  test  = 139

phase_hampel_shift:
  train = 2608
  val   = 140
  test  = 139
Mỗi sample có tensor gốc:
(3, 64, 192)
Ý nghĩa:
3   = RX/channel
64  = subcarrier
192 = time frames
6 training scenarios cho Part 2
Dù chỉ có 4 data.npz, ta có 6 scenario vì Gaussian noise là runtime option:
1. raw + no noise
2. raw_shift + no noise
3. raw_shift + noise 0.01
4. phase_hampel + no noise
5. phase_hampel_shift + no noise
6. phase_hampel_shift + noise 0.01
Không tạo thêm data.npz cho noise.
Dữ liệu trước khi vào GRU
Trong data.npz:
X_train: (N, 3, 64, 192)
Trong Dataset runtime:
(3, 64, 192)
-> normalize
-> optional Gaussian noise nếu train
-> transpose/reshape
-> (192, 192)
Trong DataLoader:
x: (B, 192, 192)
y: (B,)
Với GRU Part 2:
batch_first = True
input_size = 192
sequence_length = 192
Config chính trong workflow
Trong cell đầu của workflow.ipynb:
OVERWRITE_DENOISE = True
Ý nghĩa:
True  = rebuild denoise phase_hampel từ raw
False = không ghi đè denoise output cũ
Khi dùng:
- Xóa processed/ và chạy lại từ đầu: True
- Đã ổn định artifact, chỉ inspect lại: False hoặc bỏ qua cell denoise
OVERWRITE_AUGMENT = True
Ý nghĩa:
True  = rebuild raw_shift và phase_hampel_shift
False = giữ augmentation cũ
Khuyến nghị:
- Khi đổi stride/crop: True
- Khi không đổi gì: có thể False
DENOISE_PROFILES = ['phase_hampel']
Profile denoise chính.
Giữ:
phase_hampel
Chưa cần dùng:
phase_hampel_dwt
AUGMENT_STRIDE = 16
AUGMENT_MAX_CROPS = 4
Ý nghĩa:
stride = khoảng cách giữa các crop theo thời gian
max_crops = số crop tối đa mỗi train sample
Bạn đang chọn:
stride 16, crop 4
Đây là lựa chọn cân bằng, tránh sample dài áp đảo.
NORMALIZE_EPS = 1e-6
Dùng để tránh chia cho 0 khi feature hằng số.
Giữ mặc định.
DATASET_PROFILE = 'raw_shift'
Profile dùng để demo DataLoader.
Có thể đổi:
'raw'
'raw_shift'
'phase_hampel'
'phase_hampel_shift'
TARGET = 'pose'
Target đang inspect.
Có thể dùng:
pose      = phân loại tư thế
cell      = phân loại vị trí ô
presence  = có/người hay không
center    = regression tọa độ x,y
Nếu TARGET='center', class weights sẽ bỏ qua vì đây không phải classification.
NOISE_SIGMA = 0.01
Gaussian noise runtime.
Ý nghĩa:
0.0  = không noise
0.01 = noise nhẹ
0.02 = noise mạnh hơn, chỉ thử nếu cần
Noise chỉ áp dụng cho:
train split
Không áp dụng cho:
val/test
Workflow nên trình bày thêm gì
Tôi khuyên nâng cấp workflow.ipynb thành dạng dashboard rõ hơn, thêm các bảng:
1. Pipeline Map
Hiển thị:
raw -> raw_shift
raw -> phase_hampel -> phase_hampel_shift
2. Artifact Summary Table
Bảng:
profile | train | val | test | shape | has_norm_stats | has_nan | has_inf
3. Scenario Table
Bảng:
scenario | profile | denoise | temporal_shift | noise_sigma | train_count | GRU_shape
4. Normalization Summary
Bảng:
profile | mean_shape | std_shape | train_mean | active_feature_std | constant_features
5. DataLoader Preview
Hiển thị:
selected profile
target
noise sigma
x batch shape
y batch shape
dtype
6. Class Imbalance Summary
Bảng:
target | class_id | label_name | train_count | val_count | test_count | weight
Hiện workflow đã có logic, nhưng chưa đủ trực quan. Các bảng này sẽ giúp bạn hiểu data trước khi train.