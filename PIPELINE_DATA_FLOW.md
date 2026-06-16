# 📊 CSI Data Pipeline - Data Flow & Directory Tree

This document visualizes how raw CSI data flows through the entire preprocessing pipeline, showing exactly how data is transformed and organized at each step.

---

## 🌳 Complete Data Flow Tree (Visual Overview)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAW CSI DATA (INPUT)                           │
│  Dataset_CSI_3D_v2/session1/                                            │
│  ├─ data_P0.csv  (Empty samples)                                        │
│  ├─ data_P1.csv  (Person 1 - multiple poses, zones)                    │
│  ├─ data_P2.csv  (Person 2)                                            │
│  ├─ data_P3.csv  (Person 3)                                            │
│  ├─ data_P4.csv  (Person 4)                                            │
│  ├─ data_P5.csv  (Person 5)                                            │
│  └─ calibration.json  (TX/RX coordinates, metadata)                    │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUALITY CHECK / EDA                                  │
│  ✓ Remove invalid samples (corrupted frames, malformed sequences)      │
│  ✓ Keep valid samples (clean_frames >= 240)                           │
│  ✓ Verify CSI dimensions: 3 RX × 128 subcarriers × 192 time frames   │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    SPLIT SAMPLE-LEVEL                                  │
│  ✓ Fixed split: train / val / test                                     │
│  ✓ Train: 652 samples (75.4%)                                         │
│  ✓ Val:   140 samples (16.1%)                                         │
│  ✓ Test:  139 samples (16.0%)                                         │
│  ↓                                                                      │
│  Phần 1 tiền xử lí/processed/sample_split/                            │
│  └─ split.csv              # (sample_id, split, pose, cell, ...)      │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
            ┌──────────────────────────────────────┐
            │  DATA SPLITS INTO 2 MAIN BRANCHES   │
            └──────────────────────────────────────┘
            ↓                                       ↓
     ┌──────────────┐                    ┌──────────────────┐
     │  BRANCH 1    │                    │   BRANCH 2       │
     │   RAW        │                    │   DENOISE        │
     └──────────────┘                    └──────────────────┘

═══════════════════════════════════════════════════════════════════════════
BRANCH 1: RAW (No Denoising)
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1.1: CUT WINDOW 192 FRAMES                                         │
│  ✓ Input: (3, 64, 192) raw CSI per sample                             │
│  ✓ Window operation: sliding window or fixed window on 192 frames     │
│  ✓ Output shape: (3, 64, 192)                                         │
│  ✓ Create ARTIFACT: raw                                               │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
        Phần 1 tiền xử lí/processed/windows/raw/
        ├── data.npz
        │   ├─ X_train: (652, 3, 64, 192)
        │   ├─ y_train: (652,)  [pose labels]
        │   ├─ X_val:   (140, 3, 64, 192)
        │   ├─ y_val:   (140,)
        │   ├─ X_test:  (139, 3, 64, 192)
        │   └─ y_test:  (139,)
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1.2: TEMPORAL SHIFT AUGMENTATION                                   │
│  ✓ Apply only to TRAIN split                                          │
│  ✓ Strategy: sliding window crops with stride 16 frames               │
│  ✓ Max crops per sample: 4                                            │
│  ✓ Multiply train samples: 652 → 2608                                │
│  ✓ Val/Test unchanged: 140, 139                                       │
│  ✓ Create ARTIFACT: raw_shift                                         │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
        Phần 1 tiền xử lí/processed/windows/raw_shift/
        ├── data.npz
        │   ├─ X_train: (2608, 3, 64, 192)  ← augmented
        │   ├─ y_train: (2608,)
        │   ├─ X_val:   (140, 3, 64, 192)   ← unchanged
        │   ├─ y_val:   (140,)
        │   ├─ X_test:  (139, 3, 64, 192)   ← unchanged
        │   └─ y_test:  (139,)

═══════════════════════════════════════════════════════════════════════════
BRANCH 2: DENOISE (Phase Hampel)
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.1: DENOISE - PHASE HAMPEL                                        │
│  ✓ Input: raw CSI from Dataset_CSI_3D_v2/session1/                   │
│  ✓ Method: Phase sanitization + Hampel filter                        │
│  ✓ Goal: Remove outliers while preserving temporal structure         │
│  ✓ Operates on: amplitude + phase components                         │
│  ✓ Output: denoised CSI (3, 64, 192)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
        Phần 1 tiền xử lí/processed/denoised/phase_hampel/
        ├── denoised_P0.csv  (denoised CSI for Empty)
        ├── denoised_P1.csv
        ├── denoised_P2.csv
        ├── denoised_P3.csv
        ├── denoised_P4.csv
        └── denoised_P5.csv
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.2: CUT WINDOW 192 FRAMES (ON DENOISED DATA)                     │
│  ✓ Input: denoised CSI (3, 64, 192)                                   │
│  ✓ Window operation: same as Branch 1 Step 1.1                        │
│  ✓ Output shape: (3, 64, 192)                                         │
│  ✓ Create ARTIFACT: phase_hampel                                      │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
        Phần 1 tiền xử lí/processed/windows/phase_hampel/
        ├── data.npz
        │   ├─ X_train: (652, 3, 64, 192)
        │   ├─ y_train: (652,)
        │   ├─ X_val:   (140, 3, 64, 192)
        │   ├─ y_val:   (140,)
        │   ├─ X_test:  (139, 3, 64, 192)
        │   └─ y_test:  (139,)
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.3: TEMPORAL SHIFT AUGMENTATION (ON DENOISED DATA)                │
│  ✓ Apply only to TRAIN split (same as Step 1.2)                       │
│  ✓ Strategy: identical to Branch 1                                    │
│  ✓ Multiply: 652 → 2608                                              │
│  ✓ Val/Test unchanged: 140, 139                                       │
│  ✓ Create ARTIFACT: phase_hampel_shift                                │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
        Phần 1 tiền xử lí/processed/windows/phase_hampel_shift/
        ├── data.npz
        │   ├─ X_train: (2608, 3, 64, 192)  ← augmented
        │   ├─ y_train: (2608,)
        │   ├─ X_val:   (140, 3, 64, 192)   ← unchanged
        │   ├─ y_val:   (140,)
        │   ├─ X_test:  (139, 3, 64, 192)   ← unchanged
        │   └─ y_test:  (139,)

═══════════════════════════════════════════════════════════════════════════
UNIFIED PROCESSING (After all 4 artifacts created)
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CREATE 4 DATA PROFILES                                          │
│                                                                          │
│  Profile 1: raw                  (652/140/139)                          │
│  Profile 2: raw_shift            (2608/140/139)  ← main profile        │
│  Profile 3: phase_hampel         (652/140/139)                          │
│  Profile 4: phase_hampel_shift   (2608/140/139)                         │
│                                                                          │
│  All profiles undergo the following steps:                              │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: FIT NORMALIZATION STATS                                         │
│  ✓ Compute mean & std on TRAIN SPLIT ONLY                             │
│  ✓ Do NOT use val/test data for computing stats                       │
│  ✓ Store stats: per-feature normalization parameters                  │
│  ✓ Output: normalization_stats.npz per profile                        │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
        For each profile in {raw, raw_shift, phase_hampel, phase_hampel_shift}:
        
        Phần 1 tiền xử lí/processed/windows/<profile>/
        └── normalization_stats.npz
            ├─ mean:  per-feature means computed from train
            ├─ std:   per-feature stds computed from train
            ├─ eps:   small constant to avoid division by zero
            └─ metadata: feature_dim, total_features

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: DATASET / DATALOADER RUNTIME PROCESSING                        │
│  ✓ Load X from data.npz: shape (3, 64, 192)                           │
│  ✓ Apply normalization using stats from normalization_stats.npz      │
│  ✓ If TRAIN: add optional Gaussian noise (σ = 0.01)                   │
│  ✓ If VAL/TEST: no noise added                                        │
│  ✓ Reshape to GRU input: (192, 192)                                   │
│  ✓ Batch in DataLoader: (B, 192, 192)                                 │
│  ✓ Ready for GRU training (sequence_length=192, input_size=192)      │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
                    ╔════════════════════════╗
                    ║ PART 2: MODEL TRAINING ║
                    ║  Uses processed data   ║
                    ║  4 scenarios available:║
                    ║  • raw                 ║
                    ║  • raw_shift           ║
                    ║  • phase_hampel        ║
                    ║  • phase_hampel_shift  ║
                    ║  + runtime noise option║
                    ╚════════════════════════╝
```

---

## 📁 Complete Directory Structure

```
Deep-Learning_CSI-/
│
├── Dataset_CSI_3D_v2/                    # Raw collected data
│   └── session1/
│       ├── session.md                    # Session metadata
│       ├── calibration.json              # TX/RX positions
│       ├── manifest.csv                  # File inventory
│       ├── quality.csv                   # QC report (clean_frames, etc)
│       ├── data_P0.csv                   # Empty samples (438 cols)
│       ├── data_P1.csv                   # Person 1
│       ├── data_P2.csv
│       ├── data_P3.csv
│       ├── data_P4.csv
│       └── data_P5.csv
│
├── Phần 1 tiền xử lí/                    # PREPROCESSING (Part 1)
│   │
│   ├── README.md
│   ├── workflow.ipynb                    # Main preprocessing notebook
│   │
│   ├── split_and_window.py               # Step 1: Split & window
│   ├── Denoise/
│   │   └── CSI_denoise.py               # Step 2.1: Hampel denoising
│   ├── Augmentation/
│   │   └── CSI_augment.py               # Step 1.2 & 2.3: Temporal shift
│   ├── Normalization/
│   │   └── CSI_normalize.py             # Step 4: Fit normalization stats
│   ├── Dataset/
│   │   └── CSI_dataset.py               # Step 5: PyTorch Dataset
│   │
│   └── processed/                        # OUTPUT ARTIFACTS
│       │
│       ├── sample_split/
│       │   ├── split.csv                # Train/val/test assignment
│       │   └── report.csv               # Class distribution report
│       │
│       ├── denoised/
│       │   └── phase_hampel/            # Denoised CSI (Branch 2.1)
│       │       ├── denoised_P0.csv
│       │       ├── denoised_P1.csv
│       │       ├── denoised_P2.csv
│       │       ├── denoised_P3.csv
│       │       ├── denoised_P4.csv
│       │       └── denoised_P5.csv
│       │
│       └── windows/                     # Main processed data (Steps 1.1, 2.2, 1.2, 2.3, 4, 5)
│           │
│           ├── raw/                     # ARTIFACT 1: Raw (Branch 1, no augment)
│           │   ├── data.npz
│           │   │   ├─ X_train: (652, 3, 64, 192)
│           │   │   ├─ y_train: (652,)
│           │   │   ├─ X_val:   (140, 3, 64, 192)
│           │   │   ├─ y_val:   (140,)
│           │   │   ├─ X_test:  (139, 3, 64, 192)
│           │   │   └─ y_test:  (139,)
│           │   └── normalization_stats.npz
│           │       ├─ mean: per-feature mean from train
│           │       ├─ std:  per-feature std from train
│           │       └─ eps:  small constant
│           │
│           ├── raw_shift/               # ARTIFACT 2: Raw + Temporal Shift (Branch 1, augmented)
│           │   ├── data.npz
│           │   │   ├─ X_train: (2608, 3, 64, 192)  ← 4× more samples
│           │   │   ├─ y_train: (2608,)
│           │   │   ├─ X_val:   (140, 3, 64, 192)   ← unchanged
│           │   │   ├─ y_val:   (140,)
│           │   │   ├─ X_test:  (139, 3, 64, 192)   ← unchanged
│           │   │   └─ y_test:  (139,)
│           │   └── normalization_stats.npz
│           │       ├─ mean: computed only from original 652 train samples
│           │       ├─ std:  computed only from original 652 train samples
│           │       └─ eps:  small constant
│           │
│           ├── phase_hampel/            # ARTIFACT 3: Denoised (Branch 2, no augment)
│           │   ├── data.npz
│           │   │   ├─ X_train: (652, 3, 64, 192)
│           │   │   ├─ y_train: (652,)
│           │   │   ├─ X_val:   (140, 3, 64, 192)
│           │   │   ├─ y_val:   (140,)
│           │   │   ├─ X_test:  (139, 3, 64, 192)
│           │   │   └─ y_test:  (139,)
│           │   └── normalization_stats.npz
│           │
│           └── phase_hampel_shift/      # ARTIFACT 4: Denoised + Shift (Branch 2, augmented)
│               ├── data.npz
│               │   ├─ X_train: (2608, 3, 64, 192)  ← 4× more samples
│               │   ├─ y_train: (2608,)
│               │   ├─ X_val:   (140, 3, 64, 192)
│               │   ├─ y_val:   (140,)
│               │   ├─ X_test:  (139, 3, 64, 192)
│               │   └─ y_test:  (139,)
│               └── normalization_stats.npz
│
├── Phần 2 mô hình/                      # TRAINING (Part 2)
│   ├── README.md
│   ├── workflow_all.ipynb               # Main training notebook (35 runs)
│   ├── gru_baseline.py
│   ├── attention_gru.py
│   ├── multi_output_attention_gru.py
│   ├── train_gru.py
│   ├── train_attention_gru.py
│   ├── train_multi_output_best_model.py
│   │
│   └── runs/                            # Training outputs
│       ├── gru_baseline/
│       │   ├── S1_G01_raw_h64_b32_lr001/
│       │   ├── S1_G02_raw_shift_h64_anchor/
│       │   └── ... (7 Stage 1 runs)
│       │
│       ├── attention_gru/
│       │   ├── S2_A01_top_gru_a32/
│       │   ├── S2_A02_top_gru_a48/
│       │   └── ... (10 Stage 2 runs)
│       │
│       └── multi_output_best_model/
│           ├── S3_M01_baseline_linear/
│           ├── S3_M02_baseline_mlp/
│           ├── ... (10 Stage 3 runs)
│           ├── S4_F01_attention_top_seed43/
│           ├── S4_F02_attention_second_seed43/
│           ├── ... (8 Stage 4 runs)
│
├── Phần 3 đánh giá/                     # EVALUATION (Part 3)
│   ├── README.md
│   ├── evaluation_dashboard.ipynb
│   ├── stage_comparison.ipynb
│   ├── ablation_analysis.ipynb
│   ├── test_metrics_report.ipynb
│   └── reports/
│       ├── stage_summary.csv
│       ├── model_ranking.csv
│       ├── test_results.csv
│       └── final_report.md
│
└── README.md                            # Main project README
```

---

## 🔄 Data Transformation at Each Step

### **Branch 1: RAW Data Processing**

| Step | Input | Operation | Output | Size |
|------|-------|-----------|--------|------|
| 1.1 | Raw CSI from dataset | Window 192 frames | (3, 64, 192) per sample | 652 train |
| 1.2 | 652 samples | Temporal shift + crops (stride=16, max=4) | 4× multiplication | 2608 train |
| - | Val/Test | No augmentation | Unchanged | 140/139 |

### **Branch 2: Denoised Data Processing**

| Step | Input | Operation | Output | Size |
|------|-------|-----------|--------|------|
| 2.1 | Raw CSI | Phase Hampel filter | (3, 64, 192) denoised | Still raw files |
| 2.2 | Denoised CSI | Window 192 frames | (3, 64, 192) per sample | 652 train |
| 2.3 | 652 samples | Temporal shift + crops (stride=16, max=4) | 4× multiplication | 2608 train |
| - | Val/Test | No augmentation | Unchanged | 140/139 |

### **Unified Post-Processing (All 4 Profiles)**

| Step | Input | Operation | Output | Per Profile |
|------|-------|-----------|--------|-------------|
| 3 | (3, 64, 192) per sample | Organize into train/val/test split | data.npz with X, y | 4 profiles |
| 4 | X_train from data.npz | Fit mean/std on TRAIN ONLY | normalization_stats.npz | 4 profiles |
| 5 | X + stats | Normalize, optional noise on train, reshape (3, 64, 192) → (192, 192) | Ready for GRU | (B, 192, 192) batch |

---

## 📊 4 Data Profiles Summary

| Profile | Denoise | Augment | Train Size | Val Size | Test Size | Use Case |
|---------|---------|---------|-----------|----------|-----------|----------|
| **raw** | ❌ | ❌ | 652 | 140 | 139 | Baseline, raw only |
| **raw_shift** | ❌ | ✅ | 2608 | 140 | 139 | **Main profile** |
| **phase_hampel** | ✅ | ❌ | 652 | 140 | 139 | Denoise baseline |
| **phase_hampel_shift** | ✅ | ✅ | 2608 | 140 | 139 | Denoise + augment |

---

## 🎯 Key Principles

1. **Normalization Stats**: Computed on **TRAIN ONLY** → applied uniformly to train/val/test
2. **Augmentation**: Applied **TRAIN ONLY** → val/test unchanged
3. **Noise (Runtime)**: Added **TRAIN ONLY** during DataLoader → val/test always clean
4. **Splits**: **Fixed at Step 1** → same val/test in all profiles
5. **Two Branches**: Parallel processing allows comparing raw vs denoised independently

---

## 📝 Usage in Part 2 (Training)

Part 2 reads from:

```
Phần 1 tiền xử lí/processed/windows/<profile>/
```

For each profile, loads:
- `data.npz` (train/val/test data)
- `normalization_stats.npz` (mean/std/eps)

Then creates a Dataset:
```python
dataset = CSI_Dataset(
    profile='raw_shift',  # or 'raw', 'phase_hampel', 'phase_hampel_shift'
    split='train',        # 'val' or 'test'
    noise_sigma=0.01,     # runtime noise for train only
)
```

DataLoader produces batches of shape `(B, 192, 192)` ready for GRU.

---

## 🔍 Quick Reference: Finding Data

| Question | Location |
|----------|----------|
| Where are raw CSI files? | `Dataset_CSI_3D_v2/session1/data_P*.csv` |
| Where is denoised CSI? | `Phần 1 tiền xử lí/processed/denoised/phase_hampel/` |
| Where are the 4 profiles? | `Phần 1 tiền xử lí/processed/windows/{raw, raw_shift, phase_hampel, phase_hampel_shift}/` |
| Where are normalization stats? | `Phần 1 tiền xử lí/processed/windows/<profile>/normalization_stats.npz` |
| Where are training results? | `Phần 2 mô hình/runs/{gru_baseline, attention_gru, multi_output_best_model}/` |

---

**Last Updated:** June 16, 2026
