# Phan 2 - Mo hinh va huan luyen

README nay ho tro viet bao cao va van hanh Phan 2. File chinh can chay la `workflow_all.ipynb`. Workflow gom 4 stage voi dung 35 planned training calls. Moi training cell nen chay mot model, khong gom nhieu model vao mot cell.

## Muc dich

Phan 2 bien artifact tien xu li tu Phan 1 thanh ket qua model co the dua vao bao cao. Trong bao cao, nen trinh bay Phan 2 nhu mot thiet ke thuc nghiem co kiem soat:

1. Tao baseline GRU cho chuoi CSI.
2. Them Attention-GRU de kiem tra temporal attention.
3. Kiem tra multi-output de hoc dong thoi `presence`, `cell`, `pose`, `center`.
4. Dung validation metrics de chon model qua tung stage.
5. Chi dung test metrics cho bao cao cuoi, khong dung de tune.

Workflow gioi han 35 run de du so sanh ky thuat deep learning, nhung khong bien thanh brute-force search tren dataset nho.

## Dau vao

Phan 2 doc cac profile da tao tu Phan 1:

```text
Phan 1 tien xu li/processed/windows/
|-- raw/
|-- raw_shift/
|-- phase_hampel/
|-- phase_hampel_shift/
```

| Profile | Train | Val | Test | Vai tro |
| --- | ---: | ---: | ---: | --- |
| `raw` | 652 | 140 | 139 | baseline doi chung, khong temporal shift |
| `raw_shift` | 2608 | 140 | 139 | nhanh chinh, train set lon hon |
| `phase_hampel` | 652 | 140 | 139 | denoise baseline |
| `phase_hampel_shift` | 2608 | 140 | 139 | denoise + shift, dung kiem tra stability |

Co 6 kich ban train vi Gaussian noise la runtime augmentation, khong tao them `data.npz`:

1. `raw`
2. `raw_shift`
3. `raw_shift + noise 0.01`
4. `phase_hampel`
5. `phase_hampel_shift`
6. `phase_hampel_shift + noise 0.01`

Tensor tu Phan 1 co dang `(3, 64, 192)`. Dataset runtime normalize, co the them Gaussian noise tren train split, roi reshape thanh chuoi GRU `(192, 192)`. Trong DataLoader, batch co dang `(B, 192, 192)`.

## File va workflow chinh

```text
Phan 2 mo hinh/
|-- workflow_all.ipynb                  # notebook chinh de Run All
|-- workflow.ipynb                      # reference GRU cu
|-- workflow_attention_gru.ipynb        # reference Attention-GRU cu
|-- workflow_multi_output_best_model.ipynb
|-- gru_baseline.py                     # model GRU baseline
|-- attention_gru.py                    # model Attention-GRU
|-- multi_output_attention_gru.py       # model multi-output
|-- train_gru.py                        # trainer GRU
|-- train_attention_gru.py              # trainer Attention-GRU
|-- train_multi_output_best_model.py    # trainer/selector multi-output
|-- runs/                               # output moi khi train
|-- README.md
```

Ghi chu: README dung ten folder khong dau trong vi du de de doc. Tren may hien tai, folder that co dau tieng Viet.

De nop bai va chay workflow hien tai, dung `workflow_all.ipynb`. Cac notebook con lai la reference, co the dung de doi chieu y tuong nhung khong phai workflow 4 stage hien tai. Khong can chay rieng cac file `.py` vi notebook se goi trainer/model tuong ung.

## Cach chay

1. Hoan tat Phan 1 va giu nguyen `processed/windows/`.
2. Mo Jupyter.
3. Mo `Phan 2 mo hinh/workflow_all.ipynb`.
4. Chon kernel Python co `torch`, `numpy`, `pandas`, `tqdm`.
5. Kiem tra `torch.cuda.is_available()` neu muon dung GPU.
6. Bam `Run All` de chay tron bo 35 training calls.
7. Khong chay song song notebook training khac trong luc workflow nay dang chay.
8. Doc dashboard theo thu tu Stage 1, Stage 2, Stage 3, Stage 4.

May nen cam sac va tat sleep/hibernate. Khong can xoa run cu truoc khi chay.

## Thiet ke thuc nghiem

Workflow co 4 stage:

| Stage | Prefix | So run | Vai tro | Selector |
| --- | --- | ---: | --- | --- |
| Stage 1 | `S1_G*` | 7 | GRU screening | top 3 theo `best_val_macro_f1`, tie-break `best_val_loss` |
| Stage 2 | `S2_A*` | 10 | Attention-GRU narrowing | top 4 theo `best_val_macro_f1`, tie-break `best_val_loss` |
| Stage 3 | `S3_M*` | 10 | Multi-output cell bottleneck | top 3 theo `best_val_composite_score`, tie-break `val_loss` |
| Stage 4 | `S4_F*` | 8 | final confirmation va ablation | xac nhan top configs, khong mo search moi |
| Tong | | 35 | | |

`raw_shift` duoc dung nhieu nhat vi train set lon hon va ket qua cu on dinh hon. `phase_hampel_shift` duoc giu de xem denoise co giup stability khong. `raw` va `phase_hampel` la doi chung, nen khong chia deu budget cho moi profile.

## Ky thuat model

### GRU cho chuoi CSI

GRU nhan chuoi CSI `(192 time steps, 192 features)` va hoc quan he theo thoi gian. Stage 1 dung hidden state cuoi de phan loai 7 pose classes. Day la baseline ro rang, it thanh phan, de biet loi ich cua cac ky thuat sau co that su den tu attention hay multi-output khong.

```text
CSI window (B, 192, 192)
-> GRU
-> hidden state cuoi
-> Linear classifier 7 pose classes
```

### Attention-GRU va temporal attention

Stage 2 giu GRU lam encoder, nhung thay vi chi lay hidden state cuoi, temporal attention hoc trong so cho tung frame. Y tuong bao cao: khong phai frame nao trong cua so CSI cung quan trong nhu nhau, nen attention pooling co the tap trung vao doan tin hieu ro hon.

```text
CSI window (B, 192, 192)
-> GRU / optional BiGRU
-> temporal attention pooling
-> dropout
-> Linear classifier 7 pose classes
```

Optional BiGRU chi la controlled probe. BiGRU doc chuoi theo hai chieu thoi gian nen tang capacity va chi phi. Vi dataset nho, no khong nam o tat ca run, chi dung mot vai run de xem loi ich co dang tin hay chi la overfit.

### Multi-output shared trunk va heads

Stage 3 dung shared Attention-GRU trunk, sau do tach thanh cac head:

```text
CSI window (B, 192, 192)
-> shared Attention-GRU trunk
-> presence head: binary classification
-> cell head: 25-class cell classification
-> pose head: 7-class pose classification
-> center head: x/y regression
```

Cach nay cho phep model hoc representation chung tu CSI, dong thoi bao cao duoc nhieu muc tieu. `cell` la bottleneck chinh vi support thap va 25 lop kho hon pose. Vi vay Stage 3 tap trung vao cell head va loss cho cell.

MLP cell head co dang `LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear`. So voi linear head, MLP head them phi tuyen tinh rieng cho `cell`, giup kiem tra xem bottleneck den tu head qua yeu hay tu du lieu.

## Ky thuat huan luyen

| Ky thuat | Dung de lam gi | Ghi chu bao cao |
| --- | --- | --- |
| AdamW | toi uu trong so voi weight decay tach rieng | phu hop cho neural network nho va vua |
| ReduceLROnPlateau | giam learning rate khi validation metric dung yen | khac early stopping, scheduler khong dung training |
| Early stopping | dung training khi validation khong cai thien | checkpoint tot nhat la `best_model.pt` theo validation |
| Gradient clipping | giam nguy co gradient qua lon trong RNN | quan trong voi GRU tren chuoi dai |
| Runtime Gaussian noise | them noise nhe chi tren train split | val/test khong them noise, dung de regularize |
| Focal loss | tang trong so cho example kho | dung chu yeu cho cell bottleneck |
| Effective-number class weighting | can bang lop theo tan suat hieu dung | giam anh huong lop nhieu mau |
| Label smoothing | lam target bot qua chac | co san trong config, mac dinh `0.0` neu khong bat |
| Top-k validation selector | chon cau hinh tot nhat qua stage sau | dung validation, khong dung test |
| Composite score | gom nhieu metric multi-output | dung cho Stage 3/4 multi-output |

Tat ca training run dung validation metric de chon best epoch va checkpoint. Test set khong tham gia early stopping, selector, loss, hay checkpoint.

## Metric va cach chon model

Voi single-output pose model, metric chinh la `best_val_macro_f1`, tie-break bang `best_val_loss`. Voi multi-output, metric chinh la `best_val_composite_score`, tie-break bang `val_loss`. Test metric chi doc sau khi da chon xong bang validation.

Can phan biet hai nhom metric cho `cell`:

1. Exact cell metric nhu `cell_masked_macro_f1` yeu cau du doan dung chinh xac cell.
2. Relaxed localization metric cho phep danh gia muc do gan dung tren luoi 5x5.

Relaxed metrics chi de viet bao cao, khong thay the exact `cell_masked_*`, khong anh huong selector, loss, checkpoint, hay training loop.

Cac relaxed metrics co the tinh post-hoc tu `predictions.csv`:

| Metric | Y nghia |
| --- | --- |
| `cell_relaxed_soft_score` | exact = `1.0`, ngang/doc ke ben = `0.4`, cheo = `0.3`, xa hon = `0.0` |
| `cell_within_1cell` | ti le du doan co Chebyshev distance <= 1 cell |
| `cell_within_2cell` | ti le du doan co Chebyshev distance <= 2 cell |
| `cell_grid_l1_mean` | khoang cach Manhattan trung binh tren luoi 5x5 |
| `cell_grid_linf_mean` | khoang cach Chebyshev trung binh tren luoi 5x5 |

Cac metric nay chi tinh tren row co `cell_label_human >= 0`, voi `row = cell // 5`, `col = cell % 5`. Neu da co `predictions.csv`, khong can full `Run All` lai. Chi can chay cac cell post-hoc sau final dashboard de doc `runs/multi_output_best_model/*/predictions.csv` va tao summary rieng, khong ghi de `metrics.json`, `predictions.csv`, checkpoint, hay history cua tung run.

## Planned runs

### Stage 1: GRU screening

Selector sau stage: top 3 theo `best_val_macro_f1`, tie-break `best_val_loss`.

| Run | Profile | Hidden | Layers | Dropout | Batch | LR | Params | Muc dich |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S1_G01 | raw | 64 | 1 | 0.0 | 32 | 1e-3 | 49,991 | raw baseline |
| S1_G02 | raw_shift | 64 | 1 | 0.0 | 32 | 1e-3 | 49,991 | anchor tu ket qua G02 cu |
| S1_G03 | raw_shift | 96 | 1 | 0.0 | 32 | 1e-3 | 84,199 | capacity trung binh |
| S1_G04 | raw_shift | 128 | 1 | 0.0 | 32 | 1e-3 | 124,551 | capacity cao, can xem overfit |
| S1_G05 | raw_shift | 64 | 2 | 0.3 | 32 | 1e-3 | 74,951 | stacked GRU nho |
| S1_G06 | raw_shift | 96 | 2 | 0.3 | 32 | 1e-3 | 140,071 | stacked GRU trung binh |
| S1_G07 | raw_shift | 96 | 1 | 0.0 | 64 | 5e-4 | 84,199 | batch/LR stability |

So can doc: `best_val_macro_f1`, `test_macro_f1`, `epochs_ran`, `lr` trong `history.csv`.

### Stage 2: Attention-GRU narrowing

Selector sau stage: top 4 theo `best_val_macro_f1`, tie-break `best_val_loss`.

| Run | Profile | Hidden | Att dim | Layers | Dropout | Batch | LR | Noise | BiGRU | Params | Muc dich |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| S2_A01 | selected | selected | 32 | selected | 0.2 | selected | selected | 0.0 | selected | depends | top GRU + attention a32 |
| S2_A02 | selected | selected | 48 | selected | 0.2 | selected | selected | 0.0 | selected | depends | top GRU + attention a48 |
| S2_A03 | selected | selected | 32 | selected | 0.2 | selected | selected | 0.0 | selected | depends | second GRU branch |
| S2_A04 | selected | selected | 32 | 2 | 0.3 | selected | selected | 0.0 | selected | depends | stacked attention branch |
| S2_A05 | raw_shift | 128 | 32 | 1 | 0.3 | 32 | 1e-3 | 0.0 | no | 128,712 | h128 regularized |
| S2_A06 | raw_shift | 96 | 64 | 1 | 0.3 | 32 | 1e-3 | 0.0 | no | 90,472 | attention dim upper |
| S2_A07 | phase_hampel_shift | 96 | 32 | 1 | 0.2 | 32 | 1e-3 | 0.0 | no | 87,336 | denoise+shift stability |
| S2_A08 | phase_hampel_shift | 96 | 32 | 1 | 0.3 | 32 | 5e-4 | 0.01 | no | 87,336 | denoise + noise regularization |
| S2_A09 | raw_shift | 64 | 32 | 1 | 0.2 | 32 | 1e-3 | 0.0 | yes | 104,136 | BiGRU small probe |
| S2_A10 | raw_shift | 96 | 48 | 1 | 0.3 | 32 | 5e-4 | 0.0 | no | 88,904 | A12-style regularized |

So can doc: `best_val_macro_f1`, `best_val_loss`, `test_macro_f1`, val-test gap. Khong chon model theo test.

### Stage 3: Multi-output cell bottleneck

Selector sau stage: top 3 theo `best_val_composite_score`, tie-break `val_loss`.

| Run | Profile | Hidden/Att | Head | Cell loss | Cell weight | Layers | Dropout | Noise | BiGRU | Params | Muc dich |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| S3_M01 | selected | selected | linear | CE | 1.0 | selected | selected | selected | selected | depends | baseline tu top attention |
| S3_M02 | selected | selected | MLP | CE | 1.0 | selected | selected | selected | selected | depends | kiem tra MLP cell head |
| S3_M03 | raw_shift | h128/a32 | MLP | CE | 1.0 | 1 | 0.3 | 0.0 | no | 139,236 | raw_shift h128 MLP |
| S3_M04 | raw_shift | h128/a32 | MLP | CE | 2.0 | 1 | 0.3 | 0.0 | no | 139,236 | tang uu tien cell |
| S3_M05 | raw_shift | h128/a32 | MLP | focal gamma 2.0 | 2.0 | 1 | 0.3 | 0.0 | no | 139,236 | hard cell examples |
| S3_M06 | raw_shift | h128/a32 | MLP | focal + effective-number | 2.0 | 1 | 0.3 | 0.0 | no | 139,236 | class-balanced focal |
| S3_M07 | phase_hampel_shift | h96/a32 | MLP | CE | 1.0 | 1 | 0.2 | 0.0 | no | 93,700 | denoise multi baseline |
| S3_M08 | phase_hampel_shift | h96/a32 | MLP | CE | 1.0 | 1 | 0.3 | 0.01 | no | 93,700 | denoise + noise |
| S3_M09 | raw_shift | h96/a32 | MLP | CE | 1.0 | 2 | 0.3 | 0.0 | no | 149,572 | stacked multi |
| S3_M10 | raw_shift | h64/a32 | MLP | CE | 1.0 | 1 | 0.2 | 0.0 | yes | 114,660 | BiGRU multi probe |

So can doc nhat: `val_cell_masked_macro_f1`, `test_cell_masked_macro_f1`, `best_val_composite_score`, `test_pose_macro_f1`, `center_mean_error_m`.

### Stage 4: Final confirmation va ablation

Stage nay dung top configs tu Stage 2/3, nen params phu thuoc vao run duoc chon. Muc dich la xac nhan xu huong bang seed khac va ablation, khong mo search moi.

| Run | Loai | Muc dich |
| --- | --- | --- |
| S4_F01 | Attention | top attention seed 43 |
| S4_F02 | Attention | second attention seed 43 |
| S4_F03 | Multi-output | top multi seed 43 |
| S4_F04 | Multi-output | top multi seed 44 |
| S4_F05 | Multi-output | second multi seed 43 |
| S4_F06 | Multi-output | linear-vs-MLP ablation |
| S4_F07 | Multi-output | CE-vs-focal ablation |
| S4_F08 | Multi-output | layer/BiGRU ablation |

So can doc: seed co lap lai xu huong khong, MLP co hon linear khong, focal co giup cell khong, va 2-layer/BiGRU co dang doi chi phi khong.

## Num layers 1/2

Workflow chi dung `num_layers=1/2` trong grid chinh vi day la quyet dinh generalization, khong phai gioi han GPU. RTX A2000 8GB du chay cac model hien tai, nhung dataset nho va val/test it mau nen GRU sau hon de overfit.

| Gia tri | Vai tro |
| --- | --- |
| `num_layers=1` | baseline mac dinh, it tham so, de so sanh |
| `num_layers=2` | probe capacity co kiem soat cho GRU, Attention-GRU, Multi-output |
| `num_layers=3` | khong dua vao grid chinh vi tang thoi gian va rui ro overfit |

Neu can them `num_layers=3`, chi nen xem nhu mot ablation cuoi tren backbone tot nhat. Trong workflow nay, nen uu tien `hidden_size`, `attention_dim`, BiGRU probe, MLP cell head, focal loss, effective-number weighting, dropout, LR, va batch size hon la tang do sau cho moi run.

## Early stopping va patience

| Stage | Trainer | Ap dung cho run | Metric de xet best | Tie-break | Max epochs | Early stop patience | Scheduler patience |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Stage 1: Plain GRU | `train_gru.py` | `S1_G01` den `S1_G07` | `val_macro_f1` | `val_loss` thap hon khi F1 bang nhau | 80 | 12 | 4 |
| Stage 2: Attention-GRU | `train_attention_gru.py` | `S2_A01` den `S2_A10` | `val_macro_f1` | `val_loss` thap hon khi F1 bang nhau | 100 | 15 | 5 |
| Stage 3: Multi-output | `train_multi_output_best_model.py` | `S3_M01` den `S3_M10` | `val_composite_score` | `val_loss` thap hon khi score bang nhau | 100 | 15 | 5 |
| Stage 4: Attention final | `train_attention_gru.py` | `S4_F01` den `S4_F02` | `val_macro_f1` | `val_loss` thap hon khi F1 bang nhau | 100 | 15 | 5 |
| Stage 4: Multi final/ablation | `train_multi_output_best_model.py` | `S4_F03` den `S4_F08` | `val_composite_score` | `val_loss` thap hon khi score bang nhau | 100 | 15 | 5 |

Trong moi epoch: train tren train split, evaluate tren validation split, luu `best_model.pt` neu validation tot hon, tang stale count neu khong tot hon, dung som khi stale count dat patience. `last_model.pt` van la epoch cuoi, nhung bao cao nen dua vao checkpoint tot nhat theo validation. `epochs_ran` trong `metrics.json` hoac dashboard cho biet run dung o epoch nao.

## Output sau khi chay

Moi run tao thu muc rieng:

```text
runs/gru_baseline/<experiment>/
runs/attention_gru/<experiment>/
runs/multi_output_best_model/<experiment>/
```

File quan trong:

| File | Noi dung |
| --- | --- |
| `config.json` | cau hinh run |
| `metrics.json` | tong ket metric, dashboard doc file nay |
| `history.csv` | train/val loss, metric, LR tung epoch |
| `predictions.csv` | du doan val/test, dung duoc cho relaxed metrics post-hoc |
| `best_model.pt` | checkpoint tot nhat theo validation metric |
| `last_model.pt` | checkpoint epoch cuoi |
| `selection.json` | rieng multi-output, ghi config selector/provenance |

## Rerun va archive

Notebook chi doc planned runs co prefix `S1_`, `S2_`, `S3_`, `S4_`, nen smoke run hoac run cu khong lam hong dashboard hien tai. Khong xoa truc tiep `runs/` hay artifact neu can cleanup. Hay archive/move artifact ra ngoai project:

```text
<archive_root>/<timestamp>/Phan 2 mo hinh/runs/
<archive_root>/<timestamp>/.../__pycache__/
```

Neu rerun cung mot cell thanh cong, file output trong run folder tuong ung se duoc ghi lai. Neu run bi dung giua chung hoac loi, khong doc dashboard cua experiment do cho den khi archive/move rieng folder experiment dang loi roi chay lai. Neu muon giu ca hai lan chay, doi `experiment_name` truoc khi rerun cell do.

Sau khi move `runs/`, notebook van train lai va tao output moi. Dashboard se khong doc duoc ket qua cu cho den khi copy hoac symlink `runs/` tu archive ve dung vi tri cu.

## Uoc tinh thoi gian

Can cu thuc te cu, workflow 23 run da chay khoang 17.25 phut tren may nay. Uoc tinh cho 35 run:

| Stage | So run | Uoc tinh |
| --- | ---: | ---: |
| Stage 1 GRU | 7 | 3-5 phut |
| Stage 2 Attention-GRU | 10 | 8-14 phut |
| Stage 3 Multi-output | 10 | 8-15 phut |
| Stage 4 Final/ablation | 8 | 6-12 phut |
| Tong | 35 | 25-46 phut |

Thoi gian dao dong vi early stopping co the dung som hoac muon, BiGRU/2-layer/MLP/focal cham hon baseline, va may co the ban viec khac. Neu moi truong cham bat thuong, tong co the gan 1 gio.

## Cach doc dashboard de viet bao cao

1. Stage 1 ranking: tim GRU backbone on dinh.
2. Stage 2 ranking: xem attention co cai thien pose validation khong.
3. Stage 3 ranking: xem multi-output va rieng `cell`.
4. Stage 4: xac nhan top model va doc ablation.
5. Cuoi cung: doc test metrics cua model da chon bang validation.

Trong bao cao, nen tach ro validation va test. Validation tra loi cau hoi "chon model nao". Test tra loi cau hoi "model da chon dat ket qua cuoi ra sao". Neu `cell_masked_macro_f1` thap, can ghi ro day la diem yeu, khong che bang relaxed metrics.

## Ket qua hien tai de dua vao bao cao

Day la cac moc so lieu sau lan chay full workflow hien tai. Nen dung chung nhu bang tong hop, khong dung de tiep tuc tune model.

| Nhom ket qua | Run | Metric chinh |
| --- | --- | --- |
| Best pose test overall | `S1_G02_raw_shift_h64_anchor` | `test_macro_f1 = 0.6877`, `test_accuracy = 0.7338` |
| Best GRU theo validation | `S1_G04_raw_shift_h128` | `val_macro_f1 = 0.7474`, `test_macro_f1 = 0.6551` |
| Best Attention theo validation | `S2_A01_top_gru_a32` | `val_macro_f1 = 0.7242`, `test_macro_f1 = 0.5874` |
| Best Attention final | `S4_F02_attention_second_seed43` | `val_macro_f1 = 0.7343`, `test_macro_f1 = 0.6621`, `test_accuracy = 0.7122` |
| Best Multi-output final | `S4_F06_multi_linear_ablation` | `val_composite_score = 0.5682`, `test_pose_macro_f1 = 0.6696` |
| Best relaxed localization test | `S4_F06_multi_linear_ablation` | exact `0.2692`, within 1 cell `0.6692`, within 2 cells `0.8615`, soft score `0.4192` |

Voi multi-output best final, cac metric can bao cao them la `test_presence_macro_f1 = 0.6919`, `test_presence_accuracy = 0.9065`, `test_cell_masked_macro_f1 = 0.2066`, `test_cell_masked_accuracy = 0.2692`, va `test_center_mean_error_m = 0.7246`. Dien giai nen can bang: pose va presence kha on, center error duoi 1m, nhung exact cell van yeu do 25 class va support cell thap.

## CLI guard

Notebook la cach van hanh chinh. `train_multi_output_best_model.py` chan full CLI training neu khong truyen `--run-full`. Smoke CLI phai gioi han batch.

Vi du smoke:

```bash
python train_multi_output_best_model.py --experiment-name smoke_check --max-epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --disable-progress --no-selector
```

Vi du full CLI co chu dich:

```bash
python train_multi_output_best_model.py --experiment-name S3_M01_top_attention_linear --run-full
```

## Ghi chu han che cho bao cao

Dataset nho, dac biet val/test chi co 140/139 mau. Support cua mot so cell thap, nen cell classification la bottleneck va ket qua co the dao dong theo seed. Khong nen noi mo hinh da giai quyet bai toan localization mot cach chac chan. Nen noi workflow da so sanh GRU, Attention-GRU, va Multi-output co kiem soat, dong thoi bao cao ro exact cell metric va relaxed localization metric.

Neu checklist dau vao dat, co the bam `Run All` trong `workflow_all.ipynb`. Workflow da co early stopping, scheduler, top-k selector, ablation Stage 4, va chinh sach archive de giu lai artifact khi can cleanup.
