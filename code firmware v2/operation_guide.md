# Huong Dan Thu CSI V2 - Perimeter Part 1

Guide nay dung cho mot run Part 1 lon bang 4 board ESP32-S3: 1 TX + 3 RX, moi side cua perimeter co dung mot module. Output mong muon la 2.5D occupancy/footprint probability cloud rendered in 3D tren 25 zone, center tuong doi, coarse pose, va huong than. Khong claim true 3D reconstruction, body mesh/skeleton, camera-free full body pose, hay 17 keypoints cho Part 1.

Nguon thu hien tai nam trong `code firmware v2`. code để nạp IDE `firmware`. 
## 1. Active Layout: layout1

Quy uoc toa do bat buoc:

- Nhin top-down tu tren xuong.
- Goc (0,0) la goc tren-trai cua luoi 5 cot x 5 hang.
- `+x` tang tu west sang east.
- `+y` tang tu north sang south.
- Kich thuoc grid: 3.00m x 3.00m, moi zone 0.60m x 0.60m.
- Layout type: `perimeter_one_module_per_side`.

```text
                         North / -y
                         RX1 (1.50, -0.50, h=1.00)
                               |
                               v
        +------+------+------+------+------+   East / +x
        | C01  | C02  | C03  | C04  | C05  |   RX2 (3.50, 1.50, h=1.00)
        +------+------+------+------+------+         <---
West    | C06  | C07  | C08  | C09  | C10  |
TX ---> +------+------+------+------+------+   origin top-left of grid
(-0.50, | C11  | C12  | C13  | C14  | C15  |   +x west->east
 1.50)  +------+------+------+------+------+   +y north->south
h=1.00  | C16  | C17  | C18  | C19  | C20  |
        +------+------+------+------+------+
        | C21  | C22  | C23  | C24  | C25  |
        +------+------+------+------+------+
                                ^
                                |
                         RX3 (1.50, 3.50, h=1.00)
                         South / +y
```

Node side roles:

| Node | Side | Coordinate |
|---|---|---|
| TX | west/left | x=-0.50, y=1.50, height=1.00 |
| RX1 | north/top | x=1.50, y=-0.50, height=1.00 |
| RX2 | east/right | x=3.50, y=1.50, height=1.00 |
| RX3 | south/bottom | x=1.50, y=3.50, height=1.00 |

Cac toa do nay la outside-grid offsets. Neu phong thuc te khong dat duoc node ngoai grid, do lai toa do that va tao `layout_id`/`calibration_id` moi, khong ghi tiep vao layout nay.

## 2. Huong Than

Huong la huong mat/nguc actor theo map top-down.

| Label | Degree | Nghia vat ly |
|---|---:|---|
| `Empty` | 0 | Khong co nguoi trong vung thu |
| `Facing_RX` | 0 | +x, nhin east ve RX2 |
| `Facing_TX` | 180 | -x, nhin west ve TX |
| `Facing_Left` | 270 | -y, nhin north ve RX1 |
| `Facing_Right` | 90 | +y, nhin south ve RX3 |
| `Diagonal_RX1` | 315 | +x/-y, huong northeast ve quadrant RX1/RX2 |
| `Diagonal_RX3` | 45 | +x/+y, huong southeast ve quadrant RX2/RX3 |

```text
Facing_Left / north / RX1
          ^
          |
Facing_TX < actor > Facing_RX
west/TX       |      east/RX2
              v
Facing_Right / south / RX3
```

## 3. Controller Session

Active constants trong `controller.py`:

```python
SESSION_ID = "session1"
ROOM_NAME = "room1"
LAYOUT_ID = "layout1"
LAYOUT_TYPE = "perimeter_one_module_per_side"
COLLECTION_PLAN_ID = "PART1_25ZONE_PERIMETER_V1"
CALIBRATION_ID = "calibration1"
COLLECTION_MODE = "part1_large"
```

Operator thuong chi sua `COM_PORTS`, `CHANNEL`, `SESSION_ID`, `CALIBRATION_ID`, va toa do node neu da tao layout/calibration moi.

Empty-room samples dung ID `P0` va `NO_CELL` de tao sample/file ro rang, vi du `session1_P0_NO_CELL_Empty_Empty_T001`. Day chi la identifier cho phong trong, khong phai nguoi/cell vat ly. Label trong CSV/manifest van la `presence=0`, `cell_id=Empty`, `occupied_cells=''`, `orientation_label=Empty`.

## 4. Firmware Va CSV Schema Khong Doi

Firmware TX/RX khong doi label va khong doi serial schema:

```text
CSI_V2_13_METADATA_128_IQ
13 metadata fields + 128 I/Q values moi RX
100 Hz, SEND_INTERVAL_US = 10000UL
Baudrate controller/firmware giu 921600 cho cap 5m hien tai; khong tang baud neu chua test rieng voi cap ngan.
```

Frame CSV giu nguyen 438 cot:

```text
27 base label/metadata columns + 3 RX x (9 RX metadata + 128 CSI) = 438 columns
```

Layout la session-level metadata trong `session.md` va `calibration.json`, khong them cot vao raw frame CSV. Raw collection format van la CSV, khong doi sang NPZ.

## 5. Large Part 1 Collection Plan

Default `part1_large` la plan optimized perimeter `PART1_25ZONE_PERIMETER_V1` co 1370 samples. Dung `python controller.py --plan-only` de xem count, Empty checkpoints, pose/person balance, va duplicate check truoc khi mo serial.

Chien luoc session:

- Session training chinh: dung `session1`, `room1`, `layout1`, `calibration1`, mode `part1_large`, va thu du 1370 samples neu phan cung/QC on.
- Session sau de calibration/hieu chinh: tao `SESSION_ID`/`CALIBRATION_ID` moi, giu `LAYOUT_ID` neu toa do khong doi, va dung `smoke` hoac `pilot` de thu it hon.
- Neu node/phong/toa do thay doi, tao `LAYOUT_ID`/`CALIBRATION_ID` moi thay vi tron vao session training chinh.

| Block | Count | Ly do |
|---|---:|---|
| Empty checkpoints | 80 | 16 checkpoint x 5 trial, rai deu trong flow thu de kiem drift/false positive |
| Standing | 390 | Moi nguoi 78: interior 3x3 x 4 huong, perimeter 16 x 2 huong quan trong, center/corner diagonal |
| Sitting | 310 | Moi nguoi 62: interior 3x3 x 4 huong, perimeter inward, center/corner diagonal |
| Arms_Out | 150 | Moi nguoi 30: cross5 x 4 huong, corner inward, center diagonal |
| Lying | 190 | Moi nguoi 38: cross5 x 4 huong, corner/edge inward, C13/C08/C18 diagonal |
| Walk_in_place | 160 | Moi nguoi 32: cross5 x 4 huong, corner/edge inward |
| Transition | 90 | Moi nguoi 18: cross5 x 2 axis, corner/edge inward |
| Total | 1370 | Expected clean rows: 1370 x 300 = 411000 |

Core cross zones: `C08,C12,C13,C14,C18`. Real actor zones dung `C01..C25`; cac pose lon/dynamic uu tien cross, interior corners, edge centers, va inward/diagonal directions de giam label noise tren cell 0.60m. Khong doi 438-column raw CSV schema.

`smoke` va `pilot` van ton tai nhu cac check nho truoc khi thu that, va cung la mode nho cho calibration/hieu chinh sau nay. Run chinh Part 1 dung `part1_large`.

## 6. QC Checklist Truoc, Trong, Sau Thu

Truoc thu:

```text
[ ] Don phong, khong co nguoi khac di qua.
[ ] Dan luoi C01..C25 dung kich thuoc 3m x 3m.
[ ] Dat TX west, RX1 north, RX2 east, RX3 south dung toa do/calibration.
[ ] Anten cao 1.00m, cung huong, khong cham trong ca session.
[ ] Chup anh setup phong va ghi toa do neu khac default.
[ ] Kiem tra channel TX/RX/controller deu la 1, tru khi da chu dong doi ca ba noi va soak lai.
[ ] Kiem tra TX Serial Monitor `TX_MAC,...` va cap nhat dung `TX_MAC_FILTER` trong RX firmware truoc khi flash ca 3 RX.
[ ] Chay: python controller.py --plan-only
[ ] Chay: python controller.py --preflight-only
[ ] Neu can soak dai hon: python controller.py --soak-only --soak-seconds N
[ ] Preflight/soak PASS chi khi first_word_invalid=0, malformed=0, channel dung, hard firmware counters bang 0, va no_csi/output_busy trong nguong controller.
```

Trong thu:

```text
[ ] Actor dung center/footprint controller yeu cau.
[ ] Actor quay dung orientation theo map, khong theo trai/phai cua laptop.
[ ] Empty sample `P0/NO_CELL` phai khong co nguoi trong grid.
[ ] Transition dung pattern sit_down_then_stand_up.
[ ] FAIL thi thu lai dung sample do, khong skip neu khong co ly do.
```

Sau thu:

```text
[ ] `quality.csv` PASS co clean_frames=240.
[ ] `quality.csv` khong co fail reason `first_word_invalid_*`; neu co thi thu lai sample do.
[ ] `missing_seq <= 20` cho PASS.
[ ] `data_P0.csv`..`data_P5.csv` co 438 cot.
[ ] Moi RX co `RX*_csi_len = 128`.
[ ] Kiem tra firmware counters: output_busy/no_csi khong tang bat thuong, malformed=0, channel dung.
[ ] `calibration.json` co layout_type, collection_plan_id, node_side_roles, coordinate convention, orientation definitions, planned_sample_count=1370.
[ ] Khong tron session khac layout/channel vao `session1`.
```

## 7. File Dau Ra

```text
Dataset_CSI_3D_v2/
  session1/
    session.md
    calibration.json
    manifest.csv
    quality.csv
    data_P0.csv
    data_P1.csv
    data_P2.csv
    data_P3.csv
    data_P4.csv
    data_P5.csv
```

`calibration.json` la sidecar session-level ghi layout, side roles, RX order note, orientation definitions, collection plan summary, planned sample count, va schema. `manifest.csv` la sample index. `quality.csv` ghi pass/fail va signal QC.

## 8. Decision Log

- Active layout cho lan thu dau la perimeter one-module-per-side `layout1` trong `room1`, vung thu 3m x 3m, 25 zone.
- Active collection doi sang explicit `PART1_25ZONE_PERIMETER_V1`, 1370 samples, voi 80 Empty samples phan bo thanh 16 checkpoint va 1290 human samples can bang 258/person.
- CSV frame schema, firmware serial schema, 100Hz, va 128 I/Q moi RX giu nguyen.
- Baudrate giu 921600 cho setup cap 5m hien tai de giam rui ro noise; chi tang baud khi co cap ngan va soak validation rieng.
- Part 1 chi la 2.5D occupancy/footprint probability cloud rendered in 3D tren floor grid, khong claim true 3D body mesh/skeleton hay 17 keypoints.

## 9. Debug 3 RX Drop

Neu 1 RX va 2 RX pass nhung 3 RX cung luc bi rot rate, khong ha nguong `MIN_RX_RATE_HZ`. Flash lai ca 3 RX bang `rx_firmware.ino` hien tai va chay:

```text
python controller.py --preflight-only
python controller.py --soak-only --soak-seconds 30
```

Doc dong `firmware:` cua tung RX:

- `csi_cb` la tong CSI callback firmware nhin thay.
- `csi_seen` la CSI callback da qua length va TX MAC filter.
- `espnow_recv` la tong packet ESP-NOW RX firmware nhin thay.
- `espnow_valid` la ESP-NOW packet da qua magic/version/TX/channel/checksum.
- `seq_gap` tang thi RX dang mat packet tren duong radio/PHY/TX timing, khong phai loi parser CSV.
- `csi_mac_mismatch` tang thi kiem tra `TX_MAC_FILTER` hoac test tam `USE_TX_MAC_FILTER 0`.
- `csi_null` hoac `bad_csi_len` tang thi dang loi tang CSI config/driver.
- `output_busy` hoac `queue` tang thi moi nghi serial output bottleneck.

TX Serial Monitor van phai cho `STAT,seq,...` tang gan 100Hz. Chi thu dataset khi ca 3 RX pass preflight/soak voi nguong hien tai.

