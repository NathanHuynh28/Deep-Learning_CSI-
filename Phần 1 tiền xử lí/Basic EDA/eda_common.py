
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RX_ORDER = ("RX1", "RX2", "RX3")
CSI_VALUES_PER_RX = 128
SUBCARRIERS_PER_RX = CSI_VALUES_PER_RX // 2
EXPECTED_RAW_COLUMNS = 438
GRID_ROWS = 5
GRID_COLS = 5
CELL_IDS = tuple(f"C{index:02d}" for index in range(1, GRID_ROWS * GRID_COLS + 1))

BASE_COLUMNS = (
    "session_id",
    "calibration_id",
    "layout_id",
    "channel",
    "sample_id",
    "person",
    "presence",
    "cell_id",
    "cell_row",
    "cell_col",
    "center_x_m",
    "center_y_m",
    "cell_width_m",
    "cell_depth_m",
    "occupied_cells",
    "coarse_pose",
    "orientation_label",
    "orientation_deg",
    "height_band",
    "footprint_length_m",
    "footprint_width_m",
    "footprint_yaw_deg",
    "label_quality",
    "trial",
    "frame_idx",
    "seq_num",
    "host_ts_ms",
)

MANIFEST_REQUIRED_COLUMNS = (
    "sample_id",
    "session_id",
    "calibration_id",
    "layout_id",
    "person",
    "presence",
    "cell_id",
    "center_x_m",
    "center_y_m",
    "occupied_cells",
    "coarse_pose",
    "orientation_label",
    "trial",
    "data_file",
    "frames",
    "status",
)

QUALITY_REQUIRED_COLUMNS = (
    "sample_id",
    "status",
    "total_synced_frames",
    "clean_frames",
    "rx1_frames",
    "rx2_frames",
    "rx3_frames",
    "duplicate_lines",
    "malformed_lines",
    "missing_seq_max_gap",
    "rssi_min",
    "rssi_mean",
    "rssi_max",
    "reason",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def part1_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_session_dir() -> Path:
    return project_root() / "Dataset_CSI_3D_v2" / "session1"


def default_figures_dir() -> Path:
    return part1_dir() / "figures" / "eda"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(session_dir: Path) -> Path:
    return session_dir / "manifest.csv"


def quality_path(session_dir: Path) -> Path:
    return session_dir / "quality.csv"


def raw_files(session_dir: Path) -> list[Path]:
    return sorted(session_dir.glob("data_P*.csv"))


def resolve_session_file(session_dir: Path, data_file: object) -> Path:
    base = session_dir.resolve()
    relative_path = Path(str(data_file))
    if relative_path.is_absolute():
        raise ValueError(f"data_file must be relative to session directory: {data_file}")
    candidate = (base / relative_path).resolve()
    try:
        _ = candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"data_file escapes session directory: {data_file}") from exc
    if not candidate.is_file():
        raise ValueError(f"data_file does not exist: {data_file}")
    return candidate


def parse_rx_list(value: str) -> list[str]:
    selected = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = [rx for rx in selected if rx not in RX_ORDER]
    if invalid:
        raise ValueError(f"Invalid RX names: {', '.join(invalid)}. Expected: {', '.join(RX_ORDER)}")
    if not selected:
        raise ValueError("At least one RX must be selected.")
    return selected


def csi_columns(rx: str) -> list[str]:
    return [f"{rx}_csi_{index:03d}" for index in range(CSI_VALUES_PER_RX)]


def rx_metadata_columns(rx: str) -> list[str]:
    return [
        f"{rx}_mac_time",
        f"{rx}_rssi",
        f"{rx}_channel",
        f"{rx}_rate",
        f"{rx}_cwb",
        f"{rx}_rx_state",
        f"{rx}_first_word_invalid",
        f"{rx}_csi_len",
        f"{rx}_src_mac",
    ]


def required_raw_columns(rx_list: Sequence[str] = RX_ORDER) -> list[str]:
    columns: list[str] = list(BASE_COLUMNS)
    for rx in rx_list:
        columns.extend(rx_metadata_columns(rx))
        columns.extend(csi_columns(rx))
    return columns


def read_header(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    present = set(columns)
    return [column for column in required if column not in present]


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = missing_columns(frame.columns, MANIFEST_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Missing manifest columns in {path}: {missing}")
    return frame


def load_quality(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = missing_columns(frame.columns, QUALITY_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Missing quality columns in {path}: {missing}")
    return frame


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def values_to_amplitude(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != CSI_VALUES_PER_RX:
        raise ValueError(f"Expected {CSI_VALUES_PER_RX} CSI values, got {values.shape[-1]}")
    iq_pairs = values.reshape(-1, SUBCARRIERS_PER_RX, 2)
    amplitude = np.hypot(iq_pairs[..., 0], iq_pairs[..., 1])
    return amplitude[0] if values.ndim == 1 else amplitude


def set_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )
