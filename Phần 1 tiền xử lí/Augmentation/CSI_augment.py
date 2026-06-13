from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PART_DIR = THIS_DIR.parent
PROJECT_ROOT = PART_DIR.parent

WINDOWING_PATH = PART_DIR / "split_and_window.py"
WINDOWING_SPEC = importlib.util.spec_from_file_location("split_and_window", WINDOWING_PATH)
if WINDOWING_SPEC is None or WINDOWING_SPEC.loader is None:
    raise ImportError(f"Cannot load split_and_window.py from {WINDOWING_PATH}")
windowing = importlib.util.module_from_spec(WINDOWING_SPEC)
sys.modules[WINDOWING_SPEC.name] = windowing
WINDOWING_SPEC.loader.exec_module(windowing)


DEFAULT_PROCESSED_DIR = PART_DIR / "processed"
DEFAULT_SPLIT_PATH = DEFAULT_PROCESSED_DIR / "sample_split" / "split.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "Dataset_CSI_3D_v2" / "session1"
DEFAULT_DENOISED_ROOT = DEFAULT_PROCESSED_DIR / "denoised"
DEFAULT_STRIDE = 16
DEFAULT_MAX_CROPS_PER_SAMPLE = 4
DEFAULT_WINDOW_LEN = 192
GAUSSIAN_NOISE_SIGMAS = (0.01, 0.02)

WINDOW_COLUMNS = (
    "window_id",
    "split",
    "sample_id",
    "data_file",
    "start",
    "end",
    "frames",
    "presence",
    "cell_id",
    "coarse_pose",
    "center_x_m",
    "center_y_m",
    "augmentation",
    "crop_rank",
    "center_start",
    "offset_from_center",
)


@dataclass(frozen=True)
class TemporalShiftConfig:
    source_name: str
    output_name: str
    input_dir: Path
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    split_path: Path = DEFAULT_SPLIT_PATH
    stride: int = DEFAULT_STRIDE
    max_crops_per_sample: int = DEFAULT_MAX_CROPS_PER_SAMPLE
    window_len: int = DEFAULT_WINDOW_LEN
    overwrite: bool = False


@dataclass(frozen=True)
class AugmentResult:
    source_name: str
    output_name: str
    input_dir: Path
    output_dir: Path
    train_windows: int
    val_windows: int
    test_windows: int
    skipped_train_samples: int


def add_gaussian_noise(
    x: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Train-time Gaussian noise for normalized tensors; not saved to data.npz."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    generator = rng if rng is not None else np.random.default_rng()
    noise = generator.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return (x.astype(np.float32, copy=False) + noise).astype(np.float32)


def candidate_starts(n_frames: int, window_len: int, stride: int) -> list[int]:
    if n_frames < window_len:
        return []
    if stride <= 0:
        raise ValueError("stride must be positive")
    max_start = n_frames - window_len
    return list(range(0, max_start + 1, stride))


def centered_stride_starts(
    n_frames: int,
    window_len: int,
    stride: int,
    max_crops_per_sample: int,
) -> list[int]:
    if n_frames < window_len:
        return []
    if max_crops_per_sample <= 0:
        raise ValueError("max_crops_per_sample must be positive")
    max_start = n_frames - window_len
    center_start = (n_frames - window_len) // 2
    selected: list[int] = []

    def add_start(start: int) -> None:
        if 0 <= start <= max_start and start not in selected and len(selected) < max_crops_per_sample:
            selected.append(start)

    radius = 0
    while len(selected) < max_crops_per_sample and radius <= max_start + stride:
        if radius == 0:
            add_start(center_start)
        else:
            add_start(center_start - radius)
            add_start(center_start + radius)
        radius += stride

    boundary_candidates = candidate_starts(n_frames, window_len, stride)
    if max_start not in boundary_candidates:
        boundary_candidates.append(max_start)
    for start in sorted(boundary_candidates, key=lambda value: (abs(value - center_start), value)):
        add_start(start)
    return sorted(selected)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_profile_name(name: str, field_name: str) -> str:
    path = Path(name)
    if not name or name in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"{field_name} must be a simple profile name, got: {name!r}")
    return name


def child_dir(root: Path, name: str, field_name: str) -> Path:
    leaf = validate_profile_name(name, field_name)
    base = root.resolve()
    path = (base / leaf).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes root directory: {name!r}") from exc
    return path


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def base_arrays(base_window_dir: Path) -> Mapping[str, np.ndarray]:
    data_path = base_window_dir / "data.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing base window data: {data_path}")
    loaded = np.load(data_path)
    return {key: loaded[key] for key in loaded.files}


def check_label_maps(manifest: pd.DataFrame, base_window_dir: Path) -> dict[str, dict[str, int]]:
    labels = windowing.label_maps(manifest)
    base_labels_path = base_window_dir / "labels.json"
    if base_labels_path.is_file():
        base_labels = load_json(base_labels_path)
        if base_labels != labels:
            raise ValueError(f"Label map mismatch with {base_labels_path}")
    return labels


def split_side_rows(base_windows: pd.DataFrame, split_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    part = base_windows.loc[base_windows["split"].astype(str).eq(split_name)].copy()
    for _, row in part.iterrows():
        start = int(row["start"])
        record = row.to_dict()
        record.update(
            {
                "augmentation": "none",
                "crop_rank": 0,
                "center_start": start,
                "offset_from_center": 0,
            }
        )
        rows.append(record)
    return rows


def pack_split(
    x_values: list[np.ndarray],
    pose_values: list[int],
    cell_values: list[int],
    presence_values: list[int],
    center_values: list[np.ndarray],
) -> dict[str, np.ndarray]:
    if not x_values:
        return windowing.empty_arrays()
    return {
        "X": np.stack(x_values).astype(np.float32),
        "pose": np.asarray(pose_values, dtype=np.int64),
        "cell": np.asarray(cell_values, dtype=np.int64),
        "presence": np.asarray(presence_values, dtype=np.int64),
        "center": np.stack(center_values).astype(np.float32),
    }


def make_temporal_shift_dataset(
    config: TemporalShiftConfig,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    if config.stride <= 0:
        raise ValueError("stride must be positive")
    if config.max_crops_per_sample <= 0:
        raise ValueError("max_crops_per_sample must be positive")
    _ = validate_profile_name(config.source_name, "source_name")
    _ = validate_profile_name(config.output_name, "output_name")

    manifest = windowing.load_manifest(config.input_dir)
    split = windowing.load_split(config.split_path, manifest)
    labels = check_label_maps(manifest, config.processed_dir / "windows" / config.source_name)
    sample_meta = manifest.merge(split[["sample_id", "split"]], on="sample_id", how="inner")
    sample_meta = sample_meta.set_index("sample_id", drop=False)
    train_meta = sample_meta.loc[sample_meta["split"].astype(str).eq("train")].copy()
    wanted_by_file = train_meta.groupby("data_file")["sample_id"].apply(lambda values: set(values.astype(str)))

    x_train: list[np.ndarray] = []
    pose_train: list[int] = []
    cell_train: list[int] = []
    presence_train: list[int] = []
    center_train: list[np.ndarray] = []
    window_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    seen: set[str] = set()

    usecols = ["sample_id", "frame_idx"] + [column for rx in windowing.RX_ORDER for column in windowing.csi_columns(rx)]
    usecols += [f"{rx}_csi_len" for rx in windowing.RX_ORDER]

    for data_file, wanted_ids in wanted_by_file.items():
        csv_path = windowing.resolve_data_file(config.input_dir, data_file)
        data = pd.read_csv(csv_path, usecols=lambda column: column in usecols)
        if "frame_idx" in data.columns:
            data = data.sort_values(["sample_id", "frame_idx"])
        for sample_id, frame in data.groupby("sample_id", sort=False):
            sample_id = str(sample_id)
            if sample_id not in wanted_ids:
                continue
            seen.add(sample_id)
            meta = train_meta.loc[sample_id]
            starts = centered_stride_starts(len(frame), config.window_len, config.stride, config.max_crops_per_sample)
            if not starts:
                skipped_rows.append({"sample_id": sample_id, "split": "train", "frames": int(len(frame)), "reason": "too_short"})
                continue
            tensor = windowing.sample_to_tensor(frame)
            pose = labels["pose"][str(meta["coarse_pose"])]
            cell = labels["cell"][str(meta["cell_id"])]
            presence = int(meta["presence"])
            center = np.array([float(meta["center_x_m"]), float(meta["center_y_m"])], dtype=np.float32)
            center_start = windowing.center_window_start(len(frame))
            if center_start is None:
                skipped_rows.append({"sample_id": sample_id, "split": "train", "frames": int(len(frame)), "reason": "too_short"})
                continue

            for crop_rank, start in enumerate(starts):
                x_train.append(tensor[:, :, start : start + config.window_len])
                pose_train.append(pose)
                cell_train.append(cell)
                presence_train.append(presence)
                center_train.append(center)
                window_rows.append(
                    {
                        "window_id": len(window_rows),
                        "split": "train",
                        "sample_id": sample_id,
                        "data_file": data_file,
                        "start": int(start),
                        "end": int(start + config.window_len),
                        "frames": int(config.window_len),
                        "presence": presence,
                        "cell_id": meta["cell_id"],
                        "coarse_pose": meta["coarse_pose"],
                        "center_x_m": float(meta["center_x_m"]),
                        "center_y_m": float(meta["center_y_m"]),
                        "augmentation": "temporal_shift",
                        "crop_rank": int(crop_rank),
                        "center_start": int(center_start),
                        "offset_from_center": int(start - center_start),
                    }
                )

    for sample_id in sorted(set(train_meta["sample_id"].astype(str)).difference(seen)):
        meta = train_meta.loc[sample_id]
        skipped_rows.append({"sample_id": sample_id, "split": "train", "frames": int(meta["frames"]), "reason": "not_found"})

    base_dir = child_dir(config.processed_dir / "windows", config.source_name, "source_name")
    base = base_arrays(base_dir)
    arrays = {
        "train": pack_split(x_train, pose_train, cell_train, presence_train, center_train),
        "val": {
            "X": base["X_val"],
            "pose": base["y_pose_val"],
            "cell": base["y_cell_val"],
            "presence": base["y_presence_val"],
            "center": base["y_center_val"],
        },
        "test": {
            "X": base["X_test"],
            "pose": base["y_pose_test"],
            "cell": base["y_cell_test"],
            "presence": base["y_presence_test"],
            "center": base["y_center_test"],
        },
    }

    base_windows = pd.read_csv(base_dir / "windows.csv")
    window_rows.extend(split_side_rows(base_windows, "val"))
    window_rows.extend(split_side_rows(base_windows, "test"))
    for index, row in enumerate(window_rows):
        row["window_id"] = index

    windows = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    skipped = pd.DataFrame(skipped_rows, columns=windowing.SKIPPED_COLUMNS)
    return arrays, windows, skipped, labels


def save_temporal_shift_dataset(
    config: TemporalShiftConfig,
    arrays: dict[str, dict[str, np.ndarray]],
    windows: pd.DataFrame,
    skipped: pd.DataFrame,
    labels: dict[str, dict[str, int]],
) -> AugmentResult:
    windows_root = config.processed_dir / "windows"
    output_dir = child_dir(windows_root, config.output_name, "output_name")
    source_dir = child_dir(windows_root, config.source_name, "source_name")
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("output_name must differ from source_name")
    ensure_output_dir(output_dir, config.overwrite)
    run_config: dict[str, object] = {
        **asdict(config),
        "augmentation_method": "temporal_shift_centered_stride",
        "train_only": True,
        "gaussian_noise": {
            "materialized": False,
            "recommended_sigmas": GAUSSIAN_NOISE_SIGMAS,
            "note": "Apply in the future Dataset/DataLoader only for train batches.",
        },
        "shape": "N,3,64,192",
        "val_test_policy": f"copied unchanged from processed/windows/{config.source_name}",
    }
    windowing.save_window_dataset(output_dir, arrays, windows, skipped, labels, run_config)
    return AugmentResult(
        source_name=config.source_name,
        output_name=config.output_name,
        input_dir=config.input_dir,
        output_dir=output_dir,
        train_windows=int(arrays["train"]["X"].shape[0]),
        val_windows=int(arrays["val"]["X"].shape[0]),
        test_windows=int(arrays["test"]["X"].shape[0]),
        skipped_train_samples=int(len(skipped)),
    )


def run_temporal_shift(config: TemporalShiftConfig) -> AugmentResult:
    arrays, windows, skipped, labels = make_temporal_shift_dataset(config)
    return save_temporal_shift_dataset(config, arrays, windows, skipped, labels)


def default_input_dir(source_name: str) -> Path:
    source_name = validate_profile_name(source_name, "source_name")
    if source_name == "raw":
        return DEFAULT_RAW_DIR
    return DEFAULT_DENOISED_ROOT / source_name


def run_default_profiles(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    split_path: Path = DEFAULT_SPLIT_PATH,
    stride: int = DEFAULT_STRIDE,
    max_crops_per_sample: int = DEFAULT_MAX_CROPS_PER_SAMPLE,
    overwrite: bool = False,
) -> list[AugmentResult]:
    pairs = (("raw", "raw_shift"), ("phase_hampel", "phase_hampel_shift"))
    results = []
    for source_name, output_name in pairs:
        config = TemporalShiftConfig(
            source_name=source_name,
            output_name=output_name,
            input_dir=default_input_dir(source_name),
            processed_dir=processed_dir,
            split_path=split_path,
            stride=stride,
            max_crops_per_sample=max_crops_per_sample,
            overwrite=overwrite,
        )
        results.append(run_temporal_shift(config))
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only temporal-shift CSI window artifacts.")
    parser.add_argument("--source-name", default="raw")
    parser.add_argument("--output-name")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-crops-per-sample", type=int, default=DEFAULT_MAX_CROPS_PER_SAMPLE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-defaults", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_defaults:
        results = run_default_profiles(
            processed_dir=args.processed_dir,
            split_path=args.split_path,
            stride=args.stride,
            max_crops_per_sample=args.max_crops_per_sample,
            overwrite=args.overwrite,
        )
    else:
        output_name = args.output_name or f"{args.source_name}_shift"
        input_dir = args.input_dir or default_input_dir(args.source_name)
        results = [
            run_temporal_shift(
                TemporalShiftConfig(
                    source_name=args.source_name,
                    output_name=output_name,
                    input_dir=input_dir,
                    processed_dir=args.processed_dir,
                    split_path=args.split_path,
                    stride=args.stride,
                    max_crops_per_sample=args.max_crops_per_sample,
                    overwrite=args.overwrite,
                )
            )
        ]
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
