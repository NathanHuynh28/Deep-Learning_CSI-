from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


T_FRAMES = 192
SPLIT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "val", "test")
CANDIDATES = 300
RX_ORDER = ("RX1", "RX2", "RX3")
CSI_VALUES_PER_RX = 128
SUBCARRIERS_PER_RX = CSI_VALUES_PER_RX // 2

PART_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PART_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Dataset_CSI_3D_v2" / "session1"
DEFAULT_PROCESSED_DIR = PART_DIR / "processed"
DEFAULT_SPLIT_PATH = DEFAULT_PROCESSED_DIR / "sample_split" / "split.csv"

MANIFEST_COLUMNS = (
    "sample_id",
    "data_file",
    "person",
    "presence",
    "cell_id",
    "center_x_m",
    "center_y_m",
    "coarse_pose",
    "frames",
    "status",
)
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
)
SKIPPED_COLUMNS = ("sample_id", "split", "frames", "reason")


@dataclass
class SplitBuffer:
    X: list[np.ndarray] = field(default_factory=list)
    pose: list[int] = field(default_factory=list)
    cell: list[int] = field(default_factory=list)
    presence: list[int] = field(default_factory=list)
    center: list[np.ndarray] = field(default_factory=list)


def csi_columns(rx: str) -> list[str]:
    return [f"{rx}_csi_{index:03d}" for index in range(CSI_VALUES_PER_RX)]


def load_manifest(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "manifest.csv"
    frame = pd.read_csv(path)
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing manifest columns in {path}: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("manifest.csv must contain one row per sample_id")
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].fillna("").astype(str).str.lower() == "pass"].copy()
    return frame.sort_values("sample_id").reset_index(drop=True)


def split_sizes(n_items: int, ratios: Sequence[float] = SPLIT_RATIOS) -> list[int]:
    train = int(round(n_items * ratios[0]))
    val = int(round(n_items * ratios[1]))
    test = n_items - train - val
    return [train, val, test]


def split_candidate(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for _, group in samples.groupby("coarse_pose", dropna=False, sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).copy()
        labels: list[str] = []
        for split_name, size in zip(SPLIT_NAMES, split_sizes(len(shuffled)), strict=True):
            labels.extend([split_name] * size)
        shuffled["split"] = labels
        parts.append(shuffled)
    candidate = pd.concat(parts, ignore_index=True).sort_values(by=["sample_id"]).reset_index(drop=True)
    return rebalance_sizes(candidate, rng)


def rebalance_sizes(split: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    target = dict(zip(SPLIT_NAMES, split_sizes(len(split)), strict=True))
    balanced = split.copy()
    while True:
        counts = balanced["split"].value_counts().to_dict()
        over = [name for name in SPLIT_NAMES if int(counts.get(name, 0)) > target[name]]
        under = [name for name in SPLIT_NAMES if int(counts.get(name, 0)) < target[name]]
        if not over or not under:
            return balanced
        source = over[0]
        dest = under[0]
        choices = balanced.index[balanced["split"].eq(source)].to_numpy()
        chosen = int(rng.choice(choices))
        balanced.loc[chosen, "split"] = dest


def distribution_score(split: pd.DataFrame, columns: str | list[str]) -> float:
    if isinstance(columns, str):
        labels = split[columns].fillna("Unknown").astype(str)
    else:
        labels = split[columns].fillna("Unknown").astype(str).agg("|".join, axis=1)
    counts = pd.DataFrame({"split": split["split"].astype(str), "label": labels})
    total = counts["label"].value_counts().to_dict()
    actual = counts.groupby(["split", "label"], dropna=False).size().to_dict()
    score = 0.0
    for split_name, ratio in zip(SPLIT_NAMES, SPLIT_RATIOS, strict=True):
        for label, count in total.items():
            expected = float(count) * ratio
            score += abs(float(actual.get((split_name, label), 0)) - expected) / max(float(count), 1.0)
    return score


def candidate_score(split: pd.DataFrame) -> float:
    target = dict(zip(SPLIT_NAMES, split_sizes(len(split)), strict=True))
    actual = split["split"].value_counts().to_dict()
    size_score = sum(abs(int(actual.get(name, 0)) - target[name]) for name in SPLIT_NAMES)
    return (
        4.0 * size_score
        + distribution_score(split, "coarse_pose")
        + distribution_score(split, "cell_id")
        + distribution_score(split, "presence")
        + distribution_score(split, ["cell_id", "coarse_pose"])
    )


def make_split(manifest: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    columns = ["sample_id", "data_file", "person", "presence", "cell_id", "coarse_pose", "frames"]
    samples = pd.DataFrame(manifest.loc[:, columns]).copy()
    rng = np.random.default_rng(seed)
    best = split_candidate(samples, rng)
    best_score = candidate_score(best)
    for _ in range(CANDIDATES - 1):
        candidate = split_candidate(samples, rng)
        score = candidate_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score
    return pd.DataFrame(best.loc[:, ["sample_id", "split", "data_file", "person", "presence", "cell_id", "coarse_pose", "frames"]])


def score_split(split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, group in split.groupby("split", sort=True):
        rows.append({"metric": "samples", "split": split_name, "label": "all", "count": int(len(group))})
        for pose, part in group.groupby("coarse_pose", dropna=False):
            rows.append({"metric": "pose", "split": split_name, "label": pose, "count": int(len(part))})
        for cell, part in group.groupby("cell_id", dropna=False):
            rows.append({"metric": "cell", "split": split_name, "label": cell, "count": int(len(part))})
        for presence, part in group.groupby("presence", dropna=False):
            rows.append({"metric": "presence", "split": split_name, "label": presence, "count": int(len(part))})
    return pd.DataFrame(rows)


def save_split(split: pd.DataFrame, split_path: Path) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False, encoding="utf-8")
    score_split(split).to_csv(split_path.parent / "report.csv", index=False, encoding="utf-8")


def load_split(split_path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(split_path)
    required = {"sample_id", "split"}
    missing = required.difference(split.columns)
    if missing:
        raise ValueError(f"Missing split columns in {split_path}: {sorted(missing)}")
    validate_split(split)
    manifest_ids = set(manifest["sample_id"].astype(str))
    split_ids = set(split["sample_id"].astype(str))
    missing_ids = sorted(manifest_ids.difference(split_ids))
    if missing_ids:
        raise ValueError(f"Split is missing {len(missing_ids)} manifest samples. Use --rebuild-split.")
    split = split.loc[split["sample_id"].astype(str).isin(list(manifest_ids)), ["sample_id", "split"]].copy()
    meta_columns = ["sample_id", "data_file", "person", "presence", "cell_id", "coarse_pose", "frames"]
    return split.merge(manifest[meta_columns], on="sample_id", how="left")


def validate_split(split: pd.DataFrame) -> None:
    allowed = {"train", "val", "test"}
    invalid = sorted(set(split["split"].astype(str)).difference(allowed))
    if invalid:
        raise ValueError(f"Invalid split names: {invalid}")
    overlap = split.groupby("sample_id")["split"].nunique()
    bad = overlap.loc[overlap > 1]
    if not bad.empty:
        raise ValueError(f"Samples appear in multiple splits: {list(bad.index[:5])}")
    if split["sample_id"].duplicated().any():
        raise ValueError("split.csv must contain one row per sample_id")


def center_window_start(n_frames: int) -> int | None:
    if n_frames < T_FRAMES:
        return None
    return (n_frames - T_FRAMES) // 2


def resolve_data_file(input_dir: Path, data_file: object) -> Path:
    base = input_dir.resolve()
    relative = Path(str(data_file))
    if relative.is_absolute():
        raise ValueError(f"data_file must be relative to input directory: {data_file}")
    path = (base / relative).resolve()
    try:
        _ = path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"data_file escapes input directory: {data_file}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    return path


def sample_to_tensor(frame: pd.DataFrame) -> np.ndarray:
    tensors = []
    for rx in RX_ORDER:
        length_column = f"{rx}_csi_len"
        if length_column in frame.columns and not (frame[length_column] == CSI_VALUES_PER_RX).all():
            raise ValueError(f"Invalid {length_column}; expected {CSI_VALUES_PER_RX}")
        values = frame[csi_columns(rx)].to_numpy(dtype=np.float32)
        pairs = values.reshape(len(frame), SUBCARRIERS_PER_RX, 2)
        tensors.append(np.hypot(pairs[..., 0], pairs[..., 1]).T)
    return np.stack(tensors, axis=0).astype(np.float32)


def label_maps(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    poses = sorted(manifest["coarse_pose"].fillna("Unknown").astype(str).unique())
    cells = sorted(manifest["cell_id"].fillna("Unknown").astype(str).unique())
    return {
        "pose": {label: index for index, label in enumerate(poses)},
        "cell": {label: index for index, label in enumerate(cells)},
    }


def empty_arrays() -> dict[str, np.ndarray]:
    return {
        "X": np.empty((0, len(RX_ORDER), SUBCARRIERS_PER_RX, T_FRAMES), dtype=np.float32),
        "pose": np.empty((0,), dtype=np.int64),
        "cell": np.empty((0,), dtype=np.int64),
        "presence": np.empty((0,), dtype=np.int64),
        "center": np.empty((0, 2), dtype=np.float32),
    }


def make_window_dataset(input_dir: Path, manifest: pd.DataFrame, split: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    validate_split(split)
    labels = label_maps(manifest)
    sample_meta = manifest.merge(split[["sample_id", "split"]], on="sample_id", how="inner")
    sample_meta = sample_meta.set_index("sample_id", drop=False)
    wanted_by_file = sample_meta.groupby("data_file")["sample_id"].apply(lambda values: set(values.astype(str)))
    arrays = {name: SplitBuffer() for name in ("train", "val", "test")}
    window_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    usecols = ["sample_id", "frame_idx"] + [column for rx in RX_ORDER for column in csi_columns(rx)]
    usecols += [f"{rx}_csi_len" for rx in RX_ORDER]

    for data_file, wanted_ids in wanted_by_file.items():
        csv_path = resolve_data_file(input_dir, data_file)
        data = pd.read_csv(csv_path, usecols=lambda column: column in usecols)
        if "frame_idx" in data.columns:
            data = data.sort_values(["sample_id", "frame_idx"])
        for sample_id, frame in data.groupby("sample_id", sort=False):
            sample_id = str(sample_id)
            if sample_id not in wanted_ids:
                continue
            seen.add(sample_id)
            meta = sample_meta.loc[sample_id]
            split_name = str(meta["split"])
            start = center_window_start(len(frame))
            if start is None:
                skipped_rows.append({"sample_id": sample_id, "split": split_name, "frames": int(len(frame)), "reason": "too_short"})
                continue
            tensor = sample_to_tensor(frame)
            pose = labels["pose"][str(meta["coarse_pose"])]
            cell = labels["cell"][str(meta["cell_id"])]
            presence = int(meta["presence"])
            center = np.array([float(meta["center_x_m"]), float(meta["center_y_m"])], dtype=np.float32)
            arrays[split_name].X.append(tensor[:, :, start : start + T_FRAMES])
            arrays[split_name].pose.append(pose)
            arrays[split_name].cell.append(cell)
            arrays[split_name].presence.append(presence)
            arrays[split_name].center.append(center)
            window_rows.append(
                {
                    "window_id": len(window_rows),
                    "split": split_name,
                    "sample_id": sample_id,
                    "data_file": data_file,
                    "start": int(start),
                    "end": int(start + T_FRAMES),
                    "frames": T_FRAMES,
                    "presence": presence,
                    "cell_id": meta["cell_id"],
                    "coarse_pose": meta["coarse_pose"],
                    "center_x_m": float(meta["center_x_m"]),
                    "center_y_m": float(meta["center_y_m"]),
                }
            )

    for sample_id in sorted(set(sample_meta["sample_id"].astype(str)).difference(seen)):
        meta = sample_meta.loc[sample_id]
        skipped_rows.append({"sample_id": sample_id, "split": meta["split"], "frames": int(meta["frames"]), "reason": "not_found"})

    packed: dict[str, dict[str, np.ndarray]] = {}
    for split_name, values in arrays.items():
        packed[split_name] = pack_arrays(values)
    return packed, pd.DataFrame(window_rows, columns=WINDOW_COLUMNS), pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS), labels


def pack_arrays(values: SplitBuffer) -> dict[str, np.ndarray]:
    if not values.X:
        return empty_arrays()
    return {
        "X": np.stack(values.X).astype(np.float32),
        "pose": np.asarray(values.pose, dtype=np.int64),
        "cell": np.asarray(values.cell, dtype=np.int64),
        "presence": np.asarray(values.presence, dtype=np.int64),
        "center": np.stack(values.center).astype(np.float32),
    }


def save_window_dataset(
    output_dir: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    windows: pd.DataFrame,
    skipped: pd.DataFrame,
    labels: dict[str, dict[str, int]],
    config: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "data.npz",
        X_train=arrays["train"]["X"],
        X_val=arrays["val"]["X"],
        X_test=arrays["test"]["X"],
        y_pose_train=arrays["train"]["pose"],
        y_pose_val=arrays["val"]["pose"],
        y_pose_test=arrays["test"]["pose"],
        y_cell_train=arrays["train"]["cell"],
        y_cell_val=arrays["val"]["cell"],
        y_cell_test=arrays["test"]["cell"],
        y_presence_train=arrays["train"]["presence"],
        y_presence_val=arrays["val"]["presence"],
        y_presence_test=arrays["test"]["presence"],
        y_center_train=arrays["train"]["center"],
        y_center_val=arrays["val"]["center"],
        y_center_test=arrays["test"]["center"],
    )
    windows.to_csv(output_dir / "windows.csv", index=False, encoding="utf-8")
    skipped.to_csv(output_dir / "skipped.csv", index=False, encoding="utf-8")
    _ = (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    _ = (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split CSI samples and build one centered 192-frame window per sample.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--name", default="raw")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild-split", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    name: str = args.name
    processed_dir: Path = args.processed_dir
    split_path: Path = args.split_path
    seed: int = args.seed
    rebuild_split: bool = args.rebuild_split

    manifest = load_manifest(input_dir)
    if rebuild_split or not split_path.exists():
        split = make_split(manifest, seed=seed)
        save_split(split, split_path)
    else:
        split = load_split(split_path, manifest)

    arrays, windows, skipped, labels = make_window_dataset(input_dir, manifest, split)
    output_dir = processed_dir / "windows" / name
    config: dict[str, object] = {
        "input_dir": input_dir,
        "name": name,
        "split_path": split_path,
        "seed": seed,
        "t_frames": T_FRAMES,
        "window_policy": "one centered 192-frame window per sample for train, val, and test",
        "split_ratios": SPLIT_RATIOS,
        "rx_order": RX_ORDER,
        "shape": "N,3,64,192",
    }
    save_window_dataset(output_dir, arrays, windows, skipped, labels, config)
    counts = {split_name: int(values["X"].shape[0]) for split_name, values in arrays.items()}
    print(f"Saved windows: {output_dir}")
    print(f"Windows: {counts}; skipped samples: {len(skipped)}")


if __name__ == "__main__":
    main()
