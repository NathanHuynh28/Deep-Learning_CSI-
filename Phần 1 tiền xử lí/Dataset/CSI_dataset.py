from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
PART_DIR = THIS_DIR.parent
DEFAULT_WINDOWS_DIR = PART_DIR / "processed" / "windows"
NORMALIZE_PATH = PART_DIR / "Normalization" / "CSI_normalize.py"
NORMALIZE_SPEC = importlib.util.spec_from_file_location("csi_normalize", NORMALIZE_PATH)
if NORMALIZE_SPEC is None or NORMALIZE_SPEC.loader is None:
    raise ImportError(f"Cannot load CSI_normalize.py from {NORMALIZE_PATH}")
normalize = importlib.util.module_from_spec(NORMALIZE_SPEC)
sys.modules[NORMALIZE_SPEC.name] = normalize
NORMALIZE_SPEC.loader.exec_module(normalize)

SPLITS = ("train", "val", "test")
TARGETS = ("pose", "cell", "presence", "center")
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BatchItem = tuple[torch.Tensor, torch.Tensor]


class NormalizeModule(Protocol):
    def load_stats(self, profile_dir: Path) -> dict[str, FloatArray]: ...

    def apply_normalization(self, x: np.ndarray, stats: dict[str, FloatArray]) -> FloatArray: ...


normalize_module = cast(NormalizeModule, cast(object, normalize))


def validate_profile_name(profile: str) -> str:
    path = Path(profile)
    if not profile or profile in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"profile must be a simple profile name, got: {profile!r}")
    return profile


class CSIDataset(Dataset[BatchItem]):
    def __init__(
        self,
        profile_dir: Path,
        split: str = "train",
        target: str = "pose",
        normalize_runtime: bool = True,
        noise_sigma: float = 0.0,
    ) -> None:
        if split not in SPLITS:
            raise ValueError("split must be train, val, or test")
        if target not in TARGETS:
            raise ValueError("target must be pose, cell, presence, or center")
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        self.profile_dir: Path = profile_dir
        self.split: str = split
        self.target: str = target
        self.noise_sigma: float = float(noise_sigma)
        data_path = profile_dir / "data.npz"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing profile data: {data_path}")
        with cast(NpzFile, np.load(data_path)) as data:
            self.x: FloatArray = cast(FloatArray, data[f"X_{split}"].astype(np.float32, copy=False))
            self.y: np.ndarray = np.asarray(data[f"y_{target}_{split}"])
        self.stats: dict[str, FloatArray] | None = normalize_module.load_stats(profile_dir) if normalize_runtime else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> BatchItem:
        x = cast(FloatArray, self.x[index].astype(np.float32, copy=True))
        if self.stats is not None:
            x = cast(FloatArray, normalize_module.apply_normalization(x[np.newaxis, ...], self.stats)[0])
        if self.split == "train" and self.noise_sigma > 0.0:
            noise = cast(FloatArray, np.random.normal(0.0, self.noise_sigma, size=x.shape).astype(np.float32))
            x = cast(FloatArray, (x + noise).astype(np.float32))
        x = cast(FloatArray, x.transpose(2, 0, 1).reshape(192, 192))
        y = self.y[index]
        if self.target == "center":
            label = torch.as_tensor(y, dtype=torch.float32)
        else:
            label = torch.as_tensor(y, dtype=torch.long)
        return torch.from_numpy(x), label


def make_dataset(
    windows_dir: Path = DEFAULT_WINDOWS_DIR,
    profile: str = "raw_shift",
    split: str = "train",
    target: str = "pose",
    normalize_runtime: bool = True,
    noise_sigma: float = 0.0,
) -> CSIDataset:
    return CSIDataset(windows_dir / validate_profile_name(profile), split=split, target=target, normalize_runtime=normalize_runtime, noise_sigma=noise_sigma)


def make_dataloader(
    windows_dir: Path = DEFAULT_WINDOWS_DIR,
    profile: str = "raw_shift",
    split: str = "train",
    target: str = "pose",
    batch_size: int = 32,
    num_workers: int = 0,
    normalize_runtime: bool = True,
    noise_sigma: float = 0.0,
    shuffle: bool | None = None,
) -> DataLoader[BatchItem]:
    dataset = make_dataset(windows_dir, profile, split, target, normalize_runtime, noise_sigma)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" if shuffle is None else shuffle),
        num_workers=num_workers,
    )


def make_dataloaders(
    windows_dir: Path = DEFAULT_WINDOWS_DIR,
    profile: str = "raw_shift",
    target: str = "pose",
    batch_size: int = 32,
    num_workers: int = 0,
    normalize_runtime: bool = True,
    noise_sigma: float = 0.0,
) -> dict[str, DataLoader[BatchItem]]:
    return {
        split: make_dataloader(
            windows_dir=windows_dir,
            profile=profile,
            split=split,
            target=target,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize_runtime=normalize_runtime,
            noise_sigma=noise_sigma,
        )
        for split in SPLITS
    }


def class_distribution(profile_dir: Path, split: str = "train", target: str = "pose") -> dict[int, int]:
    if split not in SPLITS:
        raise ValueError("split must be train, val, or test")
    if target not in {"pose", "cell", "presence"}:
        raise ValueError("class_distribution target must be pose, cell, or presence")
    with cast(NpzFile, np.load(profile_dir / "data.npz")) as data:
        labels = cast(IntArray, data[f"y_{target}_{split}"].astype(np.int64))
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def class_weights(profile_dir: Path, split: str = "train", target: str = "pose") -> torch.Tensor:
    distribution = class_distribution(profile_dir, split=split, target=target)
    n_classes = max(distribution) + 1
    total = sum(distribution.values())
    weights = np.zeros(n_classes, dtype=np.float32)
    for class_id, count in distribution.items():
        weights[class_id] = total / max(n_classes * count, 1)
    return torch.from_numpy(weights)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check CSI Dataset/DataLoader batches.")
    _ = parser.add_argument("--windows-dir", type=Path, default=DEFAULT_WINDOWS_DIR)
    _ = parser.add_argument("--profile", default="raw_shift")
    _ = parser.add_argument("--split", choices=SPLITS, default="train")
    _ = parser.add_argument("--target", choices=TARGETS, default="pose")
    _ = parser.add_argument("--batch-size", type=int, default=8)
    _ = parser.add_argument("--num-workers", type=int, default=0)
    _ = parser.add_argument("--noise-sigma", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    windows_dir = cast(Path, args.windows_dir)
    profile = cast(str, args.profile)
    split = cast(str, args.split)
    target = cast(str, args.target)
    batch_size = cast(int, args.batch_size)
    num_workers = cast(int, args.num_workers)
    noise_sigma = cast(float, args.noise_sigma)
    loader = make_dataloader(
        windows_dir=windows_dir,
        profile=profile,
        split=split,
        target=target,
        batch_size=batch_size,
        num_workers=num_workers,
        noise_sigma=noise_sigma,
    )
    x, y = next(iter(loader))
    print(f"batch x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
    if target != "center":
        print("class_distribution", class_distribution(windows_dir / profile, split, target))
        print("class_weights", class_weights(windows_dir / profile, split, target).tolist())


if __name__ == "__main__":
    main()
