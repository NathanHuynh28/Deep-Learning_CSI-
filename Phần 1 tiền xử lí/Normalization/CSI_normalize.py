from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray


THIS_DIR = Path(__file__).resolve().parent
PART_DIR = THIS_DIR.parent
DEFAULT_WINDOWS_DIR = PART_DIR / "processed" / "windows"
DEFAULT_PROFILES = ("raw", "raw_shift", "phase_hampel", "phase_hampel_shift")
STATS_FILENAME = "normalization_stats.npz"


def data_path(profile_dir: Path) -> Path:
    path = profile_dir / "data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing window artifact: {path}")
    return path


def stats_path(profile_dir: Path) -> Path:
    return profile_dir / STATS_FILENAME


FloatArray = NDArray[np.float32]


def validate_profile_name(profile: str) -> str:
    path = Path(profile)
    if not profile or profile in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"profile must be a simple profile name, got: {profile!r}")
    return profile


def fit_stats(x_train: np.ndarray, eps: float = 1e-6) -> dict[str, FloatArray]:
    if x_train.ndim != 4:
        raise ValueError(f"Expected X_train shape (N,3,64,192), got {x_train.shape}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    x = cast(FloatArray, x_train.astype(np.float32, copy=False))
    mean = cast(FloatArray, x.mean(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32))
    std = cast(FloatArray, x.std(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32))
    clipped_std = cast(FloatArray, np.maximum(std, np.float32(eps)))
    return {"mean": mean, "std": clipped_std, "eps": np.array(eps, dtype=np.float32)}


def save_stats(profile_dir: Path, stats: dict[str, FloatArray]) -> Path:
    path = stats_path(profile_dir)
    np.savez_compressed(path, mean=stats["mean"], std=stats["std"], eps=stats["eps"])
    return path


def load_stats(profile_dir: Path) -> dict[str, FloatArray]:
    path = stats_path(profile_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {path}")
    with cast(NpzFile, np.load(path)) as loaded:
        return {
            "mean": cast(FloatArray, loaded["mean"].astype(np.float32)),
            "std": cast(FloatArray, loaded["std"].astype(np.float32)),
            "eps": cast(FloatArray, loaded["eps"].astype(np.float32)),
        }


def apply_normalization(x: np.ndarray, stats: dict[str, FloatArray]) -> FloatArray:
    normalized = (x.astype(np.float32, copy=False) - stats["mean"]) / stats["std"]
    return cast(FloatArray, normalized.astype(np.float32))


def fit_profile(profile_dir: Path, eps: float = 1e-6) -> Path:
    with cast(NpzFile, np.load(data_path(profile_dir))) as data:
        stats = fit_stats(data["X_train"], eps=eps)
    return save_stats(profile_dir, stats)


def fit_profiles(windows_dir: Path = DEFAULT_WINDOWS_DIR, profiles: Iterable[str] = DEFAULT_PROFILES, eps: float = 1e-6) -> dict[str, Path]:
    return {profile: fit_profile(windows_dir / validate_profile_name(profile), eps=eps) for profile in profiles}


def normalized_split_stats(profile_dir: Path, split: str = "train") -> dict[str, float]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    with cast(NpzFile, np.load(data_path(profile_dir))) as data:
        x = apply_normalization(data[f"X_{split}"], load_stats(profile_dir))
    feature_std = x.std(axis=(0, 3), dtype=np.float64)
    active = feature_std > 1e-5
    active_std = float(feature_std[active].mean()) if bool(active.any()) else 0.0
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "active_feature_std": active_std,
        "constant_features": float((~active).sum()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit train-only z-score stats beside CSI window artifacts.")
    _ = parser.add_argument("--windows-dir", type=Path, default=DEFAULT_WINDOWS_DIR)
    _ = parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    _ = parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    windows_dir = cast(Path, args.windows_dir)
    profiles = cast(list[str], args.profiles)
    eps = cast(float, args.eps)
    for profile, path in fit_profiles(windows_dir, profiles, eps=eps).items():
        stats = load_stats(path.parent)
        print(f"{profile}: saved {path.name}; mean {stats['mean'].shape}; std {stats['std'].shape}")


if __name__ == "__main__":
    main()
