
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eda_common import (
    RX_ORDER,
    SUBCARRIERS_PER_RX,
    csi_columns,
    default_figures_dir,
    default_session_dir,
    ensure_dir,
    load_manifest,
    manifest_path,
    parse_rx_list,
    resolve_session_file,
    safe_name,
    set_plot_style,
    values_to_amplitude,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one representative CSI amplitude figure per pose.")
    parser.add_argument("--session-dir", type=Path, default=default_session_dir())
    parser.add_argument("--figures-dir", type=Path, default=default_figures_dir() / "amplitude_by_pose")
    parser.add_argument("--rx", default=",".join(RX_ORDER))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def representative_samples(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in manifest.sort_values(["coarse_pose", "frames", "sample_id"]).groupby("coarse_pose"):
        frames = pd.to_numeric(group["frames"], errors="coerce")
        target = frames.median()
        chosen_index = (frames - target).abs().sort_values().index[0]
        rows.append(manifest.loc[chosen_index])
    return pd.DataFrame(rows).sort_values("coarse_pose").reset_index(drop=True)


def load_sample_amplitude(session_dir: Path, sample: pd.Series, rx_list: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    csv_path = resolve_session_file(session_dir, sample["data_file"])
    usecols = ["sample_id", "host_ts_ms"]
    for rx in rx_list:
        usecols.extend([f"{rx}_csi_len", *csi_columns(rx)])
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=50_000):
        selected = chunk[chunk["sample_id"] == sample["sample_id"]]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"Sample not found in {csv_path}: {sample['sample_id']}")
    frame = pd.concat(chunks, ignore_index=True).sort_values("host_ts_ms")
    time_s = (frame["host_ts_ms"].to_numpy(dtype=float) - float(frame["host_ts_ms"].iloc[0])) / 1000.0
    amplitude = {}
    for rx in rx_list:
        bad_rows = frame[f"{rx}_csi_len"] != 128
        if bad_rows.any():
            raise ValueError(f"{sample['sample_id']} has invalid {rx}_csi_len rows")
        values = frame[csi_columns(rx)].to_numpy(dtype=float)
        amplitude[rx] = values_to_amplitude(values)
    return time_s, amplitude


def hampel_filter(values: np.ndarray, window: int = 9, threshold: float = 3.0) -> tuple[np.ndarray, int]:
    series = pd.Series(values)
    median = series.rolling(window=window, center=True, min_periods=1).median()
    deviation = (series - median).abs()
    mad = deviation.rolling(window=window, center=True, min_periods=1).median()
    limit = threshold * 1.4826 * mad.replace(0.0, np.nan)
    outliers = deviation > limit.fillna(np.inf)
    filtered = series.mask(outliers, median).to_numpy(dtype=float)
    return filtered, int(outliers.sum())


def save_raw_vs_filtered(
    sample: pd.Series,
    time_s: np.ndarray,
    amplitude: dict[str, np.ndarray],
    rx_list: list[str],
    figures_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, axes = plt.subplots(len(rx_list), 1, figsize=(12, max(4, 2.8 * len(rx_list))), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    rows = []
    fig.suptitle(f"Raw vs Hampel-filtered CSI mean amplitude | {sample['sample_id']}", fontsize=14)
    for axis, rx in zip(axes_array, rx_list):
        raw_trace = amplitude[rx].mean(axis=1)
        filtered_trace, outlier_count = hampel_filter(raw_trace)
        axis.plot(time_s, raw_trace, color="#94A3B8", linewidth=1.0, label="Raw mean amplitude")
        axis.plot(time_s, filtered_trace, color="#DC2626", linewidth=1.5, label="Hampel filtered")
        axis.set_title(f"{rx}: {outlier_count} outlier points replaced")
        axis.set_xlabel("Seconds")
        axis.set_ylabel("Mean amplitude")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "coarse_pose": sample["coarse_pose"],
                "rx": rx,
                "points": int(len(raw_trace)),
                "hampel_outlier_points": outlier_count,
                "outlier_percent": outlier_count / max(len(raw_trace), 1) * 100.0,
            }
        )
    fig.savefig(figures_dir.parent / "raw_vs_filtered_csi.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_pose_sample(sample: pd.Series, time_s: np.ndarray, amplitude: dict[str, np.ndarray], rx_list: list[str], output_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, axes = plt.subplots(len(rx_list) + 1, 1, figsize=(12.5, 3.0 * len(rx_list) + 3.2), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    fig.suptitle(
        f"{sample['coarse_pose']} | {sample['sample_id']} | {sample['cell_id']} | {sample['orientation_label']}",
        fontsize=14,
    )
    for axis, rx in zip(axes_array[:-1], rx_list):
        image = axis.imshow(
            amplitude[rx].T,
            aspect="auto",
            origin="lower",
            extent=[float(time_s[0]), float(time_s[-1]), 0, SUBCARRIERS_PER_RX - 1],
            cmap="magma",
        )
        axis.set_title(f"{rx} amplitude heatmap")
        axis.set_ylabel("Subcarrier")
        fig.colorbar(image, ax=axis, label="Amplitude")

    trace_axis = axes_array[-1]
    for rx in rx_list:
        trace_axis.plot(time_s, amplitude[rx].mean(axis=1), linewidth=1.4, label=f"{rx} mean amplitude")
    trace_axis.set_title("Mean amplitude trace")
    trace_axis.set_xlabel("Seconds")
    trace_axis.set_ylabel("Mean amplitude")
    trace_axis.legend(loc="best")
    trace_axis.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_amplitude_by_pose_outputs(
    session_dir: Path = default_session_dir(),
    figures_dir: Path = default_figures_dir() / "amplitude_by_pose",
    rx: str | list[str] = ",".join(RX_ORDER),
    dpi: int = 180,
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    rx_list = parse_rx_list(rx) if isinstance(rx, str) else rx
    if save_figures:
        ensure_dir(figures_dir)
    manifest = load_manifest(manifest_path(session_dir))
    selected = representative_samples(manifest)
    summary_rows = []

    for _, sample in selected.iterrows():
        time_s, amplitude = load_sample_amplitude(session_dir, sample, rx_list)
        output_path = figures_dir / f"amplitude_by_pose_{safe_name(sample['coarse_pose'])}.png"
        if save_figures:
            plot_pose_sample(sample, time_s, amplitude, rx_list, output_path, dpi)
        summary_rows.append(
            {
                "coarse_pose": sample["coarse_pose"],
                "sample_id": sample["sample_id"],
                "person": sample["person"],
                "cell_id": sample["cell_id"],
                "orientation_label": sample["orientation_label"],
                "frames": int(sample["frames"]),
                "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) else 0.0,
                "figure": str(output_path),
            }
        )

    non_empty = selected[selected["coarse_pose"] != "Empty"]
    comparison_sample = non_empty.iloc[0] if not non_empty.empty else selected.iloc[0]
    time_s, amplitude = load_sample_amplitude(session_dir, comparison_sample, rx_list)
    raw_vs_filtered = save_raw_vs_filtered(comparison_sample, time_s, amplitude, rx_list, figures_dir, dpi) if save_figures else raw_vs_filtered_summary(comparison_sample, amplitude, rx_list)

    return {
        "amplitude_by_pose_summary": pd.DataFrame(summary_rows),
        "raw_vs_filtered_csi_summary": raw_vs_filtered,
    }


def raw_vs_filtered_summary(sample: pd.Series, amplitude: dict[str, np.ndarray], rx_list: list[str]) -> pd.DataFrame:
    rows = []
    for rx in rx_list:
        raw_trace = amplitude[rx].mean(axis=1)
        _, outlier_count = hampel_filter(raw_trace)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "coarse_pose": sample["coarse_pose"],
                "rx": rx,
                "points": int(len(raw_trace)),
                "hampel_outlier_points": outlier_count,
                "outlier_percent": outlier_count / max(len(raw_trace), 1) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    figures_dir = ensure_dir(args.figures_dir)
    tables = build_amplitude_by_pose_outputs(args.session_dir, figures_dir, args.rx, args.dpi, save_figures=True)
    print(f"Saved amplitude-by-pose figures to {figures_dir}")
    print(f"Built {len(tables)} amplitude EDA tables in memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
