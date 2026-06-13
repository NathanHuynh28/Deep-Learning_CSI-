
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eda_common import (
    RX_ORDER,
    default_figures_dir,
    default_session_dir,
    ensure_dir,
    load_quality,
    parse_rx_list,
    quality_path,
    raw_files,
    set_plot_style,
)


DOCUMENTED_PASS_FRAMES_PER_SAMPLE = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RSSI and packet-stability EDA tables.")
    parser.add_argument("--session-dir", type=Path, default=default_session_dir())
    parser.add_argument("--figures-dir", type=Path, default=default_figures_dir())
    parser.add_argument("--rx", default=",".join(RX_ORDER))
    return parser.parse_args()


def load_signal_frame(session_dir: Path, rx_list: list[str]) -> pd.DataFrame:
    usecols = ["sample_id", "host_ts_ms"]
    for rx in rx_list:
        usecols.extend([f"{rx}_rssi", f"{rx}_csi_len"])
    frames = [pd.read_csv(path, usecols=usecols) for path in raw_files(session_dir)]
    if not frames:
        raise ValueError(f"No data_P*.csv files found in {session_dir}")
    return pd.concat(frames, ignore_index=True)


def rssi_summary(frame: pd.DataFrame, rx_list: list[str]) -> pd.DataFrame:
    rows = []
    for rx in rx_list:
        values = frame[f"{rx}_rssi"].dropna()
        rows.append(
            {
                "rx": rx,
                "count": int(values.count()),
                "min": float(values.min()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def packet_stability(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("sample_id")["host_ts_ms"].agg(start_ms="min", end_ms="max", rows="size").reset_index()
    duration_s = (grouped["end_ms"] - grouped["start_ms"]) / 1000.0
    grouped["duration_s"] = duration_s
    grouped["packet_rate_rows_per_s"] = np.where(duration_s > 0, grouped["rows"] / duration_s, np.nan)
    return grouped


def csi_len_summary(frame: pd.DataFrame, rx_list: list[str]) -> pd.DataFrame:
    rows = []
    for rx in rx_list:
        column = f"{rx}_csi_len"
        rows.append(
            {
                "rx": rx,
                "rows": int(frame[column].count()),
                "bad_csi_len_rows": int((frame[column] != 128).sum()),
                "min_csi_len": int(frame[column].min()),
                "max_csi_len": int(frame[column].max()),
            }
        )
    return pd.DataFrame(rows)


def quality_failure_reasons(quality: pd.DataFrame) -> pd.DataFrame:
    failed = quality[quality["status"].fillna("").astype(str).str.lower() == "fail"].copy()
    if failed.empty:
        return pd.DataFrame(columns=["reason", "count", "percent_of_failures"])
    reasons = failed["reason"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    counts = reasons.value_counts().rename_axis("reason").reset_index(name="count")
    counts["percent_of_failures"] = counts["count"] / counts["count"].sum() * 100.0
    return counts


def save_rssi_boxplot(frame: pd.DataFrame, rx_list: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    data = [frame[f"{rx}_rssi"].dropna().to_numpy() for rx in rx_list]
    fig, axis = plt.subplots(figsize=(8, 5))
    box = axis.boxplot(data, tick_labels=rx_list, patch_artist=True, showfliers=False)
    colors = ["#3B82F6", "#F97316", "#10B981"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_title("RSSI distribution by RX")
    axis.set_ylabel("RSSI (dBm)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_packet_plots(packet: pd.DataFrame, quality: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(packet["packet_rate_rows_per_s"].dropna(), bins=32, color="#2563EB", alpha=0.85)
    axis.axvline(packet["packet_rate_rows_per_s"].median(), color="#DC2626", linestyle="--", label="Median")
    axis.set_title("Packet-rate distribution by sample")
    axis.set_xlabel("Rows per second")
    axis.set_ylabel("Samples")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "packet_rate_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(pd.to_numeric(quality["clean_frames"], errors="coerce").dropna(), bins=36, color="#059669", alpha=0.85)
    axis.axvline(
        DOCUMENTED_PASS_FRAMES_PER_SAMPLE,
        color="#DC2626",
        linestyle="--",
        label=f"{DOCUMENTED_PASS_FRAMES_PER_SAMPLE}-frame target",
    )
    axis.set_title("Clean frame distribution")
    axis.set_xlabel("Clean frames")
    axis.set_ylabel("Samples")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "clean_frames_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(pd.to_numeric(quality["missing_seq_max_gap"], errors="coerce").dropna(), bins=24, color="#7C3AED", alpha=0.85)
    axis.set_title("Missing sequence max-gap distribution")
    axis.set_xlabel("Max missing seq gap")
    axis.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_seq_gap_distribution.png", bbox_inches="tight")
    plt.close(fig)


def save_failure_reason_plot(failure_reasons: pd.DataFrame, output_path: Path) -> None:
    if failure_reasons.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    ordered = failure_reasons.sort_values("count", ascending=True)
    fig, axis = plt.subplots(figsize=(8.5, max(4.2, 0.45 * len(ordered))))
    bars = axis.barh(ordered["reason"], ordered["count"], color="#EF4444", alpha=0.85)
    axis.set_title("Quality failure reasons")
    axis.set_xlabel("Failed samples")
    axis.set_ylabel("Reason")
    axis.bar_label(bars, padding=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def quality_numeric_summary(quality: pd.DataFrame) -> pd.DataFrame:
    quality_numeric = quality.copy()
    for column in ("clean_frames", "malformed_lines", "duplicate_lines", "missing_seq_max_gap", "rssi_min", "rssi_mean", "rssi_max"):
        quality_numeric[column] = pd.to_numeric(quality_numeric[column], errors="coerce")
    return quality_numeric.describe(include="all").reset_index().rename(columns={"index": "stat"})


def build_signal_quality_tables(
    session_dir: Path = default_session_dir(),
    figures_dir: Path = default_figures_dir(),
    rx: str | list[str] = ",".join(RX_ORDER),
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    rx_list = parse_rx_list(rx) if isinstance(rx, str) else rx
    if save_figures:
        ensure_dir(figures_dir)
    frame = load_signal_frame(session_dir, rx_list)
    quality = load_quality(quality_path(session_dir))
    packet = packet_stability(frame)
    failure_reasons = quality_failure_reasons(quality)
    quality_numeric = quality.copy()
    for column in ("clean_frames", "malformed_lines", "duplicate_lines", "missing_seq_max_gap", "rssi_min", "rssi_mean", "rssi_max"):
        quality_numeric[column] = pd.to_numeric(quality_numeric[column], errors="coerce")

    if save_figures:
        save_rssi_boxplot(frame, rx_list, figures_dir / "rssi_by_rx_boxplot.png")
        save_packet_plots(packet, quality_numeric, figures_dir)
        save_failure_reason_plot(failure_reasons, figures_dir / "quality_failure_reasons.png")

    return {
        "rssi_by_rx": rssi_summary(frame, rx_list),
        "csi_len_summary": csi_len_summary(frame, rx_list),
        "packet_stability": packet,
        "quality_numeric_summary": quality_numeric.describe(include="all").reset_index().rename(columns={"index": "stat"}),
        "quality_failure_reasons": failure_reasons,
    }


def main() -> int:
    args = parse_args()
    figures_dir = ensure_dir(args.figures_dir)
    tables = build_signal_quality_tables(args.session_dir, figures_dir, args.rx, save_figures=True)
    print(f"Built {len(tables)} signal quality tables in memory")
    print(f"Saved signal quality figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
