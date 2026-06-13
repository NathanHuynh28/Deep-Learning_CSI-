
from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RX_ORDER = ("RX1", "RX2", "RX3")
CSI_VALUES_PER_RX = 128
SUBCARRIERS_PER_RX = CSI_VALUES_PER_RX // 2


@dataclass(frozen=True)
class ManifestSample:
    index: int
    sample_id: str
    data_file: str
    person: str
    presence: str
    cell_id: str
    coarse_pose: str
    orientation_label: str
    trial: str
    frames: str
    status: str


@dataclass(frozen=True)
class SampleData:
    manifest: ManifestSample
    host_ts_ms: np.ndarray
    amplitude: dict[str, np.ndarray]
    rssi: dict[str, np.ndarray]


@dataclass(frozen=True)
class CliArgs:
    manifest_path: Path
    index: int | None
    output_dir: Path
    rx: str
    dpi: int
    show: bool
    list_samples: bool


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "Dataset_CSI_3D_v2" / "session1" / "manifest.csv"


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Visualize one collected CSI sample as one figure.")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path(), help="Path to manifest.csv.")
    parser.add_argument("--index", "-i", type=int, help="1-based sample index from manifest.csv to plot.")
    parser.add_argument(
        "--list",
        dest="list_samples",
        action="store_true",
        help="List sample indices from manifest.csv and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures") / "eda" / "samples",
        help="Directory for generated PNG files.",
    )
    parser.add_argument("--rx", default=",".join(RX_ORDER), help="RX list to plot, for example RX1,RX2,RX3.")
    parser.add_argument("--dpi", type=int, default=140, help="Output image DPI.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    args = parser.parse_args()

    if args.index is not None and args.index < 1:
        parser.error("--index must be >= 1")
    if not args.list_samples and args.index is None:
        parser.error("use --list to choose a sample, then render with --index N")

    return CliArgs(
        manifest_path=args.manifest,
        index=args.index,
        output_dir=args.output_dir,
        rx=args.rx,
        dpi=args.dpi,
        show=args.show,
        list_samples=args.list_samples,
    )


def parse_rx_list(value: str) -> list[str]:
    selected = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = [rx for rx in selected if rx not in RX_ORDER]
    if invalid:
        raise ValueError(f"Invalid RX names: {', '.join(invalid)}. Expected: {', '.join(RX_ORDER)}")
    if not selected:
        raise ValueError("At least one RX must be selected.")
    return selected


def load_manifest(manifest_path: Path) -> list[ManifestSample]:
    required_columns = [
        "sample_id",
        "data_file",
        "person",
        "presence",
        "cell_id",
        "coarse_pose",
        "orientation_label",
        "trial",
        "frames",
        "status",
    ]
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        for column in required_columns:
            if column not in reader.fieldnames:
                raise ValueError(f"Missing manifest column: {column}")

        samples: list[ManifestSample] = []
        for index, row in enumerate(reader, start=1):
            samples.append(
                ManifestSample(
                    index=index,
                    sample_id=row["sample_id"].strip(),
                    data_file=row["data_file"].strip(),
                    person=row["person"],
                    presence=row["presence"],
                    cell_id=row["cell_id"],
                    coarse_pose=row["coarse_pose"],
                    orientation_label=row["orientation_label"],
                    trial=row["trial"],
                    frames=row["frames"],
                    status=row["status"],
                )
            )

    if not samples:
        raise ValueError(f"No samples found in {manifest_path}")
    return samples


def print_samples(samples: Sequence[ManifestSample]) -> None:
    print("index,sample_id,person,cell_id,pose,orientation,trial,frames,status,data_file")
    for sample in samples:
        print(
            f"{sample.index},{sample.sample_id},{sample.person},{sample.cell_id},"
            + f"{sample.coarse_pose},{sample.orientation_label},{sample.trial},"
            + f"{sample.frames},{sample.status},{sample.data_file}"
        )


def select_sample(samples: Sequence[ManifestSample], index: int) -> ManifestSample:
    if index > len(samples):
        raise ValueError(f"--index {index} is out of range; manifest has {len(samples)} samples")
    sample = samples[index - 1]
    if not sample.sample_id:
        raise ValueError(f"Manifest row {index} has empty sample_id")
    if not sample.data_file:
        raise ValueError(f"Manifest row {index} has empty data_file")
    return sample


def data_path_for_sample(manifest_path: Path, sample: ManifestSample) -> Path:
    data_path = Path(sample.data_file)
    if data_path.is_absolute():
        raise ValueError(f"data_file must be relative to manifest directory: {sample.data_file}")
    base = manifest_path.parent.resolve()
    candidate = (base / data_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"data_file escapes manifest directory: {sample.data_file}") from exc
    if not candidate.is_file():
        raise ValueError(f"data_file does not exist: {sample.data_file}")
    return candidate


def validate_raw_header(header: Sequence[str], selected_rx: Sequence[str]) -> None:
    required_columns = ["sample_id", "host_ts_ms"]
    for rx in selected_rx:
        required_columns.extend([f"{rx}_rssi", f"{rx}_csi_len"])
        required_columns.extend(f"{rx}_csi_{index:03d}" for index in range(CSI_VALUES_PER_RX))
    for column in required_columns:
        if column not in header:
            raise ValueError(f"Missing raw CSV column: {column}")


def row_to_amplitude(row: dict[str, str], rx: str) -> np.ndarray:
    csi_len = int(row[f"{rx}_csi_len"])
    if csi_len != CSI_VALUES_PER_RX:
        raise ValueError(f"{row['sample_id']} has {rx}_csi_len={csi_len}; expected {CSI_VALUES_PER_RX}")
    values = np.array([int(row[f"{rx}_csi_{index:03d}"]) for index in range(CSI_VALUES_PER_RX)], dtype=float)
    iq_pairs = values.reshape(SUBCARRIERS_PER_RX, 2)
    return np.hypot(iq_pairs[:, 0], iq_pairs[:, 1])


def load_sample_data(csv_path: Path, sample: ManifestSample, selected_rx: Sequence[str]) -> SampleData:
    host_ts_ms: list[int] = []
    amplitude: dict[str, list[np.ndarray]] = {rx: [] for rx in selected_rx}
    rssi: dict[str, list[int]] = {rx: [] for rx in selected_rx}

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Raw CSV is empty: {csv_path}")
        validate_raw_header(reader.fieldnames, selected_rx)

        for row in reader:
            if row["sample_id"] != sample.sample_id:
                continue
            host_ts_ms.append(int(row["host_ts_ms"]))
            for rx in selected_rx:
                rssi[rx].append(int(row[f"{rx}_rssi"]))
                amplitude[rx].append(row_to_amplitude(row, rx))

    if not host_ts_ms:
        raise ValueError(f"Sample not found in {csv_path}: {sample.sample_id}")

    return SampleData(
        manifest=sample,
        host_ts_ms=np.asarray(host_ts_ms, dtype=float),
        amplitude={rx: np.vstack(values) for rx, values in amplitude.items()},
        rssi={rx: np.asarray(values, dtype=float) for rx, values in rssi.items()},
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_sample(sample_data: SampleData, selected_rx: Sequence[str], output_dir: Path, dpi: int, show: bool) -> Path:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = sample_data.manifest
    time_s = (sample_data.host_ts_ms - sample_data.host_ts_ms[0]) / 1000.0
    fig, axes = plt.subplots(
        len(selected_rx) + 1,
        1,
        figsize=(12, 3.1 * len(selected_rx) + 3.0),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    title = (
        f"#{sample.index} | {sample.sample_id} | {sample.cell_id} | "
        f"{sample.coarse_pose} | {sample.orientation_label} | rows {len(time_s)}"
    )
    fig.suptitle(title)

    for axis, rx in zip(axes_array[:-1], selected_rx):
        image = axis.imshow(
            sample_data.amplitude[rx].T,
            aspect="auto",
            origin="lower",
            extent=[float(time_s[0]), float(time_s[-1]), 0, SUBCARRIERS_PER_RX - 1],
            cmap="viridis",
        )
        axis.set_title(f"{rx} CSI amplitude heatmap")
        axis.set_ylabel("Subcarrier")
        fig.colorbar(image, ax=axis, label="Amplitude")

    trace_axis = axes_array[-1]
    for rx in selected_rx:
        mean_amplitude = sample_data.amplitude[rx].mean(axis=1)
        trace_axis.plot(time_s, mean_amplitude, label=f"{rx} mean amplitude")
    trace_axis.set_title("Mean amplitude trace")
    trace_axis.set_xlabel("Seconds in sample")
    trace_axis.set_ylabel("Mean amplitude")
    trace_axis.grid(True, alpha=0.25)
    trace_axis.legend(loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sample_{sample.index:03d}_{safe_name(sample.sample_id)}.png"
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    try:
        selected_rx = parse_rx_list(args.rx)
        samples = load_manifest(args.manifest_path)

        if args.list_samples:
            print_samples(samples)
            return 0

        if args.index is None:
            raise ValueError("Missing --index N")
        manifest_sample = select_sample(samples, args.index)
        csv_path = data_path_for_sample(args.manifest_path, manifest_sample)
        sample_data = load_sample_data(csv_path, manifest_sample, selected_rx)
        output_path = plot_sample(sample_data, selected_rx, args.output_dir, args.dpi, args.show)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    print(f"Sample index: {manifest_sample.index}")
    print(f"Sample ID: {manifest_sample.sample_id}")
    print(f"Pose: {manifest_sample.coarse_pose}")
    print(f"Orientation: {manifest_sample.orientation_label}")
    print(f"Rows: {len(sample_data.host_ts_ms)}")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
