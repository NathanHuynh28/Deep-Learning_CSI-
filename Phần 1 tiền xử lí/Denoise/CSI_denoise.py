from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sympy import series

try:
    pywt = importlib.import_module("pywt")
except ImportError:
    pywt = None


THIS_DIR = Path(__file__).resolve().parent
PART_DIR = THIS_DIR.parent
PROJECT_ROOT = PART_DIR.parent

RAW_SESSION_DIR = PROJECT_ROOT / "Dataset_CSI_3D_v2" / "session1"
DENOISED_ROOT = PART_DIR / "processed" / "denoised"
REPORT_ROOT = PART_DIR / "reports" / "denoise"

RX_ORDER = ("RX1", "RX2", "RX3")
CSI_VALUES_PER_RX = 128
SUBCARRIERS_PER_RX = CSI_VALUES_PER_RX // 2
CHUNKSIZE = 50_000

##### CHỈNH THAM SỐ Ở ĐÂY
HAMPEL_WINDOW = 7
HAMPEL_N_SIGMA = 6.0
WAVELET = "db4"
WAVELET_LEVEL = 3
DWT_THRESHOLD_SCALE = 0.75

BASE_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class DenoiseConfig:
    profile: str
    use_phase: bool
    use_hampel: bool
    use_dwt: bool
    rx_order: tuple[str, ...]
    hampel_window: int
    hampel_n_sigma: float
    wavelet: str
    wavelet_level: int
    dwt_threshold_scale: float


@dataclass(frozen=True)
class RunResult:
    profile: str
    input_dir: Path
    output_dir: Path
    report_dir: Path
    processed_samples: int
    processed_frames: int


def profile_config(profile: str) -> DenoiseConfig:
    if profile == "phase_hampel":
        return DenoiseConfig(
            profile=profile,
            use_phase=True,
            use_hampel=True,
            use_dwt=False,
            rx_order=RX_ORDER,
            hampel_window=HAMPEL_WINDOW,
            hampel_n_sigma=HAMPEL_N_SIGMA,
            wavelet=WAVELET,
            wavelet_level=WAVELET_LEVEL,
            dwt_threshold_scale=DWT_THRESHOLD_SCALE,
        )
    if profile == "phase_hampel_dwt":
        return DenoiseConfig(
            profile=profile,
            use_phase=True,
            use_hampel=True,
            use_dwt=True,
            rx_order=RX_ORDER,
            hampel_window=HAMPEL_WINDOW,
            hampel_n_sigma=HAMPEL_N_SIGMA,
            wavelet=WAVELET,
            wavelet_level=WAVELET_LEVEL,
            dwt_threshold_scale=DWT_THRESHOLD_SCALE,
        )
    raise ValueError("profile must be 'phase_hampel' or 'phase_hampel_dwt'")


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


def required_columns(rx_order: Sequence[str] = RX_ORDER) -> list[str]:
    columns = list(BASE_COLUMNS)
    for rx in rx_order:
        columns.extend(rx_metadata_columns(rx))
        columns.extend(csi_columns(rx))
    return columns


def raw_to_complex(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != CSI_VALUES_PER_RX:
        raise ValueError(f"Expected {CSI_VALUES_PER_RX} CSI values, got {values.shape[-1]}")
    pairs = values.reshape(values.shape[0], SUBCARRIERS_PER_RX, 2)
    return pairs[..., 0].astype(float) + 1j * pairs[..., 1].astype(float)


def complex_to_raw(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=32767.0, neginf=-32768.0)
    interleaved = np.empty((values.shape[0], CSI_VALUES_PER_RX), dtype=np.int16)
    interleaved[:, 0::2] = np.clip(np.rint(np.real(values)), -32768, 32767).astype(np.int16)
    interleaved[:, 1::2] = np.clip(np.rint(np.imag(values)), -32768, 32767).astype(np.int16)
    return interleaved


def sanitize_phase(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    phase = np.unwrap(np.angle(values), axis=1)
    x = np.arange(SUBCARRIERS_PER_RX, dtype=float)
    centered_x = x - x.mean()
    centered_phase = phase - phase.mean(axis=1, keepdims=True)
    denominator = float(np.sum(centered_x**2))
    slopes = centered_phase @ centered_x / denominator
    intercepts = phase.mean(axis=1) - slopes * x.mean()
    trend = slopes[:, np.newaxis] * x[np.newaxis, :] + intercepts[:, np.newaxis]
    metrics = {
        "phase_slope_mean": float(np.mean(slopes)),
        "phase_slope_std": float(np.std(slopes)),
        "phase_intercept_mean": float(np.mean(intercepts)),
        "phase_intercept_std": float(np.std(intercepts)),
    }
    return phase - trend, metrics


def hampel_filter_amplitude(amplitude: np.ndarray, window: int, n_sigma: float) -> tuple[np.ndarray, float]:
    frame = pd.DataFrame(amplitude)
    median = frame.rolling(window=window, center=True, min_periods=1).median()
    deviation = (frame - median).abs()
    mad = deviation.rolling(window=window, center=True, min_periods=1).median()
    limit = n_sigma * 1.4826 * mad.replace(0.0, np.nan)
    outliers = deviation > limit.fillna(np.inf)
    filtered = frame.mask(outliers, median).to_numpy(dtype=float)
    replacement_percent = float(outliers.to_numpy().sum() / max(outliers.size, 1) * 100.0)
    return filtered, replacement_percent


def dwt_denoise_amplitude(
    amplitude: np.ndarray,
    wavelet: str,
    requested_level: int,
    threshold_scale: float,
) -> tuple[np.ndarray, float]:
    if pywt is None:
        raise RuntimeError("Profile phase_hampel_dwt requires PyWavelets. Install with: pip install PyWavelets")
    amplitude = np.ascontiguousarray(amplitude, dtype=np.float64).copy()
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = min(requested_level, pywt.dwt_max_level(amplitude.shape[0], wavelet_obj.dec_len))
    if max_level < 1:
        return amplitude, 0.0

    denoised = np.empty_like(amplitude, dtype=float)
    thresholds: list[float] = []
    for subcarrier in range(amplitude.shape[1]):
        series = np.ascontiguousarray(amplitude[:, subcarrier], dtype=np.float64).copy()
        coeffs = [
            np.ascontiguousarray(coeff, dtype=np.float64).copy()
            for coeff in pywt.wavedec(series, wavelet_obj, mode="symmetric", level=max_level)
        ]
        finest_detail = coeffs[-1]
        sigma = float(np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6745) if finest_detail.size else 0.0
        threshold = threshold_scale * sigma * math.sqrt(2.0 * math.log(max(series.size, 2)))
        thresholds.append(threshold)
        filtered_coeffs = [coeffs[0].copy()]
        for detail in coeffs[1:]:
            if threshold <= 0.0 or not np.isfinite(threshold):
                filtered = detail.copy()
            else:
                filtered = pywt.threshold(detail, threshold, mode="soft")
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
            filtered_coeffs.append(np.ascontiguousarray(filtered, dtype=np.float64).copy())
        reconstructed = pywt.waverec(filtered_coeffs, wavelet_obj, mode="symmetric")[: series.size]
        denoised[:, subcarrier] = np.nan_to_num(np.asarray(reconstructed, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(denoised, 0.0), float(np.mean(thresholds))


def safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.size < 2 or np.std(left_flat) == 0.0 or np.std(right_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def denoise_rx(sample_id: str, rx: str, frame: pd.DataFrame, config: DenoiseConfig) -> tuple[np.ndarray, dict[str, object]]:
    length_column = f"{rx}_csi_len"
    if length_column in frame.columns and not (frame[length_column] == CSI_VALUES_PER_RX).all():
        raise ValueError(f"{sample_id} has invalid {length_column}; expected {CSI_VALUES_PER_RX}")

    raw_values = frame[csi_columns(rx)].to_numpy(dtype=float)
    raw_complex = raw_to_complex(raw_values)
    raw_amplitude = np.abs(raw_complex)
    phase = np.angle(raw_complex)
    phase_metrics = {
        "phase_slope_mean": 0.0,
        "phase_slope_std": 0.0,
        "phase_intercept_mean": 0.0,
        "phase_intercept_std": 0.0,
    }
    if config.use_phase:
        phase, phase_metrics = sanitize_phase(raw_complex)

    amplitude = raw_amplitude
    hampel_replacement_percent = 0.0
    if config.use_hampel:
        amplitude, hampel_replacement_percent = hampel_filter_amplitude(
            amplitude,
            window=config.hampel_window,
            n_sigma=config.hampel_n_sigma,
        )

    dwt_threshold_mean = 0.0
    if config.use_dwt:
        amplitude, dwt_threshold_mean = dwt_denoise_amplitude(
            amplitude,
            wavelet=config.wavelet,
            requested_level=config.wavelet_level,
            threshold_scale=config.dwt_threshold_scale,
        )

    denoised_complex = amplitude * np.exp(1j * phase)
    raw_energy = float(np.sum(raw_amplitude**2))
    denoised_energy = float(np.sum(amplitude**2))
    metrics: dict[str, object] = {
        "sample_id": sample_id,
        "person": frame["person"].iloc[0] if "person" in frame.columns else "",
        "coarse_pose": frame["coarse_pose"].iloc[0] if "coarse_pose" in frame.columns else "",
        "cell_id": frame["cell_id"].iloc[0] if "cell_id" in frame.columns else "",
        "rx": rx,
        "frames": int(len(frame)),
        "hampel_replacement_percent": hampel_replacement_percent,
        "correlation": safe_corrcoef(raw_amplitude, amplitude),
        "energy_ratio": denoised_energy / raw_energy if raw_energy > 0.0 else float("nan"),
        "rmse": float(np.sqrt(np.mean((amplitude - raw_amplitude) ** 2))),
        "mae": float(np.mean(np.abs(amplitude - raw_amplitude))),
        "dwt_applied": config.use_dwt,
        "dwt_threshold_mean": dwt_threshold_mean,
        **phase_metrics,
    }
    return complex_to_raw(denoised_complex), metrics


def denoise_sample(frame: pd.DataFrame, config: DenoiseConfig) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    output = frame.copy()
    sample_id = str(frame["sample_id"].iloc[0])
    metrics = []
    for rx in config.rx_order:
        values, rx_metrics = denoise_rx(sample_id, rx, frame, config)
        output.loc[:, csi_columns(rx)] = values
        metrics.append(rx_metrics)
    return output, metrics


def complete_groups_from_chunk(chunk: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    if chunk.empty:
        return [], chunk
    last_sample_id = chunk["sample_id"].iloc[-1]
    complete = pd.DataFrame(chunk.loc[chunk["sample_id"] != last_sample_id])
    carry = pd.DataFrame(chunk.loc[chunk["sample_id"] == last_sample_id])
    groups = [pd.DataFrame(group) for _, group in complete.groupby("sample_id", sort=False)]
    return groups, carry


def load_manifest(session_dir: Path) -> pd.DataFrame:
    path = session_dir / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest.csv: {path}")
    manifest = pd.read_csv(path)
    if "status" in manifest.columns:
        manifest = pd.DataFrame(manifest.loc[manifest["status"].fillna("").astype(str).str.lower() == "pass"])
    return manifest.sort_values(["data_file", "sample_id"]).reset_index(drop=True)


def raw_files(session_dir: Path) -> list[Path]:
    return sorted(session_dir.glob("data_P*.csv"))


def selected_samples_by_file(manifest: pd.DataFrame, person: str | None, sample_limit: int | None) -> dict[str, set[str]]:
    selected = manifest.copy()
    if person:
        selected = pd.DataFrame(selected.loc[selected["person"].astype(str).str.upper() == person.upper()])
    if sample_limit is not None:
        selected = selected.head(sample_limit)
    return {str(data_file): set(group["sample_id"].astype(str)) for data_file, group in selected.groupby("data_file", sort=True)}


def validate_input(session_dir: Path, config: DenoiseConfig) -> None:
    files = raw_files(session_dir)
    if not files:
        raise FileNotFoundError(f"No data_P*.csv files found in {session_dir}")
    required = set(required_columns(config.rx_order))
    for csv_path in files:
        header = set(pd.read_csv(csv_path, nrows=0).columns)
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")


def ensure_output_dirs(input_dir: Path, output_dir: Path, report_dir: Path, overwrite: bool) -> None:
    resolved_input = input_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_input or resolved_input in resolved_output.parents:
        raise ValueError("Output directory must not be inside the raw session directory")
    for path in (output_dir, report_dir):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def write_sidecars(session_dir: Path, output_dir: Path, manifest: pd.DataFrame, selected_ids: set[str]) -> None:
    selected = sorted(selected_ids)
    manifest.loc[manifest["sample_id"].astype(str).isin(selected)].to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8")
    quality_path = session_dir / "quality.csv"
    if quality_path.is_file():
        quality = pd.read_csv(quality_path)
        if "sample_id" in quality.columns:
            quality = pd.DataFrame(quality.loc[quality["sample_id"].astype(str).isin(selected)])
        quality.to_csv(output_dir / "quality.csv", index=False, encoding="utf-8")
    calibration_path = session_dir / "calibration.json"
    if calibration_path.is_file():
        shutil.copy2(calibration_path, output_dir / "calibration.json")


def append_frame(frame: pd.DataFrame, output_path: Path, include_header: bool) -> None:
    frame.to_csv(output_path, mode="a", header=include_header, index=False, encoding="utf-8")


def process_file(csv_path: Path, output_path: Path, wanted_ids: set[str], config: DenoiseConfig) -> tuple[list[dict[str, object]], int, int]:
    if not wanted_ids:
        return [], 0, 0
    metrics: list[dict[str, object]] = []
    processed_ids: set[str] = set()
    processed_samples = 0
    processed_frames = 0
    carry = pd.DataFrame()
    wrote_header = False

    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        groups, carry = complete_groups_from_chunk(chunk)
        for group in groups:
            sample_id = str(group["sample_id"].iloc[0])
            if sample_id not in wanted_ids:
                continue
            denoised, sample_metrics = denoise_sample(group, config)
            append_frame(denoised, output_path, include_header=not wrote_header)
            wrote_header = True
            metrics.extend(sample_metrics)
            processed_ids.add(sample_id)
            processed_samples += 1
            processed_frames += len(denoised)

    if not carry.empty:
        sample_id = str(carry["sample_id"].iloc[0])
        if sample_id in wanted_ids:
            denoised, sample_metrics = denoise_sample(carry, config)
            append_frame(denoised, output_path, include_header=not wrote_header)
            wrote_header = True
            metrics.extend(sample_metrics)
            processed_ids.add(sample_id)
            processed_samples += 1
            processed_frames += len(denoised)

    missing = sorted(wanted_ids - processed_ids)
    if missing:
        raise ValueError(f"Selected samples missing in {csv_path.name}: {missing[:5]}")
    if not wrote_header:
        pd.read_csv(csv_path, nrows=0).to_csv(output_path, index=False, encoding="utf-8")
    return metrics, processed_samples, processed_frames


def write_reports(
    report_dir: Path,
    config: DenoiseConfig,
    qc_rows: list[dict[str, object]],
    file_rows: list[dict[str, object]],
    result: RunResult,
) -> None:
    pd.DataFrame(qc_rows).to_csv(report_dir / "denoise_qc_by_sample_rx.csv", index=False, encoding="utf-8")
    pd.DataFrame(file_rows).to_csv(report_dir / "denoise_file_summary.csv", index=False, encoding="utf-8")
    summary = {**asdict(result), "config": asdict(config)}
    summary["input_dir"] = str(result.input_dir)
    summary["output_dir"] = str(result.output_dir)
    summary["report_dir"] = str(result.report_dir)
    (report_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_denoise(
    profile: str,
    session_dir: Path = RAW_SESSION_DIR,
    output_root: Path = DENOISED_ROOT,
    report_root: Path = REPORT_ROOT,
    person: str | None = None,
    sample_limit: int | None = None,
    overwrite: bool = False,
) -> RunResult:
    config = profile_config(profile)
    if config.use_dwt and pywt is None:
        raise RuntimeError("Profile phase_hampel_dwt requires PyWavelets. Install with: pip install PyWavelets")
    session_dir = session_dir.resolve()
    output_dir = output_root / profile
    report_dir = report_root / profile
    validate_input(session_dir, config)
    ensure_output_dirs(session_dir, output_dir, report_dir, overwrite=overwrite)

    manifest = load_manifest(session_dir)
    by_file = selected_samples_by_file(manifest, person=person, sample_limit=sample_limit)
    selected_groups = list(by_file.values())
    selected_ids: set[str] = set().union(*selected_groups) if selected_groups else set()
    if not selected_ids:
        raise ValueError("No samples selected for denoising")
    write_sidecars(session_dir, output_dir, manifest, selected_ids)

    files_by_name = {path.name: path for path in raw_files(session_dir)}
    missing_files = sorted(set(by_file) - set(files_by_name))
    if missing_files:
        raise FileNotFoundError(f"Manifest references missing data files: {missing_files}")

    qc_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    total_samples = 0
    total_frames = 0
    for file_name, wanted_ids in by_file.items():
        source_path = files_by_name[file_name]
        output_path = output_dir / file_name
        metrics, sample_count, frame_count = process_file(source_path, output_path, wanted_ids, config)
        qc_rows.extend({**row, "profile": profile} for row in metrics)
        file_rows.append(
            {
                "profile": profile,
                "source_file": str(source_path),
                "output_file": str(output_path),
                "selected_samples": len(wanted_ids),
                "processed_samples": sample_count,
                "processed_frames": frame_count,
            }
        )
        total_samples += sample_count
        total_frames += frame_count

    result = RunResult(
        profile=profile,
        input_dir=session_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        processed_samples=total_samples,
        processed_frames=total_frames,
    )
    (output_dir / "denoise_config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(report_dir, config, qc_rows, file_rows, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise CSI raw session data into split_and_window-compatible CSV files.")
    parser.add_argument("--profile", choices=("phase_hampel", "phase_hampel_dwt"), required=True)
    parser.add_argument("--session-dir", type=Path, default=RAW_SESSION_DIR)
    parser.add_argument("--output-root", type=Path, default=DENOISED_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--person", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_denoise(
        profile=args.profile,
        session_dir=args.session_dir,
        output_root=args.output_root,
        report_root=args.report_root,
        person=args.person,
        sample_limit=args.sample_limit,
        overwrite=bool(args.overwrite),
    )
    print(f"Profile: {result.profile}")
    print(f"Output: {result.output_dir}")
    print(f"Report: {result.report_dir}")
    print(f"Processed samples: {result.processed_samples}; frames: {result.processed_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
