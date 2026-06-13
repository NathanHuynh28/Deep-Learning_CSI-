
from __future__ import annotations

import argparse
import re

import pandas as pd

from eda_common import default_figures_dir, default_session_dir


DOCUMENTED_TARGET_SAMPLES = 1370
DOCUMENTED_PASS_FRAMES_PER_SAMPLE = 300
DOCUMENTED_TARGET_CLEAN_ROWS = DOCUMENTED_TARGET_SAMPLES * DOCUMENTED_PASS_FRAMES_PER_SAMPLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a preliminary Part 1 EDA summary in memory.")
    return parser.parse_args()


def metric_value(metrics: pd.DataFrame, name: str, default: str = "N/A") -> str:
    if metrics.empty or "metric" not in metrics.columns or "value" not in metrics.columns:
        return default
    selected = metrics.loc[metrics["metric"] == name, "value"]
    return str(selected.iloc[0]) if not selected.empty else default


def metric_float(metrics: pd.DataFrame, name: str, default: float = 0.0) -> float:
    value = metric_value(metrics, name, "")
    try:
        return float(value)
    except ValueError:
        return default


def percent(actual: float, target: float) -> float:
    return actual / target * 100.0 if target > 0 else 0.0


def top_rows(frame: pd.DataFrame, label_column: str, count_column: str = "count", limit: int = 5) -> str:
    if frame.empty or label_column not in frame.columns or count_column not in frame.columns:
        return "N/A"
    rows = []
    ordered = frame.sort_values(count_column, ascending=False).head(limit)
    for row in ordered.to_dict(orient="records"):
        percent = row.get("percent")
        suffix = f" ({float(percent):.1f}%)" if percent is not None and not pd.isna(percent) else ""
        rows.append(f"{row[label_column]}={int(row[count_column])}{suffix}")
    return ", ".join(rows)


def fail_reason_summary(quality: pd.DataFrame) -> str:
    if quality.empty or "quality_status" not in quality.columns:
        return "N/A"
    return ", ".join(f"{row['quality_status']}={int(row['count'])}" for row in quality.to_dict(orient="records"))


def quality_reason_summary(failure_reasons: pd.DataFrame) -> str:
    if failure_reasons.empty:
        return "none"
    frame_counts = []
    for reason in failure_reasons["reason"].astype(str):
        match = re.match(r"^not_enough_synced_frames_(\d+)$", reason)
        if match is None:
            frame_counts = []
            break
        frame_counts.append(int(match.group(1)))
    if frame_counts:
        total = sum(int(count) for count in failure_reasons["count"].tolist())
        return f"not_enough_synced_frames={total} failures, clean-frame range {min(frame_counts)}-{max(frame_counts)}"
    rows = []
    for row in failure_reasons.sort_values("count", ascending=False).to_dict(orient="records"):
        reason = re.sub(r"\s+", "_", str(row["reason"]).strip()) or "unknown"
        rows.append(f"{reason}={int(row['count'])}")
    return ", ".join(rows)


def sparse_cell_summary(cell_risk: pd.DataFrame) -> str:
    if cell_risk.empty:
        return "N/A"
    high = cell_risk[cell_risk["risk"] == "high"]
    medium = cell_risk[cell_risk["risk"] == "medium"]
    return f"high={len(high)}, medium={len(medium)}, low={int((cell_risk['risk'] == 'low').sum())}"


def calibration_status(calibration: pd.DataFrame) -> str:
    if calibration.empty or "status" not in calibration.columns:
        return "N/A"
    failed = calibration[calibration["status"] != "pass"]
    if failed.empty:
        return "pass"
    return "fail: " + ", ".join(failed["field"].astype(str).tolist())


def build_summary_lines(tables: dict[str, pd.DataFrame]) -> list[str]:
    inventory = tables.get("inventory_summary", pd.DataFrame())
    raw_files = tables.get("raw_file_inventory", pd.DataFrame())
    calibration = tables.get("calibration_contract", pd.DataFrame())
    quality_status = tables.get("quality_status_distribution", pd.DataFrame())
    failure_reasons = tables.get("quality_failure_reasons", pd.DataFrame())
    pose = tables.get("coarse_pose_distribution", pd.DataFrame())
    orientation = tables.get("orientation_label_distribution", pd.DataFrame())
    csi_len = tables.get("csi_len_summary", pd.DataFrame())
    rssi = tables.get("rssi_by_rx", pd.DataFrame())
    split_groups = tables.get("split_group_feasibility", pd.DataFrame())
    person_independent = tables.get("person_independent_feasibility", pd.DataFrame())
    label_issues = tables.get("label_consistency_issues", pd.DataFrame())
    cell_risk = tables.get("cell_risk_summary", pd.DataFrame())
    pose_geometry = tables.get("pose_geometry_summary", pd.DataFrame())

    manifest_rows = metric_value(inventory, "manifest_rows")
    raw_rows = metric_value(inventory, "raw_rows")
    manifest_count = metric_float(inventory, "manifest_rows")
    raw_row_count = metric_float(inventory, "raw_rows")
    calibration_expected_rows = metric_float(inventory, "calibration_expected_clean_rows")
    clean_row_target = calibration_expected_rows if calibration_expected_rows > 0 else DOCUMENTED_TARGET_CLEAN_ROWS
    clean_row_target_label = (
        "Calibration/session expected clean rows" if calibration_expected_rows > 0 else "Documented clean-row target"
    )
    header_ok = metric_value(inventory, "all_raw_headers_438_columns")
    bad_csi = metric_value(inventory, "total_bad_csi_len_rows")
    empty_files = [] if raw_files.empty else raw_files.loc[raw_files["rows"] == 0, "file"].astype(str).tolist()
    issue_count = max(len(label_issues), 0)
    recommended_people = []
    if not person_independent.empty:
        recommended_people = person_independent.loc[
            person_independent["recommended_as_person_independent_test"] == True, "held_out_person"
        ].astype(str).tolist()

    lines = [
        "# Preliminary EDA Report - Part 1 CSI Dataset",
        "",
        "## Dataset readiness verdict",
        "",
        f"- Manifest samples: {manifest_rows}",
        f"- Documented sample target: {DOCUMENTED_TARGET_SAMPLES} ({percent(manifest_count, DOCUMENTED_TARGET_SAMPLES):.1f}% collected)",
        f"- Raw frame rows: {raw_rows}",
        f"- {clean_row_target_label}: {int(clean_row_target)} ({percent(raw_row_count, clean_row_target):.1f}% collected)",
        f"- Calibration expected clean rows: {int(calibration_expected_rows)}",
        f"- Raw schema 438 columns valid: {header_ok}",
        f"- Calibration metadata contract: {calibration_status(calibration)}",
        f"- Bad CSI-length rows: {bad_csi}",
        f"- Quality status: {fail_reason_summary(quality_status)}",
        f"- Quality failure reasons: {quality_reason_summary(failure_reasons)}",
        f"- Header-only raw files: {', '.join(empty_files) if empty_files else 'none'}",
        "",
        "## Label balance",
        "",
        f"- Pose distribution: {top_rows(pose, 'coarse_pose')}",
        f"- Orientation distribution: {top_rows(orientation, 'orientation_label')}",
        f"- Label consistency issues: {issue_count}",
        f"- Sparse cell risk: {sparse_cell_summary(cell_risk)}",
        "",
        "## Signal quality",
        "",
    ]
    if not csi_len.empty:
        for row in csi_len.to_dict(orient="records"):
            lines.append(f"- {row['rx']}: rows={int(row['rows'])}, bad_csi_len={int(row['bad_csi_len_rows'])}, range={int(row['min_csi_len'])}-{int(row['max_csi_len'])}")
    if not rssi.empty:
        for row in rssi.to_dict(orient="records"):
            lines.append(f"- {row['rx']} RSSI: mean={float(row['mean']):.2f} dBm, median={float(row['median']):.2f}, min={float(row['min']):.2f}, max={float(row['max']):.2f}")
    lines.extend(["", "## Split guidance", ""])
    if not split_groups.empty:
        for row in split_groups.to_dict(orient="records"):
            lines.append(
                f"- {row['group_key']}: groups={int(row['group_count'])}, "
                + f"approx train/val/test={int(row['approx_train_groups_70'])}/{int(row['approx_val_groups_15'])}/{int(row['approx_test_groups_15'])}"
            )
    lines.append(f"- Recommended person-independent held-out candidates: {', '.join(recommended_people) if recommended_people else 'none'}")
    lines.extend(["", "## Geometry and footprint", ""])
    if not pose_geometry.empty:
        for row in pose_geometry.sort_values("samples", ascending=False).to_dict(orient="records"):
            lines.append(
                f"- {row['coarse_pose']}: samples={int(row['samples'])}, "
                + f"mean occupied cells={float(row['mean_occupied_cells']):.2f}, "
                + f"mean footprint area={float(row['mean_footprint_area_m2']):.3f} m2"
            )
    lines.extend(
        [
            "",
            "## Practical conclusion",
            "",
            "- The current session is usable for preliminary EDA and model prototyping, but it is not a complete Part 1 training collection yet.",
            f"- Current coverage is {int(manifest_count)}/{DOCUMENTED_TARGET_SAMPLES} samples and {int(raw_row_count)}/{int(clean_row_target)} {clean_row_target_label.lower()}.",
            "- The dataset is intentionally imbalanced by collection design; center/interior cells and Standing/Sitting have more coverage than boundary cells, Empty, and Transition.",
            "- Split must be done before windowing by sample/trial/person groups to avoid leakage.",
            "- P4 should not be used as the only person-independent test because it lacks several human poses.",
        ]
    )
    return lines


def main() -> int:
    parse_args()
    from eda_geometry import build_geometry_tables
    from eda_inventory import build_inventory_tables
    from eda_labels import build_label_tables
    from eda_signal_quality import build_signal_quality_tables

    session_dir = default_session_dir()
    figures_dir = default_figures_dir()
    inventory_outputs = build_inventory_tables(session_dir)
    tables = {key: value for key, value in inventory_outputs.items() if isinstance(value, pd.DataFrame)}
    tables.update(build_label_tables(figures_dir=figures_dir, save_figures=False))
    tables.update(build_signal_quality_tables(session_dir, figures_dir, save_figures=False))
    tables.update(build_geometry_tables(figures_dir=figures_dir, save_figures=False))
    lines = build_summary_lines(tables)
    print(f"Built preliminary EDA summary in memory: {len(lines)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
