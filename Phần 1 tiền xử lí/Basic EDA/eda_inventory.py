
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eda_common import (
    EXPECTED_RAW_COLUMNS,
    RX_ORDER,
    default_session_dir,
    load_manifest,
    load_quality,
    manifest_path,
    missing_columns,
    quality_path,
    raw_files,
    read_header,
    required_raw_columns,
)


EXPECTED_COLLECTION_PLAN_ID = "PART1_25ZONE_PERIMETER_V1"
EXPECTED_LAYOUT_TYPE = "perimeter_one_module_per_side"
EXPECTED_PLANNED_SAMPLE_COUNT = 1370
EXPECTED_NODE_ROLES = {"TX": "west", "RX1": "north", "RX2": "east", "RX3": "south"}
EXPECTED_SERIAL_SCHEMA = "CSI_V2_13_METADATA_128_IQ"


def calibration_path(session_dir: Path) -> Path:
    return session_dir / "calibration.json"


def load_calibration(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Missing calibration metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested_value(data: dict[str, object], dotted_key: str) -> object:
    current: object = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def calibration_contract_report(calibration: dict[str, object]) -> pd.DataFrame:
    checks = [
        ("layout_type", EXPECTED_LAYOUT_TYPE),
        ("collection_plan_id", EXPECTED_COLLECTION_PLAN_ID),
        ("planned_sample_count", EXPECTED_PLANNED_SAMPLE_COUNT),
        ("node_side_roles", EXPECTED_NODE_ROLES),
        ("csi_serial_schema", EXPECTED_SERIAL_SCHEMA),
        ("csi_values_per_rx", 128),
        ("data_columns", EXPECTED_RAW_COLUMNS),
        ("collection_plan_summary.expected_samples", EXPECTED_PLANNED_SAMPLE_COUNT),
    ]
    rows = []
    for field, expected in checks:
        actual = nested_value(calibration, field)
        rows.append(
            {
                "field": field,
                "expected": json.dumps(expected, ensure_ascii=False) if isinstance(expected, dict) else expected,
                "actual": json.dumps(actual, ensure_ascii=False) if isinstance(actual, dict) else actual,
                "status": "pass" if actual == expected else "fail",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CSI dataset schema and inventory tables in memory.")
    parser.add_argument("--session-dir", type=Path, default=default_session_dir())
    return parser.parse_args()


def raw_file_report(csv_path: Path) -> dict[str, object]:
    header = read_header(csv_path)
    required = required_raw_columns()
    missing = missing_columns(header, required)
    usecols = ["sample_id", *(f"{rx}_csi_len" for rx in RX_ORDER)]
    row_count = 0
    sample_ids: set[str] = set()
    bad_csi_len = {rx: 0 for rx in RX_ORDER}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=50_000):
        row_count += len(chunk)
        sample_ids.update(chunk["sample_id"].dropna().astype(str).unique())
        for rx in RX_ORDER:
            bad_csi_len[rx] += int((chunk[f"{rx}_csi_len"] != 128).sum())

    return {
        "file": csv_path.name,
        "rows": row_count,
        "sample_count": len(sample_ids),
        "column_count": len(header),
        "column_count_ok": len(header) == EXPECTED_RAW_COLUMNS,
        "missing_required_columns": "|".join(missing),
        "rx1_bad_csi_len": bad_csi_len["RX1"],
        "rx2_bad_csi_len": bad_csi_len["RX2"],
        "rx3_bad_csi_len": bad_csi_len["RX3"],
    }


def build_inventory_tables(session_dir: Path = default_session_dir()) -> dict[str, pd.DataFrame | dict[str, object]]:
    manifest = load_manifest(manifest_path(session_dir))
    quality = load_quality(quality_path(session_dir))
    calibration = load_calibration(calibration_path(session_dir))
    files = raw_files(session_dir)
    if not files:
        raise ValueError(f"No data_P*.csv files found in {session_dir}")

    raw_report = pd.DataFrame(raw_file_report(path) for path in files)
    calibration_report = calibration_contract_report(calibration)
    summary = {
        "session_dir": str(session_dir),
        "manifest_rows": len(manifest),
        "quality_rows": len(quality),
        "raw_file_count": len(files),
        "raw_rows": int(raw_report["rows"].sum()),
        "manifest_pass_rows": int((manifest["status"].str.lower() == "pass").sum()),
        "quality_pass_rows": int((quality["status"].str.lower() == "pass").sum()),
        "quality_fail_rows": int((quality["status"].str.lower() == "fail").sum()),
        "all_raw_headers_438_columns": bool(raw_report["column_count_ok"].all()),
        "calibration_contract_pass": bool((calibration_report["status"] == "pass").all()),
        "planned_sample_count": int(calibration.get("planned_sample_count", 0)),
        "calibration_expected_clean_rows": int(calibration.get("expected_clean_rows", 0)),
        "total_bad_csi_len_rows": int(
            raw_report[["rx1_bad_csi_len", "rx2_bad_csi_len", "rx3_bad_csi_len"]].sum().sum()
        ),
    }
    inventory_summary = pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()])
    return {
        "inventory_summary": inventory_summary,
        "raw_file_inventory": raw_report,
        "calibration_contract": calibration_report,
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    tables = build_inventory_tables(args.session_dir)
    summary = tables["summary"]
    if not isinstance(summary, dict):
        raise TypeError("inventory summary table bundle is malformed")
    print("Built inventory tables in memory")
    print(f"manifest_rows={summary['manifest_rows']} raw_rows={summary['raw_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
