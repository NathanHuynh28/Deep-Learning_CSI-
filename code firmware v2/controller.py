"""
CSI Dataset Controller V2 (MAC-SYNC & REAL-WORLD TOLERANCE)

Run with VS Code terminal after flashing TX firmware V3 and RX firmware V4.7+.
Manual configuration is intentionally kept at the top of this file.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import serial
except ImportError:  # pragma: no cover - operator message only
    serial = None

# ===================== EDIT BEFORE EACH SESSION =====================

SESSION_ID = "session1"
OPERATOR = "nguyen"
ROOM_NAME = "room1"
ROOM_SIZE_M = {"width": 3.00, "depth": 3.00}
LAYOUT_ID = "layout1"
LAYOUT_TYPE = "perimeter_one_module_per_side"
COLLECTION_PLAN_ID = "PART1_25ZONE_PERIMETER_V1"
CALIBRATION_ID = "calibration1"
CHANNEL = 1
EXPECTED_TX_ID = 1

COM_PORTS = {
    "RX1": "COM17",
    "RX2": "COM23",
    "RX3": "COM20",
}

BAUDRATE = 921600

EMPTY_PERSON = "P0"
EMPTY_CELL_ID = "NO_CELL"
PERSONS = ["P1", "P2", "P3", "P4", "P5"]
OUTPUT_PERSONS = [EMPTY_PERSON] + PERSONS
CELL_GRID_ROWS = 5
CELL_GRID_COLS = 5
CELLS = [f"C{index:02d}" for index in range(1, CELL_GRID_ROWS * CELL_GRID_COLS + 1)]
COARSE_POSES = ["Empty", "Standing", "Sitting", "Lying", "Arms_Out", "Walk_in_place", "Transition"]
ORIENTATION_LABELS = ["Empty", "Facing_RX", "Facing_TX", "Facing_Left", "Facing_Right", "Diagonal_RX1", "Diagonal_RX3"]
HEIGHT_BANDS = ["Empty", "Low", "Mid", "Tall"]
LABEL_QUALITIES = ["Good", "Approx", "Reject"]

CARDINAL_ORIENTATIONS = ["Facing_RX", "Facing_TX", "Facing_Left", "Facing_Right"]
DIAGONAL_ORIENTATIONS = ["Diagonal_RX1", "Diagonal_RX3"]

COLLECTION_MODE = "part1_large"
StageConfig = Dict[str, Any]
CsiPayload = Dict[str, Any]
SyncedFrame = Tuple[int, Dict[str, CsiPayload]]

STAGE_CONFIGS: Dict[str, StageConfig] = {
    "smoke": {
        "description": "Fast hardware/schema check or small calibration sanity run.",
        "persons": ["P1"],
        "cells": ["C13"],
        "poses": ["Empty", "Standing", "Sitting", "Lying"],
        "orientations": ["Empty", "Facing_RX", "Facing_TX"],
        "trials_per_case": 3,
    },
    "pilot": {
        "description": "Small calibration/adjustment run over central cells with cardinal directions.",
        "persons": ["P1"],
        "cells": ["C08", "C12", "C13", "C14", "C18"],
        "poses": ["Empty", "Standing", "Sitting", "Lying", "Arms_Out", "Walk_in_place", "Transition"],
        "orientations": ["Empty"] + CARDINAL_ORIENTATIONS,
        "trials_per_case": 5,
    },
}

ACTIVE_STAGE_CONFIG = STAGE_CONFIGS.get(COLLECTION_MODE)
COLLECTION_PERSONS = ACTIVE_STAGE_CONFIG["persons"] if ACTIVE_STAGE_CONFIG else PERSONS
COLLECTION_CELLS = ACTIVE_STAGE_CONFIG["cells"] if ACTIVE_STAGE_CONFIG else CELLS
COLLECTION_POSES = ACTIVE_STAGE_CONFIG["poses"] if ACTIVE_STAGE_CONFIG else COARSE_POSES
COLLECTION_ORIENTATIONS = ACTIVE_STAGE_CONFIG["orientations"] if ACTIVE_STAGE_CONFIG else ORIENTATION_LABELS
TRIALS_PER_CASE = ACTIVE_STAGE_CONFIG["trials_per_case"] if ACTIVE_STAGE_CONFIG else 1

LABEL_SCHEMA_VERSION = "OCCUPANCY_FOOTPRINT_V1"
CELL_WIDTH_M = ROOM_SIZE_M["width"] / CELL_GRID_COLS
CELL_DEPTH_M = ROOM_SIZE_M["depth"] / CELL_GRID_ROWS
CELL_ORIGIN_X_M = 0.00
CELL_ORIGIN_Y_M = 0.00
DEFAULT_LABEL_QUALITY = "Good"
TRANSITION_PATTERN = "sit_down_then_stand_up"

ORIENTATION_DEGREES = {
    "Empty": 0,
    "Facing_RX": 0,
    "Facing_TX": 180,
    "Facing_Left": 270,
    "Facing_Right": 90,
    "Diagonal_RX1": 315,
    "Diagonal_RX3": 45,
}

POSE_DEFAULTS = {
    "Empty": {"presence": 0, "height_band": "Empty", "footprint_length_m": 0.00, "footprint_width_m": 0.00},
    "Standing": {"presence": 1, "height_band": "Tall", "footprint_length_m": 0.45, "footprint_width_m": 0.45},
    "Sitting": {"presence": 1, "height_band": "Mid", "footprint_length_m": 0.70, "footprint_width_m": 0.60},
    "Lying": {"presence": 1, "height_band": "Low", "footprint_length_m": 1.80, "footprint_width_m": 0.65},
    "Arms_Out": {"presence": 1, "height_band": "Tall", "footprint_length_m": 0.55, "footprint_width_m": 1.50},
    "Walk_in_place": {"presence": 1, "height_band": "Tall", "footprint_length_m": 0.90, "footprint_width_m": 0.80},
    "Transition": {"presence": 1, "height_band": "Mid", "footprint_length_m": 1.00, "footprint_width_m": 0.80},
}

TX_POSITION_M = {"x": -0.50, "y": 1.50, "height": 1.00}
RX_POSITIONS_M = {
    "RX1": {"x": 1.50, "y": -0.50, "height": 1.00},
    "RX2": {"x": 3.50, "y": 1.50, "height": 1.00},
    "RX3": {"x": 1.50, "y": 3.50, "height": 1.00},
}
NODE_SIDE_ROLES = {"TX": "west", "RX1": "north", "RX2": "east", "RX3": "south"}
RX_ORDER_NOTE = "RX_ORDER is logical north/east/south: RX1=north, RX2=east, RX3=south."

COORDINATE_CONVENTION = {
    "view": "top-down room map",
    "origin": "top-left corner of the 5x5 zone grid",
    "x_axis": "+x goes west-to-east across the grid",
    "y_axis": "+y goes north-to-south down the grid",
}

LAYOUT_DESCRIPTION = (
    "Perimeter one-module-per-side layout for a 3m x 3m grid: TX west/left at "
    "(-0.50, 1.50), RX1 north/top at (1.50, -0.50), RX2 east/right at "
    "(3.50, 1.50), RX3 south/bottom at (1.50, 3.50), all height 1.00m. "
    "These are outside-grid offsets; if the room cannot place nodes outside the grid, "
    "measure the actual coordinates and start a new layout_id/calibration_id."
)

ORIENTATION_DESCRIPTIONS = {
    "Empty": "No actor in the capture area.",
    "Facing_RX": "+x: actor faces east toward RX2.",
    "Facing_TX": "-x: actor faces west toward TX.",
    "Facing_Left": "-y: actor faces north toward RX1.",
    "Facing_Right": "+y: actor faces south toward RX3.",
    "Diagonal_RX1": "+x/-y: actor faces northeast toward the RX1/RX2 quadrant.",
    "Diagonal_RX3": "+x/+y: actor faces southeast toward the RX2/RX3 quadrant.",
}

COLLECTION_STAGE_NOTES = {
    "smoke": "Small hardware/schema or later-day sanity calibration run; P0/NO_CELL Empty plus P1/C13 Standing/Sitting/Lying, minimal orientations, 3 trials.",
    "pilot": "Small calibration/adjustment session; P1, C08/C12/C13/C14/C18, core poses, 4 cardinal orientations, 5 trials.",
    "part1_large": "Full 1370-sample 25-zone training session with distributed Empty checkpoints; inspect with --plan-only before collection.",
}

INTERIOR_3X3_CELLS = ["C07", "C08", "C09", "C12", "C13", "C14", "C17", "C18", "C19"]
CROSS_CELLS = ["C08", "C12", "C13", "C14", "C18"]
INTERIOR_CORNER_CELLS = ["C07", "C09", "C17", "C19"]
CENTER_AND_INTERIOR_CORNERS = ["C13", "C07", "C09", "C17", "C19"]
EDGE_CENTER_CELLS = ["C03", "C11", "C15", "C23"]
PERIMETER_CELLS = [cell for cell in CELLS if cell not in INTERIOR_3X3_CELLS]
SAFE_BODY_CELLS = CROSS_CELLS
EMPTY_BASELINE_SAMPLES = 80
EMPTY_CHECKPOINT_COUNT = 16
EMPTY_TRIALS_PER_CHECKPOINT = 5
PART1_HUMAN_SAMPLE_COUNT = 1290
PART1_EXPECTED_SAMPLE_COUNT = 1370
EXPECTED_POSE_COUNTS = {
    "Empty": 80,
    "Standing": 390,
    "Sitting": 310,
    "Arms_Out": 150,
    "Lying": 190,
    "Walk_in_place": 160,
    "Transition": 90,
}
EXPECTED_PERSON_POSE_COUNTS = {
    "Standing": 78,
    "Sitting": 62,
    "Arms_Out": 30,
    "Lying": 38,
    "Walk_in_place": 32,
    "Transition": 18,
}

PERIMETER_IMPORTANT_DIRECTIONS = {
    "C01": ["Facing_RX", "Facing_Right"],
    "C02": ["Facing_Right", "Facing_RX"],
    "C03": ["Facing_Right", "Facing_RX"],
    "C04": ["Facing_Right", "Facing_TX"],
    "C05": ["Facing_TX", "Facing_Right"],
    "C06": ["Facing_RX", "Facing_Right"],
    "C10": ["Facing_TX", "Facing_Right"],
    "C11": ["Facing_RX", "Facing_Left"],
    "C15": ["Facing_TX", "Facing_Left"],
    "C16": ["Facing_RX", "Facing_Left"],
    "C20": ["Facing_TX", "Facing_Left"],
    "C21": ["Facing_RX", "Facing_Left"],
    "C22": ["Facing_Left", "Facing_RX"],
    "C23": ["Facing_Left", "Facing_RX"],
    "C24": ["Facing_Left", "Facing_TX"],
    "C25": ["Facing_TX", "Facing_Left"],
}
PERIMETER_INWARD_DIRECTIONS = [(cell, directions[0]) for cell, directions in PERIMETER_IMPORTANT_DIRECTIONS.items()]
INTERIOR_CORNER_INWARD_DIRECTIONS = [
    ("C07", "Facing_RX"),
    ("C07", "Facing_Right"),
    ("C09", "Facing_TX"),
    ("C09", "Facing_Right"),
    ("C17", "Facing_RX"),
    ("C17", "Facing_Left"),
    ("C19", "Facing_TX"),
    ("C19", "Facing_Left"),
]
EDGE_CENTER_INWARD_DIRECTIONS = [
    ("C03", "Facing_Right"),
    ("C11", "Facing_RX"),
    ("C15", "Facing_TX"),
    ("C23", "Facing_Left"),
]
TRANSITION_AXIS_DIRECTIONS = ["Facing_RX", "Facing_TX"]
DIAGONAL_FOCUS_DIRECTIONS = ["Diagonal_RX1", "Diagonal_RX3"]
LYING_DIAGONAL_CELLS = ["C13", "C08", "C18"]
TRANSITION_INTERIOR_CORNER_DIRECTIONS = [
    ("C07", "Facing_RX"),
    ("C09", "Facing_TX"),
    ("C17", "Facing_RX"),
    ("C19", "Facing_TX"),
]

# --- ĐIỀU CHỈNH LOGIC KIỂM DUYỆT THỰC TẾ CHO DEEP LEARNING ---
RECORD_SECONDS = 5.0
TARGET_FRAME_RATE = 100

# Chỉ cần thu gom được 200 khung đồng bộ (~40 FPS) trong 5 giây là quá đủ cho AI nội suy.
MIN_SYNCED_FRAMES = 240 
MIN_RX_RATE_HZ = 75.0

# Cho phép rớt gói rải rác, nhưng không được phép mất liên tục quá 30 gói (~0.3 giây mù tín hiệu).
MAX_CONSECUTIVE_MISSING_SEQ = 30 
PART1_EXPECTED_CLEAN_ROWS = PART1_EXPECTED_SAMPLE_COUNT * MIN_SYNCED_FRAMES

DATASET_ROOT = Path("Dataset_CSI_3D_v2")

# ====================================================================

RX_ORDER = ["RX1", "RX2", "RX3"]

CSI_VALUES_PER_RX = 128
CSI_METADATA_PARTS = 13
EXPECTED_PART_COUNT = CSI_METADATA_PARTS + CSI_VALUES_PER_RX
CSI_SERIAL_SCHEMA = "CSI_V2_13_METADATA_128_IQ"
EXPECTED_DATA_COLUMNS = 27 + len(RX_ORDER) * (9 + CSI_VALUES_PER_RX)
MAX_PREFLIGHT_SOFT_DROPS = 5

recording = False
stop_threads = False
buffer_lock = threading.Lock()

thread_buffers: Dict[str, List[CsiPayload]] = {rx: [] for rx in RX_ORDER}
recording_seen_seqs: Dict[str, set[int]] = {rx: set() for rx in RX_ORDER}

rx_total_lines = defaultdict(int)
rx_valid_lines = defaultdict(int)
rx_malformed_lines = defaultdict(int)
rx_duplicate_lines = defaultdict(int)
rx_first_word_invalid_lines = defaultdict(int)
rx_wrong_id_lines = defaultdict(int)
rx_last_seen = defaultdict(float)
rx_last_channel = defaultdict(lambda: None)
serial_handles: Dict[str, Any] = {}
passed_sample_ids: set[str] = set()
rx_stat_values: Dict[str, Dict[str, int]] = defaultdict(dict)
rx_wrong_stat_values: Dict[str, Dict[str, Any]] = defaultdict(dict)
stats_lock = threading.Lock()

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESP32 CSI Dataset Controller V2")
    parser.add_argument("--plan-only", action="store_true", help="Print the Part 1 collection plan and exit.")
    parser.add_argument("--preflight-only", action="store_true", help="Run the standard preflight check and exit.")
    parser.add_argument("--soak-seconds", type=int, default=5, help="Seconds to run preflight/soak diagnostics.")
    parser.add_argument("--soak-only", action="store_true", help="Run soak diagnostics for --soak-seconds and exit without collection.")
    args, _ = parser.parse_known_args()
    if args.soak_seconds < 1:
        parser.error("--soak-seconds must be >= 1")
    return args

CLI_ARGS = parse_cli_args()
PREFLIGHT_ONLY = CLI_ARGS.preflight_only
PLAN_ONLY = CLI_ARGS.plan_only
SOAK_ONLY = CLI_ARGS.soak_only
SOAK_SECONDS = CLI_ARGS.soak_seconds

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def session_short_id() -> str:
    return SESSION_ID.split("_")[0]

def session_dir() -> Path:
    return DATASET_ROOT / SESSION_ID

def ensure_dependency() -> None:
    if serial is None:
        print("ERROR: Missing dependency 'pyserial'. Install with: pip install pyserial")
        sys.exit(1)

def safe_pose_name(pose: str) -> str:
    return pose.replace(" ", "_")

def sample_id(person: str, cell_id: str, coarse_pose: str, orientation_label: str, trial: int) -> str:
    return f"{session_short_id()}_{person}_{cell_id}_{safe_pose_name(coarse_pose)}_{orientation_label}_T{trial:03d}"

def data_file_for_person(person: str) -> Path:
    return session_dir() / f"data_{person}.csv"

def cell_geometry(cell_id: str) -> Dict[str, Any]:
    if cell_id not in CELLS:
        raise ValueError(f"Unknown cell: {cell_id}")
    index = CELLS.index(cell_id)
    row = index // CELL_GRID_COLS
    col = index % CELL_GRID_COLS
    return {
        "cell_id": cell_id,
        "row": row,
        "col": col,
        "center_x_m": CELL_ORIGIN_X_M + (col + 0.5) * CELL_WIDTH_M,
        "center_y_m": CELL_ORIGIN_Y_M + (row + 0.5) * CELL_DEPTH_M,
        "width_m": CELL_WIDTH_M,
        "depth_m": CELL_DEPTH_M,
    }

def cell_layout() -> Dict[str, Dict[str, Any]]:
    return {cell_id: cell_geometry(cell_id) for cell_id in CELLS}

def footprint_cells(cell_id: str, coarse_pose: str, orientation_label: str) -> List[str]:
    if coarse_pose == "Empty":
        return []
    geometry = cell_geometry(cell_id)
    defaults = POSE_DEFAULTS[coarse_pose]
    length_m = float(defaults["footprint_length_m"])
    width_m = float(defaults["footprint_width_m"])
    yaw_deg = ORIENTATION_DEGREES[orientation_label]
    if yaw_deg in (45, 315):
        span = max(2, round(length_m / max(CELL_WIDTH_M, CELL_DEPTH_M)))
        row_direction = -1 if yaw_deg == 315 else 1
        col_direction = 1
        cells = {cell_id}
        center_row = int(geometry["row"])
        center_col = int(geometry["col"])
        for step in range(1, span):
            for row, col in [
                (center_row + row_direction * step, center_col + col_direction * step),
                (center_row - row_direction * step, center_col - col_direction * step),
            ]:
                if 0 <= row < CELL_GRID_ROWS and 0 <= col < CELL_GRID_COLS:
                    cells.add(CELLS[row * CELL_GRID_COLS + col])
        return [cell for cell in CELLS if cell in cells]
    yaw_is_x_axis = yaw_deg in (0, 180)
    span_cols = max(1, round((length_m if yaw_is_x_axis else width_m) / CELL_WIDTH_M))
    span_rows = max(1, round((width_m if yaw_is_x_axis else length_m) / CELL_DEPTH_M))
    center_row = int(geometry["row"])
    center_col = int(geometry["col"])
    row_start = max(0, center_row - span_rows // 2)
    row_end = min(CELL_GRID_ROWS - 1, row_start + span_rows - 1)
    row_start = max(0, row_end - span_rows + 1)
    col_start = max(0, center_col - span_cols // 2)
    col_end = min(CELL_GRID_COLS - 1, col_start + span_cols - 1)
    col_start = max(0, col_end - span_cols + 1)
    return [CELLS[row * CELL_GRID_COLS + col] for row in range(row_start, row_end + 1) for col in range(col_start, col_end + 1)]

def label_metadata(cell_id: str, coarse_pose: str, orientation_label: str) -> Dict[str, Any]:
    if coarse_pose not in COARSE_POSES:
        raise ValueError(f"Unknown coarse pose: {coarse_pose}")
    if orientation_label not in ORIENTATION_LABELS:
        raise ValueError(f"Unknown orientation: {orientation_label}")
    defaults = POSE_DEFAULTS[coarse_pose]
    if coarse_pose == "Empty":
        return {
            "presence": 0,
            "cell_id": "Empty",
            "cell_row": "",
            "cell_col": "",
            "center_x_m": "0.000",
            "center_y_m": "0.000",
            "cell_width_m": f"{CELL_WIDTH_M:.3f}",
            "cell_depth_m": f"{CELL_DEPTH_M:.3f}",
            "occupied_cells": "",
            "coarse_pose": coarse_pose,
            "orientation_label": "Empty",
            "orientation_deg": 0,
            "height_band": "Empty",
            "footprint_length_m": "0.000",
            "footprint_width_m": "0.000",
            "footprint_yaw_deg": 0,
            "label_quality": DEFAULT_LABEL_QUALITY,
        }
    geometry = cell_geometry(cell_id)
    orientation_deg = ORIENTATION_DEGREES[orientation_label]
    return {
        "presence": defaults["presence"],
        "cell_id": cell_id,
        "cell_row": geometry["row"],
        "cell_col": geometry["col"],
        "center_x_m": f"{geometry['center_x_m']:.3f}",
        "center_y_m": f"{geometry['center_y_m']:.3f}",
        "cell_width_m": f"{geometry['width_m']:.3f}",
        "cell_depth_m": f"{geometry['depth_m']:.3f}",
        "occupied_cells": "|".join(footprint_cells(cell_id, coarse_pose, orientation_label)),
        "coarse_pose": coarse_pose,
        "orientation_label": orientation_label,
        "orientation_deg": orientation_deg,
        "height_band": defaults["height_band"],
        "footprint_length_m": f"{defaults['footprint_length_m']:.3f}",
        "footprint_width_m": f"{defaults['footprint_width_m']:.3f}",
        "footprint_yaw_deg": orientation_deg,
        "label_quality": DEFAULT_LABEL_QUALITY,
    }

def should_capture_case(person: str, cell_id: str, coarse_pose: str, orientation_label: str) -> bool:
    if coarse_pose == "Empty":
        return person == EMPTY_PERSON and cell_id == EMPTY_CELL_ID and orientation_label == "Empty"
    if person not in COLLECTION_PERSONS or cell_id not in COLLECTION_CELLS:
        return False
    if coarse_pose not in COLLECTION_POSES or orientation_label not in COLLECTION_ORIENTATIONS:
        return False
    if orientation_label == "Empty":
        return False
    if coarse_pose in {"Standing", "Sitting", "Arms_Out", "Walk_in_place"}:
        return orientation_label in CARDINAL_ORIENTATIONS
    if coarse_pose in {"Lying", "Transition"}:
        return orientation_label in CARDINAL_ORIENTATIONS + DIAGONAL_ORIENTATIONS
    return False

def planned_cases() -> List[Tuple[str, str, str, str, int]]:
    if COLLECTION_MODE != "part1_large":
        cases = []
        if "Empty" in COLLECTION_POSES and "Empty" in COLLECTION_ORIENTATIONS:
            for trial in range(1, TRIALS_PER_CASE + 1):
                cases.append((EMPTY_PERSON, EMPTY_CELL_ID, "Empty", "Empty", trial))
        for person in COLLECTION_PERSONS:
            for cell_id in COLLECTION_CELLS:
                for coarse_pose in COLLECTION_POSES:
                    if coarse_pose == "Empty":
                        continue
                    for orientation_label in COLLECTION_ORIENTATIONS:
                        if not should_capture_case(person, cell_id, coarse_pose, orientation_label):
                            continue
                        for trial in range(1, TRIALS_PER_CASE + 1):
                            cases.append((person, cell_id, coarse_pose, orientation_label, trial))
        return cases

    cases: List[Tuple[str, str, str, str, int]] = []
    trial_counters: Dict[Tuple[str, str, str, str], int] = defaultdict(int)

    def add_empty_checkpoint() -> None:
        for _ in range(EMPTY_TRIALS_PER_CHECKPOINT):
            add_case(EMPTY_PERSON, EMPTY_CELL_ID, "Empty", "Empty")

    def add_case(person: str, cell_id: str, coarse_pose: str, orientation_label: str) -> None:
        key = (person, cell_id, coarse_pose, orientation_label)
        trial_counters[key] += 1
        cases.append((person, cell_id, coarse_pose, orientation_label, trial_counters[key]))

    def add_cell_orientations(person: str, cell_ids: List[str], coarse_pose: str, orientations: List[str]) -> None:
        for cell_id in cell_ids:
            for orientation_label in orientations:
                add_case(person, cell_id, coarse_pose, orientation_label)

    def add_cell_orientation_pairs(person: str, coarse_pose: str, pairs: List[Tuple[str, str]]) -> None:
        for cell_id, orientation_label in pairs:
            add_case(person, cell_id, coarse_pose, orientation_label)

    def add_person_plan(person: str) -> None:
        add_cell_orientations(person, INTERIOR_3X3_CELLS, "Standing", CARDINAL_ORIENTATIONS)
        for cell_id in PERIMETER_CELLS:
            add_cell_orientations(person, [cell_id], "Standing", PERIMETER_IMPORTANT_DIRECTIONS[cell_id])
        add_cell_orientations(person, CENTER_AND_INTERIOR_CORNERS, "Standing", DIAGONAL_FOCUS_DIRECTIONS)
        add_empty_checkpoint()

        add_cell_orientations(person, INTERIOR_3X3_CELLS, "Sitting", CARDINAL_ORIENTATIONS)
        add_cell_orientation_pairs(person, "Sitting", PERIMETER_INWARD_DIRECTIONS)
        add_cell_orientations(person, CENTER_AND_INTERIOR_CORNERS, "Sitting", DIAGONAL_FOCUS_DIRECTIONS)
        add_empty_checkpoint()

        add_cell_orientations(person, CROSS_CELLS, "Arms_Out", CARDINAL_ORIENTATIONS)
        add_cell_orientation_pairs(person, "Arms_Out", INTERIOR_CORNER_INWARD_DIRECTIONS)
        add_cell_orientations(person, ["C13"], "Arms_Out", DIAGONAL_FOCUS_DIRECTIONS)

        add_cell_orientations(person, CROSS_CELLS, "Lying", CARDINAL_ORIENTATIONS)
        add_cell_orientation_pairs(person, "Lying", INTERIOR_CORNER_INWARD_DIRECTIONS)
        add_cell_orientation_pairs(person, "Lying", EDGE_CENTER_INWARD_DIRECTIONS)
        add_cell_orientations(person, LYING_DIAGONAL_CELLS, "Lying", DIAGONAL_FOCUS_DIRECTIONS)

        add_cell_orientations(person, CROSS_CELLS, "Walk_in_place", CARDINAL_ORIENTATIONS)
        add_cell_orientation_pairs(person, "Walk_in_place", INTERIOR_CORNER_INWARD_DIRECTIONS)
        add_cell_orientation_pairs(person, "Walk_in_place", EDGE_CENTER_INWARD_DIRECTIONS)

        add_cell_orientations(person, CROSS_CELLS, "Transition", TRANSITION_AXIS_DIRECTIONS)
        add_cell_orientation_pairs(person, "Transition", TRANSITION_INTERIOR_CORNER_DIRECTIONS)
        add_cell_orientation_pairs(person, "Transition", EDGE_CENTER_INWARD_DIRECTIONS)
        add_empty_checkpoint()

    add_empty_checkpoint()
    for person in PERSONS:
        add_person_plan(person)

    return cases

def collection_plan_summary() -> Dict[str, Any]:
    sample_count = len(planned_cases())
    return {
        "plan_id": COLLECTION_PLAN_ID,
        "mode": COLLECTION_MODE,
        "expected_samples": sample_count,
        "expected_clean_rows": sample_count * MIN_SYNCED_FRAMES,
        "expected_sample_count_constant": PART1_EXPECTED_SAMPLE_COUNT,
        "expected_clean_rows_constant": PART1_EXPECTED_CLEAN_ROWS,
        "empty_baseline_samples": EMPTY_BASELINE_SAMPLES,
        "empty_checkpoints": f"{EMPTY_CHECKPOINT_COUNT} checkpoints x {EMPTY_TRIALS_PER_CHECKPOINT} trials",
        "empty_identifiers": f"{EMPTY_PERSON}/{EMPTY_CELL_ID} identifier-only; no actor in grid",
        "standing": "P1-P5 each: interior 3x3 x 4 cardinal + perimeter 16 x 2 important + center diagonals = 78; total 390",
        "sitting": "P1-P5 each: interior 3x3 x 4 cardinal + perimeter 16 x 1 inward + center diagonals = 62; total 310",
        "arms_out": "P1-P5 each: cross5 x 4 cardinal + interior-corner inward pairs + center diagonals = 30; total 150",
        "lying": "P1-P5 each: cross5 x 4 cardinal + interior-corner inward pairs + edge-center inward + C13/C08/C18 diagonals = 38; total 190",
        "walk_in_place": "P1-P5 each: cross5 x 4 cardinal + interior-corner inward pairs + edge-center inward = 32; total 160",
        "transition": "P1-P5 each: cross5 x 2 axis directions + interior-corner inward + edge-center inward = 18; total 90",
        "human_samples": PART1_HUMAN_SAMPLE_COUNT,
    }

def count_cases_by(index: int) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for case in planned_cases():
        key = str(case[index])
        counts[key] += 1
    return dict(sorted(counts.items()))

def validate_plan_or_raise() -> None:
    cases = planned_cases()
    ids = [sample_id(*case) for case in cases]
    duplicates = len(ids) - len(set(ids))
    empty_cases = [case for case in cases if case[2] == "Empty"]
    invalid_empty_cases = [case for case in empty_cases if case[:4] != (EMPTY_PERSON, EMPTY_CELL_ID, "Empty", "Empty")]
    if COLLECTION_MODE == "part1_large" and len(cases) != PART1_EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"part1_large planned case count must be {PART1_EXPECTED_SAMPLE_COUNT}, got {len(cases)}")
    if COLLECTION_MODE == "part1_large" and len(empty_cases) != EMPTY_BASELINE_SAMPLES:
        raise RuntimeError(f"part1_large Empty sample count must be {EMPTY_BASELINE_SAMPLES}, got {len(empty_cases)}")
    if COLLECTION_MODE == "part1_large":
        expected_empty_trials = list(range(1, EMPTY_BASELINE_SAMPLES + 1))
        actual_empty_trials = [case[4] for case in empty_cases]
        if actual_empty_trials != expected_empty_trials:
            raise RuntimeError(f"part1_large Empty trials must be T001..T{EMPTY_BASELINE_SAMPLES:03d}, got {actual_empty_trials}")
        pose_counts = defaultdict(int)
        for case in cases:
            pose_counts[case[2]] += 1
        for pose, expected_count in EXPECTED_POSE_COUNTS.items():
            if pose_counts[pose] != expected_count:
                raise RuntimeError(f"part1_large {pose} sample count must be {expected_count}, got {pose_counts[pose]}")
        person_pose_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for person, _cell_id, pose, _orientation_label, _trial in cases:
            if person in PERSONS:
                person_pose_counts[(person, pose)] += 1
        for person in PERSONS:
            person_total = sum(person_pose_counts[(person, pose)] for pose in EXPECTED_PERSON_POSE_COUNTS)
            if person_total != PART1_HUMAN_SAMPLE_COUNT // len(PERSONS):
                raise RuntimeError(f"part1_large {person} human sample count must be 258, got {person_total}")
            for pose, expected_count in EXPECTED_PERSON_POSE_COUNTS.items():
                actual_count = person_pose_counts[(person, pose)]
                if actual_count != expected_count:
                    raise RuntimeError(f"part1_large {person}/{pose} sample count must be {expected_count}, got {actual_count}")
        if sum(pose_counts[pose] for pose in EXPECTED_PERSON_POSE_COUNTS) != PART1_HUMAN_SAMPLE_COUNT:
            raise RuntimeError(f"part1_large human sample count must be {PART1_HUMAN_SAMPLE_COUNT}")
    if invalid_empty_cases:
        raise RuntimeError(f"Empty cases must use {EMPTY_PERSON}/{EMPTY_CELL_ID}/Empty/Empty identifiers")
    if duplicates:
        raise RuntimeError(f"Collection plan has {duplicates} duplicate sample IDs")
    if EXPECTED_DATA_COLUMNS != 438:
        raise RuntimeError(f"Frame CSV column count must be 438, got {EXPECTED_DATA_COLUMNS}")
    if CSI_VALUES_PER_RX != 128:
        raise RuntimeError(f"CSI values per RX must be 128, got {CSI_VALUES_PER_RX}")
    if len(RX_ORDER) != 3:
        raise RuntimeError(f"RX count must be 3, got {len(RX_ORDER)}")

def print_plan_only() -> None:
    validate_plan_or_raise()
    cases = planned_cases()
    print("COLLECTION PLAN")
    print(f"Session: {SESSION_ID}")
    print(f"Layout ID: {LAYOUT_ID}")
    print(f"Layout type: {LAYOUT_TYPE}")
    print(f"Collection mode: {COLLECTION_MODE}")
    print(f"Collection plan ID: {COLLECTION_PLAN_ID}")
    print(f"Planned samples: {len(cases)}")
    print(f"Expected clean rows: {len(cases) * MIN_SYNCED_FRAMES}")
    if COLLECTION_MODE == "part1_large":
        print(f"Optimized plan target: {PART1_EXPECTED_SAMPLE_COUNT} samples")
        print(f"Empty checkpoints: {EMPTY_CHECKPOINT_COUNT} x {EMPTY_TRIALS_PER_CHECKPOINT} trials = {EMPTY_BASELINE_SAMPLES}")
        print("Human plan: 1290 samples = 258 per person across Standing/Sitting/Arms_Out/Lying/Walk_in_place/Transition")
    print(f"Data columns: {EXPECTED_DATA_COLUMNS}")
    print(f"CSI serial schema: {CSI_SERIAL_SCHEMA}")
    print(f"CSI values per RX: {CSI_VALUES_PER_RX}")
    print(f"RX count: {len(RX_ORDER)}")
    print(f"Empty identifiers: {EMPTY_PERSON}/{EMPTY_CELL_ID} (identifier-only, no actor in grid)")
    print(f"Target frame rate: {TARGET_FRAME_RATE} Hz")
    print(f"Counts by pose: {count_cases_by(2)}")
    print(f"Counts by person: {count_cases_by(0)}")
    print(f"Counts by cell: {count_cases_by(1)}")
    print(f"Counts by orientation: {count_cases_by(3)}")
    print(f"Collection plan summary: {collection_plan_summary()}")

def instruction_for_case(cell_id: str, coarse_pose: str, orientation_label: str) -> str:
    metadata = label_metadata(cell_id, coarse_pose, orientation_label)
    if coarse_pose == "Empty":
        return "Leave the room empty; no actor in the capture area."
    details = (
        f"Go to {cell_id} center ({metadata['center_x_m']}m, {metadata['center_y_m']}m), "
        f"pose={coarse_pose}, orientation={orientation_label} ({metadata['orientation_deg']} deg). "
        f"Direction: {ORIENTATION_DESCRIPTIONS[orientation_label]}"
    )
    if coarse_pose in {"Lying", "Transition"}:
        details += f" Footprint-aware label: occupied_cells={metadata['occupied_cells']}. Keep the body footprint inside these cells as much as practical."
    if coarse_pose == "Transition":
        details += f" Controlled transition pattern: {TRANSITION_PATTERN}."
    return details

def rx_csi_headers(rx: str) -> List[str]:
    headers = [
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
    headers.extend([f"{rx}_csi_{i:03d}" for i in range(CSI_VALUES_PER_RX)])
    return headers

def base_label_headers() -> List[str]:
    return [
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
    ]

def data_headers() -> List[str]:
    headers = base_label_headers()
    for rx in RX_ORDER:
        headers.extend(rx_csi_headers(rx))
    return headers

def create_session_files() -> None:
    path = session_dir()
    path.mkdir(parents=True, exist_ok=True)

    session_md = path / "session.md"
    if not session_md.exists():
        session_md.write_text(
            f"# {SESSION_ID}\n\n"
            "## Manual checklist before recording\n\n"
            f"- Operator: {OPERATOR}\n"
            f"- Room: {ROOM_NAME}\n"
            f"- Room size: {ROOM_SIZE_M}\n"
            f"- Channel: {CHANNEL}\n"
            f"- Layout ID: {LAYOUT_ID}\n"
            f"- Layout type: {LAYOUT_TYPE}\n"
            f"- Collection plan ID: {COLLECTION_PLAN_ID}\n"
            f"- Calibration ID: {CALIBRATION_ID}\n"
            f"- COM RX1: {COM_PORTS['RX1']}\n"
            f"- COM RX2: {COM_PORTS['RX2']}\n"
            f"- COM RX3: {COM_PORTS['RX3']}\n"
            f"- Label schema: {LABEL_SCHEMA_VERSION}\n"
            f"- Collection mode: {COLLECTION_MODE}\n"
            f"- Stage description: {ACTIVE_STAGE_CONFIG['description'] if ACTIVE_STAGE_CONFIG else COLLECTION_STAGE_NOTES[COLLECTION_MODE]}\n"
            f"- Collection persons: {COLLECTION_PERSONS}\n"
            f"- Output person files: {OUTPUT_PERSONS}\n"
            f"- Empty identifiers: {EMPTY_PERSON}/{EMPTY_CELL_ID} (identifier-only; no actor in grid)\n"
            f"- Collection cells: {COLLECTION_CELLS}\n"
            f"- Collection poses: {COLLECTION_POSES}\n"
            f"- Collection orientations: {COLLECTION_ORIENTATIONS}\n"
            f"- Trials per case: {TRIALS_PER_CASE if ACTIVE_STAGE_CONFIG else 'explicit per planned case'}\n"
            f"- Planned samples: {len(planned_cases())}\n"
            f"- Expected clean rows: {len(planned_cases()) * MIN_SYNCED_FRAMES}\n"
            f"- Transition pattern: {TRANSITION_PATTERN}\n"
            f"- Cell grid: {CELL_GRID_COLS} cols x {CELL_GRID_ROWS} rows\n"
            f"- Cell size: {CELL_WIDTH_M:.3f}m x {CELL_DEPTH_M:.3f}m\n"
            f"- Coordinate convention: {COORDINATE_CONVENTION}\n"
            f"- Layout notes: {LAYOUT_DESCRIPTION}\n"
            f"- TX position: {TX_POSITION_M}\n"
            f"- RX positions: {RX_POSITIONS_M}\n"
            f"- Node side roles: {NODE_SIDE_ROLES}\n"
            f"- RX order note: {RX_ORDER_NOTE}\n"
            f"- Orientation definitions: {ORIENTATION_DESCRIPTIONS}\n"
            f"- Collection plan summary: {collection_plan_summary()}\n\n"
            "## Notes\n\n"
            "- Update channel in TX/RX firmware before flashing.\n"
            "- Update RX_ID in RX firmware before flashing RX1/RX2/RX3.\n"
            "- Update TX_MAC_FILTER in RX firmware from the TX Serial Monitor TX_MAC line before flashing RX1/RX2/RX3.\n"
            "- Record calibration before main samples.\n"
            "- Measure 25-zone grid, footprint labels, and TX/RX positions before collection.\n",
            encoding="utf-8",
        )

    calibration_json = path / "calibration.json"
    if not calibration_json.exists():
        calibration_json.write_text(
            json.dumps(
                {
                    "session_id": SESSION_ID,
                    "calibration_id": CALIBRATION_ID,
                    "room_name": ROOM_NAME,
                    "room_size_m": ROOM_SIZE_M,
                    "layout_id": LAYOUT_ID,
                    "layout_type": LAYOUT_TYPE,
                    "collection_plan_id": COLLECTION_PLAN_ID,
                    "channel": CHANNEL,
                    "expected_tx_id": EXPECTED_TX_ID,
                    "com_ports": COM_PORTS,
                    "label_schema_version": LABEL_SCHEMA_VERSION,
                    "collection_mode": COLLECTION_MODE,
                    "collection_stage": ACTIVE_STAGE_CONFIG,
                    "collection_plan_summary": collection_plan_summary(),
                    "planned_sample_count": len(planned_cases()),
                    "expected_clean_rows": len(planned_cases()) * MIN_SYNCED_FRAMES,
                    "collection_stage_notes": COLLECTION_STAGE_NOTES,
                    "collection_persons": COLLECTION_PERSONS,
                    "output_persons": OUTPUT_PERSONS,
                    "empty_person": EMPTY_PERSON,
                    "empty_cell_id": EMPTY_CELL_ID,
                    "collection_cells": COLLECTION_CELLS,
                    "collection_poses": COLLECTION_POSES,
                    "collection_orientations": COLLECTION_ORIENTATIONS,
                    "trials_per_case": TRIALS_PER_CASE,
                    "transition_pattern": TRANSITION_PATTERN,
                    "coordinate_convention": COORDINATE_CONVENTION,
                    "cell_grid_rows": CELL_GRID_ROWS,
                    "cell_grid_cols": CELL_GRID_COLS,
                    "cell_width_m": CELL_WIDTH_M,
                    "cell_depth_m": CELL_DEPTH_M,
                    "cell_origin_m": {"x": CELL_ORIGIN_X_M, "y": CELL_ORIGIN_Y_M},
                    "cells": cell_layout(),
                    "coarse_poses": COARSE_POSES,
                    "orientation_labels": ORIENTATION_LABELS,
                    "orientation_degrees": ORIENTATION_DEGREES,
                    "orientation_descriptions": ORIENTATION_DESCRIPTIONS,
                    "height_bands": HEIGHT_BANDS,
                    "label_qualities": LABEL_QUALITIES,
                    "pose_defaults": POSE_DEFAULTS,
                    "tx_position_m": TX_POSITION_M,
                    "rx_positions_m": RX_POSITIONS_M,
                    "node_side_roles": NODE_SIDE_ROLES,
                    "rx_order_note": RX_ORDER_NOTE,
                    "record_seconds": RECORD_SECONDS,
                    "target_frame_rate": TARGET_FRAME_RATE,
                    "csi_values_per_rx": CSI_VALUES_PER_RX,
                    "data_columns": EXPECTED_DATA_COLUMNS,
                    "csi_serial_schema": CSI_SERIAL_SCHEMA,
                    "created_at": now_iso(),
                    "layout_notes": LAYOUT_DESCRIPTION,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        validate_existing_session_metadata(calibration_json)

    ensure_csv(path / "manifest.csv", manifest_headers())
    ensure_csv(path / "quality.csv", quality_headers())
    for person in OUTPUT_PERSONS:
        ensure_csv(data_file_for_person(person), data_headers())

def validate_existing_session_metadata(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read existing calibration.json: {exc}") from exc

    expected = {
        "session_id": SESSION_ID,
        "calibration_id": CALIBRATION_ID,
        "room_name": ROOM_NAME,
        "room_size_m": ROOM_SIZE_M,
        "layout_id": LAYOUT_ID,
        "layout_type": LAYOUT_TYPE,
        "collection_plan_id": COLLECTION_PLAN_ID,
        "channel": CHANNEL,
        "expected_tx_id": EXPECTED_TX_ID,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "collection_mode": COLLECTION_MODE,
        "collection_persons": COLLECTION_PERSONS,
        "output_persons": OUTPUT_PERSONS,
        "empty_person": EMPTY_PERSON,
        "empty_cell_id": EMPTY_CELL_ID,
        "collection_cells": COLLECTION_CELLS,
        "collection_poses": COLLECTION_POSES,
        "collection_orientations": COLLECTION_ORIENTATIONS,
        "trials_per_case": TRIALS_PER_CASE,
        "orientation_degrees": ORIENTATION_DEGREES,
        "orientation_descriptions": ORIENTATION_DESCRIPTIONS,
        "transition_pattern": TRANSITION_PATTERN,
        "node_side_roles": NODE_SIDE_ROLES,
        "rx_order_note": RX_ORDER_NOTE,
        "collection_plan_summary": collection_plan_summary(),
        "planned_sample_count": len(planned_cases()),
        "expected_clean_rows": len(planned_cases()) * MIN_SYNCED_FRAMES,
        "cell_grid_rows": CELL_GRID_ROWS,
        "cell_grid_cols": CELL_GRID_COLS,
        "cell_width_m": CELL_WIDTH_M,
        "cell_depth_m": CELL_DEPTH_M,
        "cell_origin_m": {"x": CELL_ORIGIN_X_M, "y": CELL_ORIGIN_Y_M},
        "tx_position_m": TX_POSITION_M,
        "rx_positions_m": RX_POSITIONS_M,
        "com_ports": COM_PORTS,
        "record_seconds": RECORD_SECONDS,
        "target_frame_rate": TARGET_FRAME_RATE,
        "csi_values_per_rx": CSI_VALUES_PER_RX,
        "data_columns": EXPECTED_DATA_COLUMNS,
        "csi_serial_schema": CSI_SERIAL_SCHEMA,
    }
    mismatches = []
    for key, value in expected.items():
        if data.get(key) != value:
            mismatches.append(f"{key}: existing={data.get(key)!r}, current={value!r}")
    if mismatches:
        raise RuntimeError(
            "Existing session folder has different metadata. Use a new SESSION_ID or restore matching config. "
            + "; ".join(mismatches)
        )

def ensure_csv(path: Path, headers: List[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as file:
            existing = next(csv.reader(file), [])
        if existing != headers:
            raise RuntimeError(
                f"CSV header mismatch in {path}. Use a new SESSION_ID or migrate/delete the old file."
            )
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(headers)

def manifest_headers() -> List[str]:
    return [
        "sample_id",
        "session_id",
        "calibration_id",
        "layout_id",
        "channel",
        "person",
        "presence",
        "cell_id",
        "center_x_m",
        "center_y_m",
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
        "data_file",
        "start_seq",
        "end_seq",
        "start_time_ms",
        "end_time_ms",
        "frames",
        "status",
        "created_at",
    ]

def quality_headers() -> List[str]:
    return [
        "sample_id",
        "status",
        "total_synced_frames",
        "clean_frames",
        "rx1_frames",
        "rx2_frames",
        "rx3_frames",
        "duplicate_lines",
        "malformed_lines",
        "missing_seq_max_gap",
        "rssi_min",
        "rssi_mean",
        "rssi_max",
        "reason",
        "created_at",
    ]

def parse_csi_line(line: str) -> Optional[CsiPayload]:
    if len(line) > 2000:
        return None
    parts = line.strip().split(",")
    if len(parts) != EXPECTED_PART_COUNT or parts[0] != "CSI_V2":
        return None
    try:
        rx_id_num = int(parts[1])
        rx_id = f"RX{rx_id_num}"
        if rx_id not in RX_ORDER:
            return None
        csi_len = int(parts[11])
        csi_values = [int(value) for value in parts[CSI_METADATA_PARTS:]]
        if csi_len != CSI_VALUES_PER_RX or len(csi_values) != CSI_VALUES_PER_RX:
            return None
        if any(value < -128 or value > 127 for value in csi_values):
            return None
        return {
            "rx": rx_id,
            "tx_id": int(parts[2]),
            "seq": int(parts[3]),
            "mac_time": int(parts[4]),
            "rssi": int(parts[5]),
            "channel": int(parts[6]),
            "rate": int(parts[7]),
            "cwb": int(parts[8]),
            "rx_state": int(parts[9]),
            "first_word_invalid": int(parts[10]),
            "csi_len": csi_len,
            "src_mac": parts[12],
            "csi": csi_values,
        }
    except ValueError:
        return None

def parse_stat_line(line: str) -> Optional[Dict[str, int]]:
    parts = line.strip().split(",")
    if len(parts) < 3 or parts[0] != "STAT":
        return None
    values = {}
    for index in range(1, len(parts) - 1, 2):
        key = parts[index]
        value = parts[index + 1]
        try:
            values[key] = int(value)
        except ValueError:
            values[key] = value
    rx_id = values.get("rx_id")
    if not isinstance(rx_id, int):
        return None
    values["rx"] = f"RX{rx_id}"
    return values

def parse_config_rx_id(line: str) -> Optional[str]:
    parts = line.strip().split(",")
    for index in range(1, len(parts) - 1, 2):
        if parts[index] == "rx_id":
            try:
                return f"RX{int(parts[index + 1])}"
            except ValueError:
                return None
    return None

def read_serial_port(rx_name: str, port: str) -> None:
    global thread_buffers
    try:
        serial_module = serial
        if serial_module is None:
            print("ERROR: Missing dependency 'pyserial'.")
            return
        
        ser = serial_module.Serial(port, 921600, timeout=0.1) 
        
        set_buffer_size = getattr(ser, "set_buffer_size", None)
        if callable(set_buffer_size):
            set_buffer_size(rx_size=1048576, tx_size=1048576)
            
        ser.reset_input_buffer()
        serial_handles[rx_name] = ser
        print(f"[OK] {rx_name} opened on {port}")
    except Exception as exc:
        print(f"[ERROR] Cannot open {rx_name} on {port}: {exc}")
        return

    raw_buffer = ""

    while not stop_threads:
        try:
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(waiting).decode("utf-8", errors="ignore")
                raw_buffer += chunk
                
                if '\n' in raw_buffer:
                    lines = raw_buffer.split('\n')
                    raw_buffer = lines.pop() 
                    
                    for raw_line in lines:
                        line = raw_line.strip()
                        if not line: continue
                        
                        with buffer_lock:
                            rx_total_lines[rx_name] += 1
                        
                        if line.startswith("STAT,"):
                            stat = parse_stat_line(line)
                            if stat is not None and stat.get("rx") == rx_name:
                                with stats_lock:
                                    rx_stat_values[rx_name] = stat
                            elif stat is not None:
                                with stats_lock:
                                    rx_wrong_stat_values[rx_name] = stat
                            continue
                        
                        if line.startswith("CONFIG,"):
                            config_rx = parse_config_rx_id(line)
                            if config_rx is not None and config_rx != rx_name:
                                with stats_lock:
                                    rx_wrong_stat_values[rx_name] = {"rx": config_rx, "rx_id": int(config_rx[2:])}
                            continue

                        if line.startswith("RX_") or line.startswith("["):
                            continue
                        
                        parsed = parse_csi_line(line)
                        if parsed is not None and parsed["rx"] != rx_name:
                            with buffer_lock:
                                rx_wrong_id_lines[rx_name] += 1
                            continue
                        
                        if parsed is None:
                            with buffer_lock:
                                rx_malformed_lines[rx_name] += 1
                            continue

                        host_ts_ms = int(time.time() * 1000)
                        parsed["host_ts_ms"] = host_ts_ms

                        with buffer_lock:
                            rx_valid_lines[rx_name] += 1
                            if parsed["first_word_invalid"] != 0:
                                rx_first_word_invalid_lines[rx_name] += 1
                            rx_last_seen[rx_name] = host_ts_ms / 1000.0
                            rx_last_channel[rx_name] = parsed["channel"]
                            
                            # CƠ CHẾ BẢO VỆ 1: Lọc bỏ tín hiệu rác từ thiết bị ngoại lai và Stale Data trong RAM
                            if recording and parsed["rssi"] >= -85:
                                seq = parsed["seq"]
                                if seq in recording_seen_seqs[rx_name]:
                                    rx_duplicate_lines[rx_name] += 1
                                    continue
                                recording_seen_seqs[rx_name].add(seq)
                                thread_buffers[rx_name].append(parsed)
            else:
                time.sleep(0.001)
                    
        except Exception:
            time.sleep(0.01)

def start_reader_threads() -> None:
    for rx_name, port in COM_PORTS.items():
        thread = threading.Thread(target=read_serial_port, args=(rx_name, port), daemon=True)
        thread.start()

def preflight_check(seconds: int = 5) -> bool:
    print("\n=== PREFLIGHT CHECK ===")
    print(f"Checking RX streams for {seconds}s...")
    print(f"Target: {TARGET_FRAME_RATE} Hz, preflight minimum: {MIN_RX_RATE_HZ:.0f} Hz, clean synced frames: {MIN_SYNCED_FRAMES}")
    with buffer_lock:
        before = {rx: rx_valid_lines[rx] for rx in RX_ORDER}
        malformed_before = {rx: rx_malformed_lines[rx] for rx in RX_ORDER}
        first_word_invalid_before = {rx: rx_first_word_invalid_lines[rx] for rx in RX_ORDER}
        wrong_id_before = {rx: rx_wrong_id_lines[rx] for rx in RX_ORDER}
    with stats_lock:
        stat_before = {rx: dict(rx_stat_values[rx]) for rx in RX_ORDER}
    time.sleep(seconds)
    with buffer_lock:
        after = {rx: rx_valid_lines[rx] for rx in RX_ORDER}
        malformed_after = {rx: rx_malformed_lines[rx] for rx in RX_ORDER}
        first_word_invalid_after = {rx: rx_first_word_invalid_lines[rx] for rx in RX_ORDER}
        wrong_id_after = {rx: rx_wrong_id_lines[rx] for rx in RX_ORDER}
        last_channel = {rx: rx_last_channel[rx] for rx in RX_ORDER}
    with stats_lock:
        stat_after = {rx: dict(rx_stat_values[rx]) for rx in RX_ORDER}
        wrong_stat_after = {rx: dict(rx_wrong_stat_values[rx]) for rx in RX_ORDER}

    ok = True
    for rx in RX_ORDER:
        count = after[rx] - before[rx]
        rate = count / max(seconds, 1)
        malformed = malformed_after[rx] - malformed_before[rx]
        first_word_invalid = first_word_invalid_after[rx] - first_word_invalid_before[rx]
        wrong_id = wrong_id_after[rx] - wrong_id_before[rx]
        print(f"{rx}: {count} valid lines, approx {rate:.1f} Hz, malformed={malformed}, wrong_rx_id={wrong_id}, first_word_invalid={first_word_invalid}")
        wrong_stat = wrong_stat_after[rx]
        if wrong_stat:
            print(f"[FAIL] {rx} COM mapping mismatch: port {COM_PORTS[rx]} reports {wrong_stat.get('rx')}, expected {rx}")
            ok = False
        stat = stat_after[rx]
        if stat:
            previous = stat_before[rx]
            details = []
            stat_deltas = {}
            for key in [
                "csi_cb", "csi_seen", "valid", "espnow_recv", "espnow_valid", "seq_gap",
                "csi_null", "csi_mac_mismatch", "no_csi", "output_busy", "bad_payload",
                "bad_tx_id", "bad_channel", "bad_checksum", "bad_csi_len", "mac_mismatch",
            ]:
                if isinstance(stat.get(key), int):
                    delta = stat[key] - previous.get(key, stat[key])
                    stat_deltas[key] = delta
                    details.append(f"{key}+{delta}")
            if "last_csi_len" in stat:
                details.append(f"last_csi_len={stat['last_csi_len']}")
            if "last_seq" in stat:
                details.append(f"last_seq={stat['last_seq']}")
            if "queue" in stat:
                details.append(f"queue={stat['queue']}")
            if details:
                print(f"  firmware: {', '.join(details)}")
            hard_bad = sum(stat_deltas.get(key, 0) for key in [
                "csi_null", "csi_mac_mismatch", "bad_payload", "bad_tx_id",
                "bad_channel", "bad_checksum", "bad_csi_len", "mac_mismatch",
            ])
            soft_drop = stat_deltas.get("no_csi", 0) + stat_deltas.get("output_busy", 0)
            if hard_bad > 0:
                print(f"[FAIL] {rx} firmware rejected invalid packets during preflight")
                ok = False
            if soft_drop > MAX_PREFLIGHT_SOFT_DROPS:
                print(f"[FAIL] {rx} firmware dropped {soft_drop} usable packets during preflight")
                ok = False
        if rate < MIN_RX_RATE_HZ:
            print(f"[FAIL] {rx} rate below {MIN_RX_RATE_HZ:.0f} Hz")
            ok = False
        if malformed > 0:
            print(f"[FAIL] {rx} has malformed CSI lines")
            ok = False
        if wrong_id > 0:
            print(f"[FAIL] {rx} has CSI lines from another RX_ID")
            ok = False
        if first_word_invalid > 0:
            print(f"[FAIL] {rx} has CSI lines with first_word_invalid != 0")
            ok = False
        if last_channel[rx] != CHANNEL:
            print(f"[FAIL] {rx} channel mismatch: got {last_channel[rx]}, expected {CHANNEL}")
            ok = False
    if not ok:
        print("[FAIL] One or more RX streams failed rate, malformed-line, first-word-invalid, firmware, or channel checks.")
        return False
    print("[OK] All RX streams are alive. Continue only if COM mapping is correct.")
    return True

def synced_frames() -> List[SyncedFrame]:
    temp_dict = defaultdict(dict)
    
    with buffer_lock:
        buffer_snapshot = {rx: list(thread_buffers[rx]) for rx in RX_ORDER}
        
    for rx in RX_ORDER:
        for frame in buffer_snapshot[rx]:
            seq_key = frame["seq"]
            temp_dict[seq_key][rx] = frame
            
    frames = []
    for seq in sorted(temp_dict.keys()):
        if all(rx in temp_dict[seq] for rx in RX_ORDER):
            frames.append((seq, dict(temp_dict[seq])))
            
    return frames

def frame_counts() -> Dict[str, int]:
    with buffer_lock:
        return {rx: len(thread_buffers[rx]) for rx in RX_ORDER}

def recording_error_counts() -> Tuple[int, int]:
    with buffer_lock:
        duplicates = sum(rx_duplicate_lines[rx] for rx in RX_ORDER)
        malformed = sum(rx_malformed_lines[rx] for rx in RX_ORDER)
    return duplicates, malformed

def get_max_consecutive_missing_seq(frames: List[SyncedFrame]) -> int:
    """
    Quét qua các frame đã đồng bộ, tìm khoảng đứt gãy (gap) lớn nhất.
    Trả về số lượng gói tin bị mất liên tiếp nhiều nhất.
    """
    if len(frames) < 2:
        return 0
    max_gap = 0
    seqs = [seq for seq, _ in frames]
    for i in range(1, len(seqs)):
        gap = seqs[i] - seqs[i-1] - 1
        if gap > max_gap:
            max_gap = gap
    return max_gap

def rssi_stats(frames: List[SyncedFrame]) -> Tuple[str, str, str]:
    values: List[int] = []
    for _, frame in frames:
        for rx in RX_ORDER:
            values.append(frame[rx]["rssi"])
    if not values:
        return "", "", ""
    return str(min(values)), f"{sum(values) / len(values):.2f}", str(max(values))

def count_first_word_invalid_clean_frames(frames: List[SyncedFrame]) -> Dict[str, int]:
    counts = {rx: 0 for rx in RX_ORDER}
    for _, frame in frames:
        for rx in RX_ORDER:
            if frame[rx]["first_word_invalid"] != 0:
                counts[rx] += 1
    return counts

def first_word_invalid_reason(counts: Dict[str, int]) -> str:
    parts = [f"{rx}_{counts[rx]}" for rx in RX_ORDER if counts.get(rx, 0) > 0]
    if not parts:
        return ""
    return "first_word_invalid_" + "_".join(parts)

def load_passed_sample_ids() -> set[str]:
    path = session_dir() / "manifest.csv"
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") == "pass" and row.get("sample_id"):
                ids.add(row["sample_id"])
    return ids

def flush_serial_buffers() -> None:
    for rx in RX_ORDER:
        ser = serial_handles.get(rx)
        if ser is None:
            continue
        try:
            ser.reset_input_buffer()
        except Exception as exc:
            print(f"[WARN] Cannot flush {rx} serial buffer: {exc}")

def append_csv(path: Path, row: List[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)

def append_quality(sid: str, status: str, total_synced: int, clean_count: int,
                   counts: Dict[str, int], duplicates: int, malformed: int,
                   max_gap: int, frames: List[SyncedFrame], reason: str) -> None:
    rssi_min, rssi_mean, rssi_max = rssi_stats(frames)
    append_csv(
        session_dir() / "quality.csv",
        [
            sid,
            status,
            total_synced,
            clean_count,
            counts["RX1"],
            counts["RX2"],
            counts["RX3"],
            duplicates,
            malformed,
            max_gap,
            rssi_min,
            rssi_mean,
            rssi_max,
            reason,
            now_iso(),
        ],
    )

def append_manifest(sid: str, person: str, cell_id: str, coarse_pose: str, orientation_label: str,
                    trial: int, clean_frames: List[SyncedFrame]) -> None:
    start_seq = clean_frames[0][0]
    end_seq = clean_frames[-1][0]
    start_time_ms = min(frame[rx]["host_ts_ms"] for _, frame in clean_frames for rx in RX_ORDER)
    end_time_ms = max(frame[rx]["host_ts_ms"] for _, frame in clean_frames for rx in RX_ORDER)
    data_file = data_file_for_person(person).name
    labels = label_metadata(cell_id, coarse_pose, orientation_label)
    append_csv(
        session_dir() / "manifest.csv",
        [
            sid,
            SESSION_ID,
            CALIBRATION_ID,
            LAYOUT_ID,
            CHANNEL,
            person,
            labels["presence"],
            labels["cell_id"],
            labels["center_x_m"],
            labels["center_y_m"],
            labels["occupied_cells"],
            labels["coarse_pose"],
            labels["orientation_label"],
            labels["orientation_deg"],
            labels["height_band"],
            labels["footprint_length_m"],
            labels["footprint_width_m"],
            labels["footprint_yaw_deg"],
            labels["label_quality"],
            trial,
            data_file,
            start_seq,
            end_seq,
            start_time_ms,
            end_time_ms,
            len(clean_frames),
            "pass",
            now_iso(),
        ],
    )

def write_data_rows(sid: str, person: str, cell_id: str, coarse_pose: str, orientation_label: str,
                    trial: int, clean_frames: List[SyncedFrame]) -> None:
    path = data_file_for_person(person)
    labels = label_metadata(cell_id, coarse_pose, orientation_label)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for frame_idx, (seq, frame) in enumerate(clean_frames):
            row: List[object] = [
                SESSION_ID,
                CALIBRATION_ID,
                LAYOUT_ID,
                CHANNEL,
                sid,
                person,
                labels["presence"],
                labels["cell_id"],
                labels["cell_row"],
                labels["cell_col"],
                labels["center_x_m"],
                labels["center_y_m"],
                labels["cell_width_m"],
                labels["cell_depth_m"],
                labels["occupied_cells"],
                labels["coarse_pose"],
                labels["orientation_label"],
                labels["orientation_deg"],
                labels["height_band"],
                labels["footprint_length_m"],
                labels["footprint_width_m"],
                labels["footprint_yaw_deg"],
                labels["label_quality"],
                trial,
                frame_idx,
                seq,
                min(frame[rx]["host_ts_ms"] for rx in RX_ORDER),
            ]
            for rx in RX_ORDER:
                payload = frame[rx]
                row.extend([
                    payload["mac_time"],
                    payload["rssi"],
                    payload["channel"],
                    payload["rate"],
                    payload["cwb"],
                    payload["rx_state"],
                    payload["first_word_invalid"],
                    payload["csi_len"],
                    payload["src_mac"],
                ])
                row.extend(payload["csi"])
            writer.writerow(row)

def capture_one_sample(person: str, cell_id: str, coarse_pose: str, orientation_label: str, trial: int) -> bool:
    global recording

    sid = sample_id(person, cell_id, coarse_pose, orientation_label, trial)
    if sid in passed_sample_ids:
        print(f"[SKIP] {sid} already exists in manifest.csv")
        return True
    print("\n" + "=" * 72)
    print(f"NEXT SAMPLE: {sid}")
    print(
        f"PERSON: {person} | CELL: {cell_id} | POSE: {coarse_pose} | "
        f"ORIENTATION: {orientation_label} | TRIAL: {trial}"
    )
    print("Actor instruction: " + instruction_for_case(cell_id, coarse_pose, orientation_label))
    print("=" * 72)
    input("Press ENTER to start 5s recording...")

    with buffer_lock:
        for rx in RX_ORDER:
            thread_buffers[rx].clear()
            recording_seen_seqs[rx].clear()
            rx_duplicate_lines[rx] = 0
            rx_malformed_lines[rx] = 0

    print("Recording starts in:")
    for value in [3, 2, 1]:
        print(value)
        time.sleep(1)

    flush_serial_buffers()

    print("RECORDING... keep pose/orientation stable unless label is Transition.")
    recording = True
    time.sleep(RECORD_SECONDS)
    recording = False
    
    # [CƠ CHẾ BẢO VỆ 2]: Chờ thêm 0.2s để gom nốt các gói tin "đến muộn" (Jitter từ USB)
    time.sleep(0.2) 
    print("Recording stopped. Checking quality...")

    frames = synced_frames()
    counts = frame_counts()
    duplicates, malformed = recording_error_counts()
    
    # Tính toán khoảng đứt gãy lớn nhất trực tiếp trên danh sách frames
    max_gap = get_max_consecutive_missing_seq(frames)
    first_word_invalid_counts = count_first_word_invalid_clean_frames(frames)
    first_word_invalid_failure = first_word_invalid_reason(first_word_invalid_counts)

    reason = ""
    # Kiểm tra số lượng khung hình đồng bộ có tối thiểu không (Ngưỡng 200)
    if len(frames) < MIN_SYNCED_FRAMES:
        reason = f"not_enough_synced_frames_{len(frames)}"
    # Kiểm tra Mù tín hiệu (Rớt liên tục quá 30 gói)
    elif max_gap > MAX_CONSECUTIVE_MISSING_SEQ:
        reason = f"massive_seq_gap_{max_gap}"
    elif first_word_invalid_failure:
        reason = first_word_invalid_failure
    elif any(frame[rx]["tx_id"] != EXPECTED_TX_ID for _, frame in frames for rx in RX_ORDER):
        reason = "tx_id_mismatch"
    elif any(frame[rx]["channel"] != CHANNEL for _, frame in frames for rx in RX_ORDER):
        reason = "channel_mismatch"
    elif any(frame[rx]["csi_len"] != CSI_VALUES_PER_RX for _, frame in frames for rx in RX_ORDER):
        reason = "bad_csi_len"

    if reason:
        append_quality(sid, "fail", len(frames), len(frames), counts, duplicates, malformed, max_gap, frames, reason)
        print(f"[FAIL] {sid}: {reason}. Please record this same sample again.")
        print(f"RX frames: RX1={counts['RX1']} RX2={counts['RX2']} RX3={counts['RX3']} | synced={len(frames)}")
        return False

    write_data_rows(sid, person, cell_id, coarse_pose, orientation_label, trial, frames)
    append_manifest(sid, person, cell_id, coarse_pose, orientation_label, trial, frames)
    append_quality(sid, "pass", len(frames), len(frames), counts, duplicates, malformed, max_gap, frames, "")
    passed_sample_ids.add(sid)
    print(f"[PASS] Saved {sid}: {len(frames)} clean synced frames -> {data_file_for_person(person).name}")
    return True

def print_session_summary() -> None:
    print("\nSESSION CONFIG")
    print(f"Session: {SESSION_ID}")
    print(f"Channel: {CHANNEL}")
    print(f"Layout: {LAYOUT_ID}")
    print(f"Layout type: {LAYOUT_TYPE}")
    print(f"Collection plan ID: {COLLECTION_PLAN_ID}")
    print(f"Calibration: {CALIBRATION_ID}")
    print(f"COM: {COM_PORTS}")
    print(f"Output: {session_dir()}")
    print(f"Label schema: {LABEL_SCHEMA_VERSION}")
    print(f"Cell grid: {CELL_GRID_COLS} cols x {CELL_GRID_ROWS} rows = {len(CELLS)} cells")
    print(f"Collection mode: {COLLECTION_MODE}")
    print(f"Stage: {ACTIVE_STAGE_CONFIG['description'] if ACTIVE_STAGE_CONFIG else COLLECTION_STAGE_NOTES[COLLECTION_MODE]}")
    print(f"Collection persons: {COLLECTION_PERSONS}")
    print(f"Output person files: {OUTPUT_PERSONS}")
    print(f"Empty identifiers: {EMPTY_PERSON}/{EMPTY_CELL_ID} (identifier-only, no actor in grid)")
    print(f"Collection cells: {COLLECTION_CELLS}")
    print(f"Collection poses: {COLLECTION_POSES}")
    print(f"Collection orientations: {COLLECTION_ORIENTATIONS}")
    print(f"Trials per case: {TRIALS_PER_CASE if ACTIVE_STAGE_CONFIG else 'explicit per planned case'}")
    print(f"Planned samples: {len(planned_cases())}")
    print(f"Expected clean rows: {len(planned_cases()) * MIN_SYNCED_FRAMES}")
    print(f"Layout notes: {LAYOUT_DESCRIPTION}")
    print("Directions: Facing_RX=+x/east/RX2, Facing_TX=-x/west/TX, Facing_Left=-y/north/RX1, Facing_Right=+y/south/RX3")
    print("Plan: explicit planned_cases() list; use --plan-only to inspect counts before opening serial")

def confirm_or_exit(prompt: str) -> None:
    answer = input(prompt + " [y/N]: ").strip().lower()
    if answer != "y":
        print("Stopped by operator.")
        sys.exit(0)

def main() -> None:
    global stop_threads, passed_sample_ids
    validate_plan_or_raise()
    if PLAN_ONLY:
        print_plan_only()
        sys.exit(0)
    ensure_dependency()
    create_session_files()
    passed_sample_ids = load_passed_sample_ids()
    print_session_summary()
    if passed_sample_ids:
        print(f"Existing passed samples in this session: {len(passed_sample_ids)}. They will be skipped.")

    confirm_or_exit("Have you manually checked COM ports, channel, RX_ID, TX_MAC_FILTER, and calibration notes?")

    start_reader_threads()
    time.sleep(2)
    preflight_seconds = SOAK_SECONDS if (PREFLIGHT_ONLY or SOAK_ONLY) else 5
    preflight_ok = preflight_check(seconds=preflight_seconds)
    if PREFLIGHT_ONLY or SOAK_ONLY:
        stop_threads = True
        time.sleep(0.5)
        if preflight_ok:
            print("Soak-only check passed." if SOAK_ONLY else "Preflight-only check passed.")
            sys.exit(0)
        print("Soak-only check failed." if SOAK_ONLY else "Preflight-only check failed.")
        sys.exit(1)
    if not preflight_ok:
        stop_threads = True
        time.sleep(0.5)
        print("Preflight failed. Collection is blocked until all RX pass at the configured thresholds.")
        sys.exit(1)

    confirm_or_exit(f"Start {COLLECTION_MODE} collection loop now?")

    try:
        cases = planned_cases()
        for index, (person, cell_id, coarse_pose, orientation_label, trial) in enumerate(cases, start=1):
            print(f"Plan progress: {index}/{len(cases)}")
            while True:
                ok = capture_one_sample(person, cell_id, coarse_pose, orientation_label, trial)
                if ok:
                    break
                input("Press ENTER to retry the same sample...")
    except KeyboardInterrupt:
        print("\nStopped by operator. Already saved accepted samples remain valid.")
    finally:
        stop_threads = True
        time.sleep(0.5)
        print(f"Session folder: {session_dir()}")

if __name__ == "__main__":
    main()