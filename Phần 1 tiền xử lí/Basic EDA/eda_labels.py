
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eda_common import (
    CELL_IDS,
    GRID_COLS,
    GRID_ROWS,
    default_figures_dir,
    default_session_dir,
    ensure_dir,
    load_manifest,
    load_quality,
    manifest_path,
    quality_path,
    set_plot_style,
)


ORIENTATION_DEGREES = {
    "Empty": 0,
    "Facing_RX": 0,
    "Facing_TX": 180,
    "Facing_Left": 270,
    "Facing_Right": 90,
    "Diagonal_RX1": 315,
    "Diagonal_RX3": 45,
}

VALID_POSES = {"Empty", "Standing", "Sitting", "Lying", "Arms_Out", "Walk_in_place", "Transition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label distribution and consistency EDA tables.")
    session_dir = default_session_dir()
    parser.add_argument("--manifest", type=Path, default=manifest_path(session_dir))
    parser.add_argument("--quality", type=Path, default=quality_path(session_dir))
    parser.add_argument("--figures-dir", type=Path, default=default_figures_dir())
    return parser.parse_args()


def count_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = frame[column].fillna("Missing").astype(str).value_counts().rename_axis(column).reset_index(name="count")
    counts["percent"] = counts["count"] / counts["count"].sum() * 100.0
    return counts


def save_bar(table: pd.DataFrame, label_column: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    ordered = table.sort_values("count", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(ordered)))
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.42 * len(ordered))))
    bars = axis.barh(ordered[label_column], ordered["count"], color=colors)
    axis.set_title(title)
    axis.set_xlabel("Samples")
    axis.set_ylabel(label_column)
    axis.bar_label(bars, padding=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_cell_heatmap(manifest: pd.DataFrame, output_png: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    counts = manifest["cell_id"].fillna("Missing").astype(str).value_counts()
    rows = []
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for index, cell_id in enumerate(CELL_IDS):
        row = index // GRID_COLS
        col = index % GRID_COLS
        value = int(counts.get(cell_id, 0))
        grid[row, col] = value
        rows.append({"cell_id": cell_id, "row": row + 1, "col": col + 1, "count": value})
    cell_counts = pd.DataFrame(rows)

    set_plot_style()
    fig, axis = plt.subplots(figsize=(7.5, 6.6))
    image = axis.imshow(grid, cmap="YlGnBu")
    axis.set_title("Sample count by 5x5 cell")
    axis.set_xticks(range(GRID_COLS), labels=[str(i) for i in range(1, GRID_COLS + 1)])
    axis.set_yticks(range(GRID_ROWS), labels=[str(i) for i in range(1, GRID_ROWS + 1)])
    axis.set_xlabel("Cell column")
    axis.set_ylabel("Cell row")
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            axis.text(col, row, f"C{row * GRID_COLS + col + 1:02d}\n{grid[row, col]}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Samples")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return cell_counts


def consistency_issues(manifest: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, object]] = []

    for row in manifest.to_dict(orient="records"):
        sample_id = row["sample_id"]
        presence = int(row["presence"])
        cell_id = str(row["cell_id"])
        pose = str(row["coarse_pose"])
        orientation = str(row["orientation_label"])
        occupied = "" if pd.isna(row.get("occupied_cells")) else str(row.get("occupied_cells", ""))
        occupied_cells = [part.strip() for part in occupied.split("|") if part.strip()]
        x = float(row["center_x_m"])
        y = float(row["center_y_m"])
        orientation_deg = int(row.get("orientation_deg", -1)) if not pd.isna(row.get("orientation_deg")) else -1
        footprint_yaw_deg = int(row.get("footprint_yaw_deg", -1)) if not pd.isna(row.get("footprint_yaw_deg")) else -1

        def add(reason: str) -> None:
            issues.append(
                {
                    "sample_id": sample_id,
                    "presence": presence,
                    "cell_id": cell_id,
                    "coarse_pose": pose,
                    "orientation_label": orientation,
                    "reason": reason,
                }
            )

        if pose not in VALID_POSES:
            add(f"unknown_pose:{pose}")
        if orientation not in ORIENTATION_DEGREES:
            add(f"unknown_orientation:{orientation}")
        elif orientation_deg != ORIENTATION_DEGREES[orientation]:
            add(f"orientation_deg_mismatch:expected_{ORIENTATION_DEGREES[orientation]}_got_{orientation_deg}")
        if orientation in ORIENTATION_DEGREES and footprint_yaw_deg != ORIENTATION_DEGREES[orientation]:
            add(f"footprint_yaw_deg_mismatch:expected_{ORIENTATION_DEGREES[orientation]}_got_{footprint_yaw_deg}")
        if presence == 0 and (cell_id != "Empty" or pose != "Empty" or orientation != "Empty" or occupied):
            add("empty_presence_has_non_empty_label")
        if presence == 1 and cell_id not in CELL_IDS:
            add("human_presence_has_invalid_cell")
        if presence == 1 and not occupied_cells:
            add("human_presence_has_empty_footprint")
        if presence == 1 and cell_id in CELL_IDS and cell_id not in occupied_cells:
            add("cell_id_not_in_occupied_cells")
        if presence == 1 and not (0.0 <= x <= 3.0 and 0.0 <= y <= 3.0):
            add("center_outside_room")
        for cell in occupied_cells:
            if cell not in CELL_IDS:
                add(f"invalid_occupied_cell:{cell}")
    return pd.DataFrame(issues, columns=["sample_id", "presence", "cell_id", "coarse_pose", "orientation_label", "reason"])


def split_feasibility(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("coarse_pose", "orientation_label", "person", "cell_id"):
        counts = manifest[column].fillna("Missing").astype(str).value_counts()
        for label, count in counts.items():
            rows.append(
                {
                    "group": column,
                    "label": label,
                    "count": int(count),
                    "approx_train_70": int(round(count * 0.70)),
                    "approx_val_15": int(round(count * 0.15)),
                    "approx_test_15": int(count - round(count * 0.70) - round(count * 0.15)),
                    "risk": "high" if count < 60 else "medium" if count < 120 else "low",
                }
            )
    return pd.DataFrame(rows)


def join_unique(values: pd.Series) -> str:
    return "|".join(sorted(set(values.astype(str))))


def split_group_feasibility(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = {
        "sample_id": ["sample_id"],
        "session_person_trial": ["session_id", "person", "trial"],
        "person": ["person"],
    }
    for group_name, columns in group_specs.items():
        grouped = manifest.groupby(columns, dropna=False).agg(
            samples=("sample_id", "count"),
            poses=("coarse_pose", join_unique),
            orientations=("orientation_label", join_unique),
            cells=("cell_id", join_unique),
        )
        group_count = len(grouped)
        rows.append(
            {
                "group_key": group_name,
                "group_count": group_count,
                "min_samples_per_group": int(grouped["samples"].min()) if group_count else 0,
                "median_samples_per_group": float(grouped["samples"].median()) if group_count else 0.0,
                "max_samples_per_group": int(grouped["samples"].max()) if group_count else 0,
                "approx_train_groups_70": int(round(group_count * 0.70)),
                "approx_val_groups_15": int(round(group_count * 0.15)),
                "approx_test_groups_15": int(group_count - round(group_count * 0.70) - round(group_count * 0.15)),
                "leakage_note": "split before windowing; never split windows from the same group",
            }
        )
    return pd.DataFrame(rows)


def person_independent_feasibility(manifest: pd.DataFrame) -> pd.DataFrame:
    human = manifest[manifest["presence"] == 1]
    all_human_poses = set(human["coarse_pose"].astype(str).unique())
    rows = []
    for person, group in human.groupby("person"):
        test_poses = set(group["coarse_pose"].astype(str).unique())
        train_pool = human[human["person"] != person]
        train_poses = set(train_pool["coarse_pose"].astype(str).unique())
        rows.append(
            {
                "held_out_person": person,
                "test_samples": int(len(group)),
                "train_pool_samples": int(len(train_pool)),
                "test_pose_count": len(test_poses),
                "missing_test_poses": "|".join(sorted(all_human_poses - test_poses)),
                "missing_train_poses_if_held_out": "|".join(sorted(all_human_poses - train_poses)),
                "recommended_as_person_independent_test": bool(len(test_poses) == len(all_human_poses) and len(train_poses) == len(all_human_poses)),
            }
        )
    return pd.DataFrame(rows)


def build_label_tables(
    manifest_path_value: Path | None = None,
    quality_path_value: Path | None = None,
    figures_dir: Path = default_figures_dir(),
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    session_dir = default_session_dir()
    manifest_path_value = manifest_path(session_dir) if manifest_path_value is None else manifest_path_value
    quality_path_value = quality_path(session_dir) if quality_path_value is None else quality_path_value
    manifest = load_manifest(manifest_path_value)
    quality = load_quality(quality_path_value)
    if save_figures:
        ensure_dir(figures_dir)

    tables: dict[str, pd.DataFrame] = {}
    for column, title, filename in (
        ("coarse_pose", "Pose distribution", "pose_distribution.png"),
        ("orientation_label", "Orientation distribution", "orientation_distribution.png"),
        ("person", "Person distribution", "person_distribution.png"),
        ("presence", "Presence distribution", "presence_distribution.png"),
    ):
        table = count_table(manifest, column)
        tables[f"{column}_distribution"] = table
        if save_figures:
            save_bar(table, column, title, figures_dir / filename)

    merged_status = quality["status"].fillna("Missing").astype(str).value_counts().rename_axis("quality_status")
    tables["quality_status_distribution"] = merged_status.reset_index(name="count")
    tables["cell_5x5_counts"] = save_cell_heatmap(manifest, figures_dir / "cell_5x5_heatmap.png") if save_figures else cell_count_table(manifest)
    tables["label_consistency_issues"] = consistency_issues(manifest)
    tables["split_feasibility"] = split_feasibility(manifest)
    tables["split_group_feasibility"] = split_group_feasibility(manifest)
    tables["person_independent_feasibility"] = person_independent_feasibility(manifest)
    return tables


def cell_count_table(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = manifest["cell_id"].fillna("Missing").astype(str).value_counts()
    rows = []
    for index, cell_id in enumerate(CELL_IDS):
        rows.append({"cell_id": cell_id, "row": index // GRID_COLS + 1, "col": index % GRID_COLS + 1, "count": int(counts.get(cell_id, 0))})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    figures_dir = ensure_dir(args.figures_dir)
    tables = build_label_tables(args.manifest, args.quality, figures_dir, save_figures=True)
    print(f"Built {len(tables)} label EDA tables in memory")
    print(f"Saved label EDA figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
