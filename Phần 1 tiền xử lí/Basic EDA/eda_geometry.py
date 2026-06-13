
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
    manifest_path,
    set_plot_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Part 1 label geometry and footprint EDA tables.")
    session_dir = default_session_dir()
    parser.add_argument("--manifest", type=Path, default=manifest_path(session_dir))
    parser.add_argument("--figures-dir", type=Path, default=default_figures_dir())
    return parser.parse_args()


def occupied_cell_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and np.isnan(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len([cell for cell in text.split("|") if cell.strip()])


def add_geometry_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    frame = manifest.copy()
    numeric_columns = [
        "presence",
        "center_x_m",
        "center_y_m",
        "footprint_length_m",
        "footprint_width_m",
        "footprint_yaw_deg",
        "frames",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["occupied_cell_count"] = frame["occupied_cells"].map(occupied_cell_count)
    frame["footprint_area_m2"] = frame["footprint_length_m"] * frame["footprint_width_m"]
    frame["is_boundary_cell"] = frame["cell_id"].isin(boundary_cells())
    return frame


def boundary_cells() -> set[str]:
    cells = set()
    for index, cell_id in enumerate(CELL_IDS):
        row = index // GRID_COLS
        col = index % GRID_COLS
        if row in (0, GRID_ROWS - 1) or col in (0, GRID_COLS - 1):
            cells.add(cell_id)
    return cells


def normalize_crosstab(table: pd.DataFrame, index_name: str) -> pd.DataFrame:
    table = table.reset_index().rename(columns={table.index.name or "index": index_name})
    return table


def crosstab(frame: pd.DataFrame, row: str, column: str) -> pd.DataFrame:
    table = pd.crosstab(frame[row].fillna("Missing"), frame[column].fillna("Missing"))
    return normalize_crosstab(table, row)


def pose_geometry_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("coarse_pose", dropna=False).agg(
        samples=("sample_id", "count"),
        mean_frames=("frames", "mean"),
        mean_occupied_cells=("occupied_cell_count", "mean"),
        max_occupied_cells=("occupied_cell_count", "max"),
        mean_footprint_area_m2=("footprint_area_m2", "mean"),
        mean_center_x_m=("center_x_m", "mean"),
        mean_center_y_m=("center_y_m", "mean"),
        std_center_x_m=("center_x_m", "std"),
        std_center_y_m=("center_y_m", "std"),
        boundary_cell_samples=("is_boundary_cell", "sum"),
    )
    summary = grouped.reset_index()
    summary["boundary_cell_percent"] = summary["boundary_cell_samples"] / summary["samples"].clip(lower=1) * 100.0
    return summary


def cell_risk_summary(frame: pd.DataFrame) -> pd.DataFrame:
    human = frame[frame["presence"] == 1].copy()
    counts = human["cell_id"].value_counts()
    rows = []
    for index, cell_id in enumerate(CELL_IDS):
        row = index // GRID_COLS + 1
        col = index % GRID_COLS + 1
        count = int(counts.get(cell_id, 0))
        rows.append(
            {
                "cell_id": cell_id,
                "row": row,
                "col": col,
                "samples": count,
                "is_boundary_cell": cell_id in boundary_cells(),
                "risk": "high" if count < 20 else "medium" if count < 60 else "low",
            }
        )
    return pd.DataFrame(rows)


def save_center_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    human = frame[frame["presence"] == 1].copy()
    poses = sorted(human["coarse_pose"].dropna().astype(str).unique())
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(poses), 1)))
    color_map = dict(zip(poses, colors))

    fig, axis = plt.subplots(figsize=(7.2, 7.0))
    for pose in poses:
        group = human[human["coarse_pose"].astype(str) == pose]
        axis.scatter(
            group["center_x_m"],
            group["center_y_m"],
            s=38,
            alpha=0.72,
            label=pose,
            color=color_map[pose],
            edgecolors="white",
            linewidths=0.4,
        )
    for tick in np.linspace(0.0, 3.0, 6):
        axis.axhline(tick, color="#CBD5E1", linewidth=0.7, zorder=0)
        axis.axvline(tick, color="#CBD5E1", linewidth=0.7, zorder=0)
    axis.set_xlim(-0.05, 3.05)
    axis.set_ylim(3.05, -0.05)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Center label distribution in 3m x 3m room")
    axis.set_xlabel("center_x_m")
    axis.set_ylabel("center_y_m")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_footprint_boxplot(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    human = frame[frame["presence"] == 1].copy()
    poses = sorted(human["coarse_pose"].dropna().astype(str).unique())
    data = [human.loc[human["coarse_pose"].astype(str) == pose, "occupied_cell_count"].to_numpy() for pose in poses]

    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    box = axis.boxplot(data, tick_labels=poses, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, max(len(poses), 1)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axis.set_title("Footprint size by pose")
    axis.set_xlabel("Coarse pose")
    axis.set_ylabel("Occupied cell count")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_cell_risk_heatmap(cell_risk: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for row in cell_risk.to_dict(orient="records"):
        grid[int(row["row"]) - 1, int(row["col"]) - 1] = int(row["samples"])

    fig, axis = plt.subplots(figsize=(7.5, 6.6))
    image = axis.imshow(grid, cmap="YlOrRd")
    axis.set_title("Human sample density and sparse-cell risk")
    axis.set_xticks(range(GRID_COLS), labels=[str(index) for index in range(1, GRID_COLS + 1)])
    axis.set_yticks(range(GRID_ROWS), labels=[str(index) for index in range(1, GRID_ROWS + 1)])
    axis.set_xlabel("Cell column")
    axis.set_ylabel("Cell row")
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cell_id = f"C{row * GRID_COLS + col + 1:02d}"
            axis.text(col, row, f"{cell_id}\n{grid[row, col]}", ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=axis, label="Human samples")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_geometry_tables(
    manifest_path_value: Path | None = None,
    figures_dir: Path = default_figures_dir(),
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    session_dir = default_session_dir()
    manifest_path_value = manifest_path(session_dir) if manifest_path_value is None else manifest_path_value
    if save_figures:
        ensure_dir(figures_dir)
    manifest = load_manifest(manifest_path_value)
    frame = add_geometry_columns(manifest)
    cell_risk = cell_risk_summary(frame)

    if save_figures:
        save_center_scatter(frame, figures_dir / "center_distribution.png")
        save_footprint_boxplot(frame, figures_dir / "footprint_size_by_pose.png")
        save_cell_risk_heatmap(cell_risk, figures_dir / "cell_risk_heatmap.png")

    return {
        "sample_geometry_features": frame,
        "pose_geometry_summary": pose_geometry_summary(frame),
        "cell_risk_summary": cell_risk,
        "cell_pose_crosstab": crosstab(frame, "cell_id", "coarse_pose"),
        "cell_orientation_crosstab": crosstab(frame, "cell_id", "orientation_label"),
        "person_pose_crosstab": crosstab(frame, "person", "coarse_pose"),
        "pose_orientation_crosstab": crosstab(frame, "coarse_pose", "orientation_label"),
    }


def main() -> int:
    args = parse_args()
    figures_dir = ensure_dir(args.figures_dir)
    tables = build_geometry_tables(args.manifest, figures_dir, save_figures=True)
    print(f"Built {len(tables)} geometry EDA tables in memory")
    print(f"Saved geometry EDA figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
