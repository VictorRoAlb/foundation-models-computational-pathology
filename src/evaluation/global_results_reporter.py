from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


MODEL_ORDER = ["KEEP", "TITAN", "CONCH", "MUSK", "Patho-CLIP", "PRISM"]
MODEL_COLORS = {
    "KEEP": "#1b9e77",
    "TITAN": "#d95f02",
    "CONCH": "#7570b3",
    "MUSK": "#66a61e",
    "Patho-CLIP": "#e7298a",
    "PRISM": "#1f78b4",
}
K_VALUES = ["1", "3", "5", "10"]
REQUIRED_FILENAMES = {
    "ai4skin_metrics": "metrics_all_tasks.json",
    "sicap_metrics": "sicap_metrics_all.json",
    "domain_metrics": "metrics_binary_crossmodal_ai4skin_vs_assist_6models.json",
    "majority_voting_confusion": "majority_voting_returned_class_pct.json",
}
OPTIONAL_FILENAMES = {
    "sicap_summary_csv": "sicap_metrics_summary_long.csv",
    "majority_voting_records": "majority_voting_records.jsonl",
    "majority_voting_confusion_xlsx": "majority_voting_returned_class_pct.xlsx",
}
PREFERRED_PATHS = {
    "ai4skin_metrics": Path("MEGA_EVAL_OUTPUT") / "metrics_all_tasks.json",
    "sicap_metrics": Path("SICAP_Evaluation") / "metrics" / "sicap_metrics_all.json",
    "domain_metrics": Path("MEGA_EVAL_OUTPUT_BINARY_AI4SKIN_VS_ASSIST")
    / "metrics_binary_crossmodal_ai4skin_vs_assist_6models.json",
    "majority_voting_confusion": Path("MEGA_EVAL_OUTPUT") / "majority_voting_returned_class_pct.json",
    "sicap_summary_csv": Path("SICAP_Evaluation") / "metrics" / "sicap_metrics_summary_long.csv",
    "majority_voting_records": Path("MEGA_EVAL_OUTPUT") / "majority_voting_records.jsonl",
    "majority_voting_confusion_xlsx": Path("MEGA_EVAL_OUTPUT") / "majority_voting_returned_class_pct.xlsx",
}
SEARCH_PRUNE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "conch_env",
    "node_modules",
}
FIGURE_FILENAMES = {
    "scoreboard": "01_global_scoreboard_heatmap.png",
    "recall_curves": "02_recall_at_k_curves.png",
    "fine_grained": "03_fine_grained_ranking.png",
    "domain_bias": "04_domain_bias_vs_exact_pair.png",
    "gleason_heatmap": "05_sicap_gleason_per_class_heatmap.png",
    "majority_voting": "06_majority_voting_confusion_matrices.png",
}
CLASS_LABEL_OVERRIDES = {
    "AI4SKIN": {
        "0": "Class 0",
        "1": "Class 1",
        "2": "Class 2",
        "3": "Class 3",
        "4": "Class 4",
        "5": "Class 5",
        "6": "Class 6",
    }
}


@dataclass
class InputFiles:
    required: dict[str, Path]
    optional: dict[str, Path]
    missing_optional: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate global tables and figures for foundation-model retrieval evaluation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory where the metrics files are located. Defaults to current working directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "GLOBAL_RESULTS_REPORT",
        help="Output directory for the generated report assets.",
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    canonical = str(name).strip().replace("_", "-").upper()
    mapping = {
        "KEEP": "KEEP",
        "TITAN": "TITAN",
        "CONCH": "CONCH",
        "MUSK": "MUSK",
        "PATHO-CLIP": "Patho-CLIP",
        "PATHOCLIP": "Patho-CLIP",
        "PRISM": "PRISM",
    }
    return mapping.get(canonical, name.strip())


def normalize_direction(direction: str) -> str:
    mapping = {
        "report_to_image": "report_to_image",
        "image_to_report": "image_to_report",
        "report_to_report": "report_to_report",
        "image_to_image": "image_to_image",
        "exact_pair_report_to_image": "report_to_image",
        "exact_pair_image_to_report": "image_to_report",
        "same_dataset_pct_report_to_image": "report_to_image",
        "same_dataset_pct_image_to_report": "image_to_report",
        "per_class_report_to_image": "report_to_image",
        "per_class_image_to_report": "image_to_report",
        "per_class_report_to_report": "report_to_report",
        "per_class_image_to_image": "image_to_image",
        "per_class_same_dataset_pct_report_to_image": "report_to_image",
        "per_class_same_dataset_pct_image_to_report": "image_to_report",
    }
    return mapping.get(direction, direction)


def direction_to_display(direction: str) -> str:
    mapping = {
        "report_to_image": "R→I",
        "image_to_report": "I→R",
        "report_to_report": "R→R",
        "image_to_image": "I→I",
    }
    return mapping.get(direction, direction)


def wrap_tick_label(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def sorted_models(models: Iterable[str]) -> list[str]:
    unique = []
    seen = set()
    for model in models:
        normalized = normalize_model_name(model)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    order_index = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    return sorted(unique, key=lambda name: (order_index.get(name, len(order_index)), name))


def find_single_file(input_dir: Path, filename: str, preferred_relative: Path | None = None) -> Path | None:
    if preferred_relative is not None:
        preferred_candidate = input_dir / preferred_relative
        if preferred_candidate.exists():
            return preferred_candidate

    matches: list[Path] = []
    for root, dirs, files in os.walk(input_dir, topdown=True):
        dirs[:] = [dirname for dirname in dirs if dirname not in SEARCH_PRUNE_DIRS]
        if filename in files:
            matches.append(Path(root) / filename)
    if not matches:
        return None
    return min(matches, key=lambda path: (len(path.parts), len(str(path))))


def discover_input_files(input_dir: Path) -> InputFiles:
    required: dict[str, Path] = {}
    optional: dict[str, Path] = {}
    missing_optional: list[str] = []

    for key, filename in REQUIRED_FILENAMES.items():
        match = find_single_file(input_dir, filename, PREFERRED_PATHS.get(key))
        if match is None:
            raise FileNotFoundError(
                f"Required file '{filename}' was not found under '{input_dir.resolve()}'."
            )
        required[key] = match

    for key, filename in OPTIONAL_FILENAMES.items():
        match = find_single_file(input_dir, filename, PREFERRED_PATHS.get(key))
        if match is None:
            missing_optional.append(filename)
        else:
            optional[key] = match

    return InputFiles(required=required, optional=optional, missing_optional=missing_optional)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def append_scalar_metrics(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    task: str,
    metric_key: str,
    model: str,
    value_map: dict[str, Any],
) -> None:
    direction = normalize_direction(metric_key)
    if metric_key.startswith("exact_pair_"):
        metric_type = "exact_pair_recall"
    elif metric_key.startswith("same_dataset_pct_"):
        metric_type = "same_dataset_pct"
    else:
        metric_type = "recall"

    for k, value in value_map.items():
        if str(k) not in K_VALUES and str(k).lower() != "mean":
            continue
        records.append(
            {
                "dataset": dataset,
                "task": task,
                "direction": direction,
                "model": normalize_model_name(model),
                "metric_type": metric_type,
                "k": str(k),
                "value": safe_float(value),
            }
        )


def parse_nested_direction_first(
    payload: dict[str, Any],
    *,
    dataset: str,
    task: str,
    records: list[dict[str, Any]],
) -> None:
    for metric_key, model_map in payload.items():
        if not isinstance(model_map, dict):
            continue
        if metric_key.startswith("per_class_"):
            continue
        for model, value_map in model_map.items():
            if isinstance(value_map, dict):
                append_scalar_metrics(
                    records,
                    dataset=dataset,
                    task=task,
                    metric_key=metric_key,
                    model=model,
                    value_map=value_map,
                )


def parse_nested_model_first(
    payload: dict[str, Any],
    *,
    dataset: str,
    task: str,
    records: list[dict[str, Any]],
) -> None:
    for model, metric_map in payload.items():
        if not isinstance(metric_map, dict):
            continue
        for metric_key, value_map in metric_map.items():
            if metric_key.startswith("per_class_"):
                continue
            if isinstance(value_map, dict):
                append_scalar_metrics(
                    records,
                    dataset=dataset,
                    task=task,
                    metric_key=metric_key,
                    model=model,
                    value_map=value_map,
                )


def build_global_long_table(
    ai4skin_metrics: dict[str, Any],
    sicap_metrics: dict[str, Any],
    domain_metrics: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for task, payload in ai4skin_metrics.items():
        if isinstance(payload, dict):
            parse_nested_direction_first(
                payload,
                dataset="AI4SKIN",
                task=task,
                records=records,
            )

    for task, payload in sicap_metrics.items():
        if isinstance(payload, dict):
            parse_nested_direction_first(
                payload,
                dataset="SICAP",
                task=task,
                records=records,
            )

    domain_root = domain_metrics.get("binary_crossmodal", {})
    if isinstance(domain_root, dict):
        parse_nested_model_first(
            domain_root,
            dataset="AI4SKIN vs ASSIST",
            task="binary_crossmodal",
            records=records,
        )

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        raise ValueError("No scalar metrics could be parsed from the provided files.")

    long_df["model"] = long_df["model"].map(normalize_model_name)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.sort_values(
        by=["dataset", "task", "metric_type", "direction", "model", "k"]
    ).reset_index(drop=True)


def mean_over_k(
    long_df: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    direction: str,
    metric_type: str = "recall",
    model: str | None = None,
) -> float:
    mask = (
        (long_df["dataset"] == dataset)
        & (long_df["task"] == task)
        & (long_df["direction"] == direction)
        & (long_df["metric_type"] == metric_type)
        & (long_df["k"].isin(K_VALUES))
    )
    if model is not None:
        mask &= long_df["model"] == model
    values = long_df.loc[mask, "value"].astype(float)
    if values.empty:
        return float("nan")
    return float(values.mean())


def build_scoreboard(long_df: pd.DataFrame) -> pd.DataFrame:
    scoreboard_columns = [
        ("AI4SKIN multiclass R→I", "AI4SKIN", "crossmodal_multiclass", "report_to_image"),
        ("AI4SKIN multiclass I→R", "AI4SKIN", "crossmodal_multiclass", "image_to_report"),
        (
            "AI4SKIN majority voting R→I",
            "AI4SKIN",
            "majority_voting_multiclass",
            "report_to_image",
        ),
        (
            "AI4SKIN majority voting I→R",
            "AI4SKIN",
            "majority_voting_multiclass",
            "image_to_report",
        ),
        ("AI4SKIN binary R→I", "AI4SKIN", "binary_crossmodal", "report_to_image"),
        ("AI4SKIN binary I→R", "AI4SKIN", "binary_crossmodal", "image_to_report"),
        ("SICAP binary R→I", "SICAP", "binary_crossmodal", "report_to_image"),
        ("SICAP binary I→R", "SICAP", "binary_crossmodal", "image_to_report"),
        ("SICAP Gleason R→I", "SICAP", "gleason_crossmodal", "report_to_image"),
        ("SICAP Gleason I→R", "SICAP", "gleason_crossmodal", "image_to_report"),
        (
            "AI4SKIN vs ASSIST binary R→I",
            "AI4SKIN vs ASSIST",
            "binary_crossmodal",
            "report_to_image",
        ),
        (
            "AI4SKIN vs ASSIST binary I→R",
            "AI4SKIN vs ASSIST",
            "binary_crossmodal",
            "image_to_report",
        ),
    ]

    rows = []
    models = sorted_models(long_df["model"].unique())
    for model in models:
        row = {"model": model}
        for label, dataset, task, direction in scoreboard_columns:
            row[label] = mean_over_k(
                long_df,
                dataset=dataset,
                task=task,
                direction=direction,
                metric_type="recall",
                model=model,
            )
        fine_cols = [
            "AI4SKIN multiclass R→I",
            "AI4SKIN multiclass I→R",
            "AI4SKIN majority voting R→I",
            "AI4SKIN majority voting I→R",
            "SICAP Gleason R→I",
            "SICAP Gleason I→R",
        ]
        binary_cols = [
            "AI4SKIN binary R→I",
            "AI4SKIN binary I→R",
            "SICAP binary R→I",
            "SICAP binary I→R",
            "AI4SKIN vs ASSIST binary R→I",
            "AI4SKIN vs ASSIST binary I→R",
        ]
        primary_cols = [label for label, *_ in scoreboard_columns]
        row["Fine-grained score"] = float(pd.Series([row[c] for c in fine_cols]).mean())
        row["Binary score"] = float(pd.Series([row[c] for c in binary_cols]).mean())
        row["Overall mean"] = float(pd.Series([row[c] for c in primary_cols]).mean())
        rows.append(row)

    scoreboard = pd.DataFrame(rows)
    scoreboard["model"] = pd.Categorical(scoreboard["model"], categories=MODEL_ORDER, ordered=True)
    scoreboard = scoreboard.sort_values("model").reset_index(drop=True)
    scoreboard["model"] = scoreboard["model"].astype(str)
    return scoreboard


def build_model_rankings(scoreboard: pd.DataFrame) -> pd.DataFrame:
    ranking = scoreboard[["model", "Fine-grained score", "Binary score", "Overall mean"]].copy()
    ranking["Fine-grained rank"] = ranking["Fine-grained score"].rank(ascending=False, method="min")
    ranking["Binary rank"] = ranking["Binary score"].rank(ascending=False, method="min")
    ranking["Overall rank"] = ranking["Overall mean"].rank(ascending=False, method="min")
    return ranking.sort_values(
        by=["Fine-grained rank", "Overall rank", "Binary rank", "model"]
    ).reset_index(drop=True)


def build_domain_bias_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted_models(long_df["model"].unique()):
        if model not in set(long_df.loc[long_df["dataset"] == "AI4SKIN vs ASSIST", "model"]):
            continue
        same_dataset_values = [
            mean_over_k(
                long_df,
                dataset="AI4SKIN vs ASSIST",
                task="binary_crossmodal",
                direction=direction,
                metric_type="same_dataset_pct",
                model=model,
            )
            for direction in ["report_to_image", "image_to_report"]
        ]
        exact_pair_values = [
            mean_over_k(
                long_df,
                dataset="AI4SKIN vs ASSIST",
                task="binary_crossmodal",
                direction=direction,
                metric_type="exact_pair_recall",
                model=model,
            )
            for direction in ["report_to_image", "image_to_report"]
        ]
        binary_values = [
            mean_over_k(
                long_df,
                dataset="AI4SKIN vs ASSIST",
                task="binary_crossmodal",
                direction=direction,
                metric_type="recall",
                model=model,
            )
            for direction in ["report_to_image", "image_to_report"]
        ]
        rows.append(
            {
                "model": model,
                "same_dataset_pct_mean": float(pd.Series(same_dataset_values).mean()),
                "exact_pair_mean": float(pd.Series(exact_pair_values).mean()),
                "binary_recall_mean": float(pd.Series(binary_values).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def build_sicap_gleason_table(sicap_metrics: dict[str, Any]) -> pd.DataFrame:
    crossmodal = sicap_metrics.get("gleason_crossmodal", {})
    per_class_r2i = crossmodal.get("per_class_report_to_image", {})
    per_class_i2r = crossmodal.get("per_class_image_to_report", {})

    class_names: list[str] = []
    for model_map in [per_class_r2i, per_class_i2r]:
        for k_map in model_map.values():
            if isinstance(k_map, dict):
                for class_key in next(iter(k_map.values())).keys():
                    if class_key not in class_names:
                        class_names.append(class_key)
                if class_names:
                    break
        if class_names:
            break

    rows = []
    models = sorted_models(set(per_class_r2i.keys()) | set(per_class_i2r.keys()))
    for class_name in class_names:
        row = {"gleason_class": class_name}
        for model in models:
            values = []
            for per_class_map in [per_class_r2i, per_class_i2r]:
                raw_model = next(
                    (raw_name for raw_name in per_class_map if normalize_model_name(raw_name) == model),
                    None,
                )
                if raw_model is None:
                    continue
                for k in K_VALUES:
                    value = per_class_map.get(raw_model, {}).get(k, {}).get(class_name)
                    if value is not None:
                        values.append(safe_float(value))
            row[model] = float(np.mean(values)) if values else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_best_model_table(long_df: pd.DataFrame) -> pd.DataFrame:
    recall_df = long_df[
        (long_df["metric_type"] == "recall")
        & (long_df["k"].isin(K_VALUES))
        & (long_df["direction"].isin(["report_to_image", "image_to_report", "report_to_report", "image_to_image"]))
    ].copy()
    grouped = (
        recall_df.groupby(["dataset", "task", "direction", "model"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_recall"})
    )

    rows = []
    for (dataset, task, direction), block in grouped.groupby(["dataset", "task", "direction"]):
        block = block.sort_values("mean_recall", ascending=False).reset_index(drop=True)
        if block.empty:
            continue
        best = block.iloc[0]
        second = block.iloc[1] if len(block) > 1 else None
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "direction": direction_to_display(direction),
                "best_model": best["model"],
                "best_score": best["mean_recall"],
                "second_model": second["model"] if second is not None else "—",
                "second_score": second["mean_recall"] if second is not None else np.nan,
                "margin": best["mean_recall"] - second["mean_recall"] if second is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["dataset", "task", "direction"]
    ).reset_index(drop=True)


def audit_inputs(files: InputFiles, datasets: list[str], models: list[str]) -> None:
    print("=" * 88)
    print("GLOBAL RESULTS REPORT AUDIT")
    print("=" * 88)
    print("Required files detected:")
    for key, path in files.required.items():
        print(f"  - {key}: {path.resolve()}")
    print("Optional files detected:")
    if files.optional:
        for key, path in files.optional.items():
            print(f"  - {key}: {path.resolve()}")
    else:
        print("  - none")
    if files.missing_optional:
        print("Optional files missing:")
        for filename in files.missing_optional:
            print(f"  - {filename}")
    print("Datasets detected:")
    for dataset in datasets:
        print(f"  - {dataset}")
    print("Models detected:")
    for model in models:
        print(f"  - {model}")
    print("Figures scheduled:")
    for filename in FIGURE_FILENAMES.values():
        print(f"  - {filename}")
    print("=" * 88)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.grid": False,
        }
    )


def create_report_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "tfm_report_cmap",
        ["#f7fcfd", "#ccece6", "#66c2a4", "#2b8cbe", "#084081"],
    )


def annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    fmt: str = "{:.2f}",
    text_size: int = 9,
    nan_text: str = "—",
) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                label = nan_text
                color = "#666666"
            else:
                label = fmt.format(value)
                color = "white" if value >= 0.62 else "#1f1f1f"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=text_size)


def add_best_value_boxes(ax: plt.Axes, matrix: np.ndarray, by: str = "column") -> None:
    if by == "column":
        for col in range(matrix.shape[1]):
            values = matrix[:, col]
            if np.all(np.isnan(values)):
                continue
            best = np.nanmax(values)
            winners = np.where(np.isclose(values, best, equal_nan=False))[0]
            for row in winners:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.2,
                    )
                )
    elif by == "row":
        for row in range(matrix.shape[0]):
            values = matrix[row, :]
            if np.all(np.isnan(values)):
                continue
            best = np.nanmax(values)
            winners = np.where(np.isclose(values, best, equal_nan=False))[0]
            for col in winners:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.0,
                    )
                )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_global_scoreboard(scoreboard: pd.DataFrame, output_path: Path) -> None:
    plot_df = scoreboard.set_index("model")
    matrix = plot_df.to_numpy(dtype=float)
    cmap = create_report_cmap()
    cmap.set_bad(color="#ececec")

    fig, ax = plt.subplots(figsize=(23, 6.8))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("Global Retrieval Scoreboard", pad=18, fontweight="bold")
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels([wrap_tick_label(col, 18) for col in plot_df.columns], rotation=0)
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist())
    ax.set_xticks(np.arange(-0.5, plot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, plot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for xpos in [3.5, 5.5, 9.5, 11.5]:
        ax.axvline(x=xpos, color="#222222", linewidth=2.0)

    annotate_heatmap(ax, matrix, fmt="{:.2f}", text_size=9)
    add_best_value_boxes(ax, matrix, by="column")

    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Mean Recall@K", rotation=90, labelpad=12)

    fig.text(
        0.01,
        0.01,
        "Cells show mean Recall@K over K = {1, 3, 5, 10}. Black boxes indicate the best model per column. "
        "Fine-grained score excludes binary/domain tasks.",
        fontsize=10,
        color="#444444",
    )
    save_figure(fig, output_path)


def task_curve_value(
    long_df: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    model: str,
    k: str,
) -> float:
    values = []
    for direction in ["report_to_image", "image_to_report"]:
        mask = (
            (long_df["dataset"] == dataset)
            & (long_df["task"] == task)
            & (long_df["direction"] == direction)
            & (long_df["metric_type"] == "recall")
            & (long_df["model"] == model)
            & (long_df["k"] == k)
        )
        direction_values = long_df.loc[mask, "value"].astype(float)
        if not direction_values.empty:
            values.extend(direction_values.tolist())
    if not values:
        return float("nan")
    return float(np.mean(values))


def plot_recall_curves(long_df: pd.DataFrame, output_path: Path) -> None:
    task_specs = [
        ("AI4SKIN", "crossmodal_multiclass", "AI4SKIN multiclass cross-modal"),
        ("AI4SKIN", "majority_voting_multiclass", "AI4SKIN majority voting multiclass"),
        ("AI4SKIN", "binary_crossmodal", "AI4SKIN binary cross-modal"),
        ("SICAP", "binary_crossmodal", "SICAP binary cross-modal"),
        ("SICAP", "gleason_crossmodal", "SICAP Gleason cross-modal"),
        ("AI4SKIN vs ASSIST", "binary_crossmodal", "AI4SKIN vs ASSIST binary cross-modal"),
    ]
    models = sorted_models(long_df["model"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    x_values = [int(k) for k in K_VALUES]

    for ax, (dataset, task, title) in zip(axes, task_specs):
        for model in models:
            y_values = [
                task_curve_value(long_df, dataset=dataset, task=task, model=model, k=k)
                for k in K_VALUES
            ]
            if all(np.isnan(y) for y in y_values):
                continue
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.4,
                markersize=5.5,
                color=MODEL_COLORS.get(model, "#444444"),
                label=model,
            )
        ax.set_title(title, pad=10)
        ax.set_xticks(x_values)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("K")
        ax.set_ylabel("Recall@K")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Recall@K Curves Across Evaluation Tasks", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Each curve averages the two cross-modal directions when both are available, highlighting early retrieval quality at low K.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_figure(fig, output_path)


def plot_fine_grained_ranking(scoreboard: pd.DataFrame, output_path: Path) -> None:
    ranking = scoreboard[["model", "Fine-grained score"]].sort_values(
        "Fine-grained score", ascending=True
    )
    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = [MODEL_COLORS.get(model, "#555555") for model in ranking["model"]]
    bars = ax.barh(ranking["model"], ranking["Fine-grained score"], color=colors, edgecolor="none", height=0.62)
    ax.set_title("Fine-grained retrieval ranking", pad=16, fontweight="bold")
    ax.text(
        0.0,
        1.02,
        "multiclass + majority voting + SICAP Gleason",
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )
    ax.set_xlabel("Mean Recall@K")
    ax.set_xlim(0, max(1.0, float(np.nanmax(ranking["Fine-grained score"].to_numpy(dtype=float))) + 0.08))
    ax.grid(True, axis="x", color="#dedede", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, ranking["Fine-grained score"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_domain_bias_scatter(domain_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for _, row in domain_df.iterrows():
        size = 350 + 1300 * float(row["binary_recall_mean"])
        ax.scatter(
            row["same_dataset_pct_mean"],
            row["exact_pair_mean"],
            s=size,
            color=MODEL_COLORS.get(row["model"], "#666666"),
            alpha=0.82,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.text(
            row["same_dataset_pct_mean"] + 0.004,
            row["exact_pair_mean"] + 0.002,
            row["model"],
            fontsize=10,
            weight="bold",
        )

    ax.set_title("Domain Bias vs Exact Pair Recovery", pad=16, fontweight="bold")
    ax.set_xlabel("Mean same-dataset percentage")
    ax.set_ylabel("Mean exact-pair recall")
    ax.grid(True, color="#dddddd", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_min = max(0.0, float(domain_df["same_dataset_pct_mean"].min()) - 0.03)
    x_max = min(1.0, float(domain_df["same_dataset_pct_mean"].max()) + 0.03)
    y_min = max(0.0, float(domain_df["exact_pair_mean"].min()) - 0.02)
    y_max = min(1.0, float(domain_df["exact_pair_mean"].max()) + 0.03)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.text(
        0.02,
        0.015,
        "Same-dataset percentage measures domain clustering and should not be interpreted as diagnostic performance.",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, output_path)


def plot_gleason_heatmap(gleason_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = gleason_df.set_index("gleason_class")
    plot_df = plot_df[[column for column in MODEL_ORDER if column in plot_df.columns]]
    matrix = plot_df.to_numpy(dtype=float)
    cmap = create_report_cmap()
    cmap.set_bad(color="#ececec")

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("SICAP Gleason retrieval by class", pad=16, fontweight="bold")
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns.tolist())
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist())
    ax.set_xticks(np.arange(-0.5, plot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, plot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate_heatmap(ax, matrix, fmt="{:.2f}", text_size=10)
    add_best_value_boxes(ax, matrix, by="row")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean Recall@K", rotation=90, labelpad=10)
    fig.text(0.02, 0.015, "Mean over K and both cross-modal directions.", fontsize=10, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, output_path)


def infer_class_labels(dataset_name: str, class_ids: list[str]) -> list[str]:
    overrides = CLASS_LABEL_OVERRIDES.get(dataset_name, {})
    return [overrides.get(class_id, f"Class {class_id}") for class_id in class_ids]


def plot_majority_voting_matrices(
    confusion_payload: dict[str, Any],
    output_path: Path,
    *,
    task: str = "multiclass_crossmodal",
    dataset: str = "AI4SKIN",
    direction: str = "report_to_image",
    k: str = "5",
) -> None:
    task_payload = confusion_payload.get(task, {})
    models = sorted_models(task_payload.keys())
    if not models:
        raise ValueError("No majority voting confusion matrices were found for plotting.")

    panels = len(models)
    ncols = 3
    nrows = int(math.ceil(panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 5.0 * nrows))
    axes = np.atleast_1d(axes).flatten()
    cmap = create_report_cmap()
    cmap.set_bad(color="#ececec")
    shared_image = None

    for ax in axes[panels:]:
        ax.axis("off")

    for ax, model in zip(axes, models):
        raw_model = next(
            (raw_name for raw_name in task_payload if normalize_model_name(raw_name) == model),
            None,
        )
        matrix_dict = (
            task_payload.get(raw_model, {})
            .get(dataset, {})
            .get(direction, {})
            .get(k, {})
        )
        if not matrix_dict:
            ax.axis("off")
            ax.set_title(f"{model}\nmissing data")
            continue
        class_ids = sorted(matrix_dict.keys(), key=lambda item: int(item) if str(item).isdigit() else str(item))
        labels = infer_class_labels(dataset, class_ids)
        matrix = np.array(
            [
                [safe_float(matrix_dict.get(true_cls, {}).get(pred_cls)) for pred_cls in class_ids]
                for true_cls in class_ids
            ],
            dtype=float,
        )
        shared_image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(model, pad=10, fontweight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        annotate_heatmap(ax, matrix, fmt="{:.2f}", text_size=8)
        for idx in range(len(labels)):
            ax.add_patch(
                Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.7)
            )
        ax.set_xlabel("Retrieved class")
        ax.set_ylabel("True class")

    fig.suptitle(
        "AI4SKIN Majority Voting Confusion Matrices",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.02,
        "Default view: K = 5 and report-to-image retrieval. Values are returned-class percentages with a shared 0-1 scale.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    if shared_image is not None:
        cbar = fig.colorbar(shared_image, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Returned-class percentage", rotation=90, labelpad=10)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.10, wspace=0.28, hspace=0.32)
    save_figure(fig, output_path)


def export_tables(
    *,
    tables_dir: Path,
    long_df: pd.DataFrame,
    scoreboard: pd.DataFrame,
    rankings: pd.DataFrame,
    domain_bias: pd.DataFrame,
    gleason_table: pd.DataFrame,
    best_models: pd.DataFrame,
) -> list[Path]:
    exported_paths = []

    tables = {
        "global_metrics_long.csv": long_df,
        "global_scoreboard.csv": scoreboard,
        "model_rankings.csv": rankings,
        "domain_bias_vs_exact_pair.csv": domain_bias,
        "sicap_gleason_per_class.csv": gleason_table,
        "best_model_per_task.csv": best_models,
    }
    for filename, dataframe in tables.items():
        output_path = tables_dir / filename
        dataframe.to_csv(output_path, index=False)
        exported_paths.append(output_path)

    xlsx_path = tables_dir / "global_results_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="global_metrics_long", index=False)
        scoreboard.to_excel(writer, sheet_name="global_scoreboard", index=False)
        rankings.to_excel(writer, sheet_name="model_rankings", index=False)
        domain_bias.to_excel(writer, sheet_name="domain_bias_vs_exact_pair", index=False)
        gleason_table.to_excel(writer, sheet_name="sicap_gleason_per_class", index=False)
        best_models.to_excel(writer, sheet_name="best_model_per_task", index=False)
    exported_paths.append(xlsx_path)
    return exported_paths


def print_final_summary(
    *,
    figure_paths: list[Path],
    table_paths: list[Path],
    models: list[str],
    datasets: list[str],
) -> None:
    print("=" * 88)
    print("REPORT COMPLETED")
    print("=" * 88)
    print("Figures saved:")
    for path in figure_paths:
        print(f"  - {path.resolve()}")
    print("Tables saved:")
    for path in table_paths:
        print(f"  - {path.resolve()}")
    print("Models summarized:")
    for model in models:
        print(f"  - {model}")
    print("Datasets summarized:")
    for dataset in datasets:
        print(f"  - {dataset}")
    print("=" * 88)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    files = discover_input_files(input_dir)
    ai4skin_metrics = load_json(files.required["ai4skin_metrics"])
    sicap_metrics = load_json(files.required["sicap_metrics"])
    domain_metrics = load_json(files.required["domain_metrics"])
    majority_confusion = load_json(files.required["majority_voting_confusion"])

    long_df = build_global_long_table(ai4skin_metrics, sicap_metrics, domain_metrics)
    models = sorted_models(long_df["model"].unique())
    datasets = sorted(long_df["dataset"].unique().tolist())
    audit_inputs(files, datasets, models)

    scoreboard = build_scoreboard(long_df)
    rankings = build_model_rankings(scoreboard)
    domain_bias = build_domain_bias_table(long_df)
    gleason_table = build_sicap_gleason_table(sicap_metrics)
    best_models = build_best_model_table(long_df)

    plot_global_scoreboard(scoreboard, figures_dir / FIGURE_FILENAMES["scoreboard"])
    plot_recall_curves(long_df, figures_dir / FIGURE_FILENAMES["recall_curves"])
    plot_fine_grained_ranking(scoreboard, figures_dir / FIGURE_FILENAMES["fine_grained"])
    plot_domain_bias_scatter(domain_bias, figures_dir / FIGURE_FILENAMES["domain_bias"])
    plot_gleason_heatmap(gleason_table, figures_dir / FIGURE_FILENAMES["gleason_heatmap"])
    plot_majority_voting_matrices(
        majority_confusion,
        figures_dir / FIGURE_FILENAMES["majority_voting"],
    )

    table_paths = export_tables(
        tables_dir=tables_dir,
        long_df=long_df,
        scoreboard=scoreboard,
        rankings=rankings,
        domain_bias=domain_bias,
        gleason_table=gleason_table,
        best_models=best_models,
    )
    figure_paths = [figures_dir / filename for filename in FIGURE_FILENAMES.values()]
    print_final_summary(
        figure_paths=figure_paths,
        table_paths=table_paths,
        models=models,
        datasets=datasets,
    )


if __name__ == "__main__":
    main()
