from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import build_summary_dataframe


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_table(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "summary_table.csv"
    df.drop(columns=["precision_order"], errors="ignore").to_csv(table_path, index=False)
    return table_path


def _resolve_single_file(candidates: Iterable[Path], kind: str, search_dir: Path) -> Path:
    files = sorted({path.resolve() for path in candidates})

    if len(files) == 0:
        raise FileNotFoundError(
            f"Could not find {kind} result file in {search_dir}."
        )

    if len(files) > 1:
        joined = "\n".join(str(path) for path in files)
        raise ValueError(
            f"Expected exactly one {kind} result file in {search_dir}, but found {len(files)}:\n{joined}"
        )

    return files[0]


def autodiscover_result_files(
    baseline: Optional[str],
    speculative: Optional[str],
) -> tuple[Path, Path]:
    runs_dir = ROOT / "results" / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {runs_dir}")

    baseline_path = Path(baseline) if baseline else _resolve_single_file(
        runs_dir.glob("baseline_*.json"),
        kind="baseline",
        search_dir=runs_dir,
    )

    speculative_path = Path(speculative) if speculative else _resolve_single_file(
        list(runs_dir.glob("speculative_*.json")) + list(runs_dir.glob("prompt_lookup_*.json")),
        kind="speculative/prompt_lookup",
        search_dir=runs_dir,
    )

    return baseline_path, speculative_path


def plot_ms_per_token(df: pd.DataFrame, output_dir: Path) -> Path:
    chart_df = df[df["mean_ms_per_token"].notna()].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(chart_df["precision"], chart_df["mean_ms_per_token"])
    plt.xlabel("Draft precision")
    plt.ylabel("Mean ms/token")
    plt.title("Latency per token")
    plt.tight_layout()

    path = output_dir / "latency_ms_per_token.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_acceptance_rate(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    chart_df = df[(df["mode"] == "speculative") & (df["mean_acceptance_rate"].notna())].copy()
    if len(chart_df) == 0:
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(chart_df["precision"], chart_df["mean_acceptance_rate"])
    plt.xlabel("Draft precision")
    plt.ylabel("Mean acceptance rate")
    plt.title("Acceptance rate by draft precision")
    plt.tight_layout()

    path = output_dir / "acceptance_rate.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_acceptance_length(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    chart_df = df[(df["mode"] == "speculative") & (df["mean_acceptance_length"].notna())].copy()
    if len(chart_df) == 0:
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(chart_df["precision"], chart_df["mean_acceptance_length"])
    plt.xlabel("Draft precision")
    plt.ylabel("Mean acceptance length")
    plt.title("Acceptance length by draft precision")
    plt.tight_layout()

    path = output_dir / "acceptance_length.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_vram(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    chart_df = df[df["max_peak_runtime_vram_gb"].notna()].copy()
    if len(chart_df) == 0:
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(chart_df["precision"], chart_df["max_peak_runtime_vram_gb"])
    plt.xlabel("Draft precision")
    plt.ylabel("Peak VRAM (GB)")
    plt.title("Peak runtime VRAM")
    plt.tight_layout()

    path = output_dir / "peak_vram_gb.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_speedup(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    chart_df = df[
        (df["precision"] != "baseline") &
        (df["speedup_vs_baseline"].notna())
    ].copy()
    if len(chart_df) == 0:
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(chart_df["precision"], chart_df["speedup_vs_baseline"])
    plt.xlabel("Method")
    plt.ylabel("Speedup vs baseline")
    plt.title("Speedup over baseline")
    plt.tight_layout()

    path = output_dir / "speedup_vs_baseline.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_draft_vs_target_time(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    chart_df = df[
        (df["mode"] == "speculative")
        & (df["mean_draft_time_ms"].notna())
        & (df["mean_target_time_ms"].notna())
    ].copy()
    if len(chart_df) == 0:
        return None

    x = range(len(chart_df))

    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], chart_df["mean_draft_time_ms"], width=0.4, label="Draft")
    plt.bar([i + 0.2 for i in x], chart_df["mean_target_time_ms"], width=0.4, label="Target")
    plt.xticks(list(x), chart_df["precision"].tolist())
    plt.xlabel("Draft precision")
    plt.ylabel("Mean time (ms)")
    plt.title("Draft vs target time")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "draft_vs_target_time.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speculative")
    parser.add_argument("--baseline")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "figures"))
    args = parser.parse_args()

    baseline_path, speculative_path = autodiscover_result_files(
        baseline=args.baseline,
        speculative=args.speculative,
    )

    comparison_payload = load_json(speculative_path)
    baseline_payload = load_json(baseline_path)

    if "summary_by_precision" in comparison_payload:
        summary_by_precision = comparison_payload["summary_by_precision"]
    elif "summary" in comparison_payload:
        summary = dict(comparison_payload["summary"])
        mode = summary.get("mode", "comparison")

        if mode == "prompt_lookup":
            label = "prompt_lookup"
        elif mode == "baseline":
            label = "comparison"
        else:
            label = mode

        summary["precision"] = label
        summary_by_precision = {label: summary}
    else:
        raise ValueError("Unsupported result file format: expected summary_by_precision or summary.")

    df = build_summary_dataframe(
        baseline_summary=baseline_payload["summary"] if baseline_payload is not None else None,
        summary_by_precision=summary_by_precision,
    )

    if len(df) == 0:
        raise ValueError("No rows available for plotting.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = save_table(df, output_dir)

    figure_paths = [
        plot_ms_per_token(df, output_dir),
        plot_acceptance_rate(df, output_dir),
        plot_acceptance_length(df, output_dir),
        plot_vram(df, output_dir),
        plot_speedup(df, output_dir),
        plot_draft_vs_target_time(df, output_dir),
    ]
    figure_paths = [path for path in figure_paths if path is not None]

    print()
    print(df.drop(columns=["precision_order"], errors="ignore").to_string(index=False))
    print()
    print(f"Saved table to {table_path}")
    for path in figure_paths:
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
