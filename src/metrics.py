from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _safe_values(records: Iterable[Dict[str, Any]], key: str) -> List[Any]:
    values = []
    for record in records:
        value = record.get(key)
        if value is not None:
            values.append(value)
    return values


def _safe_mean(records: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = _safe_values(records, key)
    if not values:
        return None
    return float(mean(values))


def _safe_max(records: Iterable[Dict[str, Any]], key: str) -> Optional[Any]:
    values = _safe_values(records, key)
    if not values:
        return None
    return max(values)


def _bytes_to_gb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024 ** 3)


def detect_mode(records: List[Dict[str, Any]]) -> str:
    if not records:
        raise ValueError("records must be non-empty")
    mode = records[0].get("mode")
    if mode not in {"baseline", "speculative"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("records must be non-empty")

    mode = detect_mode(records)

    summary: Dict[str, Any] = {
        "mode": mode,
        "num_samples": len(records),
        "mean_prompt_token_count": _safe_mean(records, "prompt_token_count"),
        "mean_generated_token_count": _safe_mean(records, "generated_token_count"),
        "mean_wall_time_ms": _safe_mean(records, "wall_time_ms"),
        "mean_ms_per_token": _safe_mean(records, "ms_per_token"),
        "max_resident_vram_bytes": _safe_max(records, "resident_vram_bytes"),
        "max_peak_runtime_vram_bytes": _safe_max(records, "peak_runtime_vram_bytes"),
    }

    if mode == "speculative":
        summary.update(
            {
                "mean_draft_time_ms": _safe_mean(records, "draft_time_ms"),
                "mean_target_time_ms": _safe_mean(records, "target_time_ms"),
                "mean_acceptance_rate": _safe_mean(records, "acceptance_rate"),
                "mean_acceptance_length": _safe_mean(records, "mean_acceptance_length"),
                "mean_proposed_token_count": _safe_mean(records, "proposed_token_count"),
                "mean_accepted_token_count": _safe_mean(records, "accepted_token_count"),
                "mean_iterations": _safe_mean(records, "iterations"),
            }
        )

    return summary


def compute_metrics_by_precision(results_by_precision: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for precision, records in results_by_precision.items():
        summary = compute_metrics(records)
        summary["precision"] = precision
        output[precision] = summary
    return output


def detect_mode(records: List[Dict[str, Any]]) -> str:
    if not records:
        raise ValueError("records must be non-empty")
    mode = records[0].get("mode")
    if mode not in {"baseline", "speculative", "prompt_lookup"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode


def build_summary_dataframe(
    baseline_summary: Optional[Dict[str, Any]] = None,
    summary_by_precision: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if baseline_summary is not None:
        rows.append(
            {
                "mode": baseline_summary["mode"],
                "precision": "baseline",
                "num_samples": baseline_summary["num_samples"],
                "mean_prompt_token_count": baseline_summary.get("mean_prompt_token_count"),
                "mean_generated_token_count": baseline_summary.get("mean_generated_token_count"),
                "mean_wall_time_ms": baseline_summary.get("mean_wall_time_ms"),
                "mean_ms_per_token": baseline_summary.get("mean_ms_per_token"),
                "mean_draft_time_ms": baseline_summary.get("mean_draft_time_ms"),
                "mean_target_time_ms": baseline_summary.get("mean_target_time_ms"),
                "mean_acceptance_rate": baseline_summary.get("mean_acceptance_rate"),
                "mean_acceptance_length": baseline_summary.get("mean_acceptance_length"),
                "mean_proposed_token_count": baseline_summary.get("mean_proposed_token_count"),
                "mean_accepted_token_count": baseline_summary.get("mean_accepted_token_count"),
                "mean_iterations": baseline_summary.get("mean_iterations"),
                "max_resident_vram_bytes": baseline_summary.get("max_resident_vram_bytes"),
                "max_peak_runtime_vram_bytes": baseline_summary.get("max_peak_runtime_vram_bytes"),
                "max_resident_vram_gb": _bytes_to_gb(baseline_summary.get("max_resident_vram_bytes")),
                "max_peak_runtime_vram_gb": _bytes_to_gb(baseline_summary.get("max_peak_runtime_vram_bytes")),
            }
        )

    if summary_by_precision is not None:
        for precision, summary in summary_by_precision.items():
            rows.append(
                {
                    "mode": summary["mode"],
                    "precision": precision,
                    "num_samples": summary["num_samples"],
                    "mean_prompt_token_count": summary.get("mean_prompt_token_count"),
                    "mean_generated_token_count": summary.get("mean_generated_token_count"),
                    "mean_wall_time_ms": summary.get("mean_wall_time_ms"),
                    "mean_ms_per_token": summary.get("mean_ms_per_token"),
                    "mean_draft_time_ms": summary.get("mean_draft_time_ms"),
                    "mean_target_time_ms": summary.get("mean_target_time_ms"),
                    "mean_acceptance_rate": summary.get("mean_acceptance_rate"),
                    "mean_acceptance_length": summary.get("mean_acceptance_length"),
                    "mean_proposed_token_count": summary.get("mean_proposed_token_count"),
                    "mean_accepted_token_count": summary.get("mean_accepted_token_count"),
                    "mean_iterations": summary.get("mean_iterations"),
                    "max_resident_vram_bytes": summary.get("max_resident_vram_bytes"),
                    "max_peak_runtime_vram_bytes": summary.get("max_peak_runtime_vram_bytes"),
                    "max_resident_vram_gb": _bytes_to_gb(summary.get("max_resident_vram_bytes")),
                    "max_peak_runtime_vram_gb": _bytes_to_gb(summary.get("max_peak_runtime_vram_bytes")),
                }
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    precision_order = {"baseline": 0, "bf16": 1, "int8": 2, "int4": 3}
    df["precision_order"] = df["precision"].map(lambda x: precision_order.get(str(x).lower(), 999))
    df = df.sort_values(["precision_order", "precision"]).reset_index(drop=True)

    baseline_rows = df[df["precision"] == "baseline"]
    if len(baseline_rows) > 0:
        baseline_ms_per_token = float(baseline_rows.iloc[0]["mean_ms_per_token"])
        df["speedup_vs_baseline"] = df["mean_ms_per_token"].apply(
            lambda x: baseline_ms_per_token / x if pd.notna(x) and x not in (0, None) else None
        )
    else:
        df["speedup_vs_baseline"] = None

    return df
