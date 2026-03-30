from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def _activities() -> List[ProfilerActivity]:
    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def _default_sort_key() -> str:
    if torch.cuda.is_available():
        return "self_cuda_time_total"
    return "self_cpu_time_total"


def profile_generation(
    step_fn: Callable[[int], Dict[str, Any]],
    num_steps: int,
    trace_path: str | Path | None = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
    row_limit: int = 30,
    sort_by: Optional[str] = None,
) -> Dict[str, Any]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    trace_file = Path(trace_path) if trace_path is not None else None
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)

    sort_key = sort_by or _default_sort_key()
    step_results: List[Dict[str, Any]] = []

    with profile(
        activities=_activities(),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        acc_events=True,
    ) as prof:
        for step_idx in range(num_steps):
            with record_function("generation_step"):
                step_results.append(step_fn(step_idx))
            prof.step()

    if trace_file is not None:
        prof.export_chrome_trace(str(trace_file))

    profiler_table = prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)

    return {
        "step_results": step_results,
        "profiler_table": profiler_table,
        "trace_path": str(trace_file) if trace_file is not None else None,
        "sort_by": sort_key,
        "row_limit": row_limit,
    }
