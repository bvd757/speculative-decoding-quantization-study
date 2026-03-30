from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_subset
from src.metrics import compute_metrics
from src.models import load_draft_model, load_target_model
from src.profiler import profile_generation
from src.speculative import baseline_generate, speculative_generate


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_run_dir(output_dir: str | Path, mode: str, draft_precision: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{draft_precision}" if draft_precision is not None else ""
    run_dir = Path(output_dir) / f"{mode}{suffix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenizer_compatibility_check(target_tokenizer, draft_tokenizer) -> None:
    if target_tokenizer.vocab_size != draft_tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab mismatch: target={target_tokenizer.vocab_size}, draft={draft_tokenizer.vocab_size}"
        )

    probe = "Speculative decoding tokenizer compatibility check."
    target_ids = target_tokenizer.encode(probe, add_special_tokens=False)
    draft_ids = draft_tokenizer.encode(probe, add_special_tokens=False)
    if target_ids != draft_ids:
        raise ValueError("Target and draft tokenizers encode text differently.")

    if target_tokenizer.eos_token_id != draft_tokenizer.eos_token_id:
        raise ValueError(
            f"eos_token_id mismatch: target={target_tokenizer.eos_token_id}, draft={draft_tokenizer.eos_token_id}"
        )


def load_prompts(args, config: Dict[str, Any]) -> List[Any]:
    if args.prompt is not None:
        return [args.prompt]

    config_for_data = dict(config)
    if args.input_jsonl is not None:
        config_for_data["prompt_path"] = args.input_jsonl

    return load_dataset_subset(
        config=config_for_data,
        max_samples=args.max_samples,
    )


def maybe_warmup(
    mode: str,
    prompts: List[Any],
    warmup_runs: int,
    config: Dict[str, Any],
    target_model,
    tokenizer,
    draft_model=None,
) -> None:
    if warmup_runs <= 0:
        return

    warmup_count = min(warmup_runs, len(prompts))
    for prompt in prompts[:warmup_count]:
        if mode == "baseline":
            baseline_generate(
                model=target_model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
            )
        else:
            speculative_generate(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["baseline", "speculative"], default="speculative")
    parser.add_argument("--draft-precision", default="bf16")
    parser.add_argument("--prompt")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "profiles"))
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = load_prompts(args, config)
    run_dir = build_run_dir(
        output_dir=args.output_dir,
        mode=args.mode,
        draft_precision=args.draft_precision if args.mode == "speculative" else None,
    )

    target_model, tokenizer = load_target_model(config)

    draft_model = None
    if args.mode == "speculative":
        draft_model, draft_tokenizer = load_draft_model(config, args.draft_precision)
        tokenizer_compatibility_check(tokenizer, draft_tokenizer)
        del draft_tokenizer
        gc.collect()

    maybe_warmup(
        mode=args.mode,
        prompts=prompts,
        warmup_runs=args.warmup_runs,
        config=config,
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
    )

    def step_fn(step_idx: int) -> Dict[str, Any]:
        prompt = prompts[step_idx]
        if args.mode == "baseline":
            result = baseline_generate(
                model=target_model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
            )
        else:
            result = speculative_generate(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
            )
            result["draft_precision"] = args.draft_precision

        result["sample_index"] = step_idx + 1
        return result

    profile_payload = profile_generation(
        step_fn=step_fn,
        num_steps=len(prompts),
        trace_path=run_dir / "trace.json",
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
        row_limit=args.row_limit,
    )

    results = profile_payload["step_results"]
    summary = compute_metrics(results)

    payload = {
        "config_path": args.config,
        "mode": args.mode,
        "target_model": config["target_model"],
        "draft_model": config.get("draft_model") if args.mode == "speculative" else None,
        "draft_precision": args.draft_precision if args.mode == "speculative" else None,
        "num_profiled_samples": len(results),
        "summary": summary,
        "results": results,
        "trace_path": profile_payload["trace_path"],
        "profiler_table_path": str(run_dir / "profiler_table.txt"),
        "sort_by": profile_payload["sort_by"],
        "row_limit": profile_payload["row_limit"],
    }

    save_json(run_dir / "profile_results.json", payload)
    save_text(run_dir / "profiler_table.txt", profile_payload["profiler_table"])

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"Saved profile JSON to {run_dir / 'profile_results.json'}")
    print(f"Saved profiler table to {run_dir / 'profiler_table.txt'}")
    print(f"Saved trace to {run_dir / 'trace.json'}")

    if draft_model is not None:
        unload_model(draft_model)
    unload_model(target_model)
    del tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
