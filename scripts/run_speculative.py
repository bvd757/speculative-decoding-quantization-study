from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_subset
from src.metrics import compute_metrics
from src.models import load_draft_model, load_target_model
from src.speculative import speculative_generate


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_output_path(args) -> Path:
    if args.output is not None:
        return Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "runs" / f"speculative_{timestamp}.json"


def save_results(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


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


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--precisions", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        config_for_data = dict(config)
        if args.input_jsonl is not None:
            config_for_data["prompt_path"] = args.input_jsonl
        prompts = load_dataset_subset(config_for_data, max_samples=args.max_samples)

    precisions = args.precisions
    if not precisions:
        precisions = list(config.get("draft_precisions", ["bf16", "int8", "int4"]))

    target_model, target_tokenizer = load_target_model(config)

    all_results: Dict[str, Any] = {}
    precision_summaries: Dict[str, Any] = {}

    for precision in precisions:
        print()
        print(f"=== Running draft precision: {precision} ===")

        draft_model, draft_tokenizer = load_draft_model(config, precision)
        tokenizer_compatibility_check(target_tokenizer, draft_tokenizer)

        precision_results: List[Dict[str, Any]] = []

        for index, prompt in enumerate(prompts, start=1):
            result = speculative_generate(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tokenizer,
                prompt=prompt,
                config=config,
            )
            result["sample_index"] = index
            result["draft_precision"] = precision
            precision_results.append(result)

            print(
                f"[{index}/{len(prompts)}] "
                f"tokens={result['generated_token_count']} "
                f"wall_ms={result['wall_time_ms']:.2f} "
                f"ms_per_token={result['ms_per_token']:.2f} "
                f"accept_rate={result['acceptance_rate']:.4f} "
                f"accept_len={result['mean_acceptance_length']:.2f}"
            )

        summary = compute_metrics(precision_results)
        summary["precision"] = precision
        all_results[precision] = precision_results
        precision_summaries[precision] = summary

        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        unload_model(draft_model)
        del draft_tokenizer
        gc.collect()

    payload = {
        "config_path": args.config,
        "target_model": config["target_model"],
        "draft_model": config["draft_model"],
        "draft_precisions": precisions,
        "results_by_precision": all_results,
        "summary_by_precision": precision_summaries,
    }

    output_path = build_output_path(args)
    save_results(payload, output_path)

    print()
    print(f"Saved to {output_path}")

    unload_model(target_model)
    del target_tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
