from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_subset
from src.metrics import compute_metrics
from src.models import load_target_model
from src.speculative import prompt_lookup_generate


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_output_path(args) -> Path:
    if args.output is not None:
        return Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "runs" / f"prompt_lookup_{timestamp}.json"


def save_results(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--prompt-lookup-num-tokens", type=int, default=4)
    parser.add_argument("--max-matching-ngram-size", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    config = dict(config)
    config["generation"] = dict(config.get("generation", {}))
    config["generation"]["prompt_lookup_num_tokens"] = args.prompt_lookup_num_tokens
    config["generation"]["max_matching_ngram_size"] = args.max_matching_ngram_size

    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        config_for_data = dict(config)
        if args.input_jsonl is not None:
            config_for_data["prompt_path"] = args.input_jsonl
        prompts = load_dataset_subset(config_for_data, max_samples=args.max_samples)

    model, tokenizer = load_target_model(config)

    results: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        result = prompt_lookup_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
        )
        result["sample_index"] = index
        results.append(result)
        print(
            f"[{index}/{len(prompts)}] "
            f"generated_tokens={result['generated_token_count']} "
            f"wall_time_ms={result['wall_time_ms']:.2f} "
            f"ms_per_token={result['ms_per_token']:.2f}"
        )

    payload = {
        "config_path": args.config,
        "target_model": config["target_model"],
        "results": results,
        "summary": compute_metrics(results),
    }

    output_path = build_output_path(args)
    save_results(payload, output_path)

    print()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
