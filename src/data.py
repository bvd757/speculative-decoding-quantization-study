from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_record(payload: Dict[str, Any]) -> Any:
    for key in ("messages", "prompt", "text", "input", "question"):
        if key in payload:
            return payload[key]
    raise ValueError("Each record must contain one of: messages, prompt, text, input, question.")


def _load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Any]:
    prompts: List[Any] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            prompts.append(_normalize_record(payload))
            if max_samples is not None and len(prompts) >= max_samples:
                break
    return prompts


def _load_json(path: Path, max_samples: Optional[int] = None) -> List[Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        if len(payload) == 0:
            return []

        if isinstance(payload[0], dict):
            prompts = [_normalize_record(item) for item in payload]
        else:
            prompts = payload
    elif isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            records = payload["records"]
            if len(records) > 0 and isinstance(records[0], dict):
                prompts = [_normalize_record(item) for item in records]
            else:
                prompts = records
        else:
            prompts = [_normalize_record(payload)]
    else:
        raise ValueError("Unsupported JSON structure.")

    if max_samples is not None:
        prompts = prompts[:max_samples]
    return prompts


def _load_txt(path: Path, max_samples: Optional[int] = None) -> List[str]:
    prompts: List[str] = []
    current_block: List[str] = []

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                if current_block:
                    prompts.append("\n".join(current_block).strip())
                    current_block = []
                    if max_samples is not None and len(prompts) >= max_samples:
                        break
            else:
                current_block.append(line)

    if current_block and (max_samples is None or len(prompts) < max_samples):
        prompts.append("\n".join(current_block).strip())

    return prompts


def load_prompts_from_path(path: str | Path, max_samples: Optional[int] = None) -> List[Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        prompts = _load_jsonl(file_path, max_samples=max_samples)
    elif suffix == ".json":
        prompts = _load_json(file_path, max_samples=max_samples)
    elif suffix in {".txt", ".md"}:
        prompts = _load_txt(file_path, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported prompt file format: {suffix}")

    if not prompts:
        raise ValueError(f"No prompts were loaded from {file_path}")

    return prompts


def load_dataset_subset(config: Dict[str, Any], max_samples: Optional[int] = None, split: Optional[str] = None) -> List[Any]:
    if max_samples is None:
        max_samples = config.get("max_samples")

    if "prompts" in config:
        prompts = config["prompts"]
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("config['prompts'] must be a non-empty list.")
        if max_samples is not None:
            prompts = prompts[:max_samples]
        return prompts

    prompt_path = config.get("prompt_path")
    if prompt_path is not None:
        return load_prompts_from_path(prompt_path, max_samples=max_samples)

    dataset_name = str(config.get("dataset_name", "")).lower()
    if dataset_name == "specbench":
        specbench_path = config.get("specbench_path")
        if specbench_path is None:
            raise ValueError(
                "dataset_name='specbench' is set, but no local prompt file is provided. "
                "Set config['specbench_path'] or config['prompt_path'] to a local .jsonl/.json/.txt file."
            )
        return load_prompts_from_path(specbench_path, max_samples=max_samples)

    raise ValueError(
        "No data source configured. Use one of: "
        "config['prompts'], config['prompt_path'], or config['specbench_path']."
    )
