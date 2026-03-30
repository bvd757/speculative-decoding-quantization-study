from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _requested_device(config: Dict[str, Any]) -> str:
    device = str(config.get("device", "cuda")).lower()
    if device.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _preferred_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_model_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for value in hf_device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError(f"Tokenizer for {model_name} does not define eos_token or unk_token.")
    tokenizer.padding_side = "left"
    return tokenizer


def _build_quantization_config(precision: str, compute_dtype: torch.dtype):
    value = precision.lower()
    if value in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        return None
    if value in {"int8", "8bit", "8-bit"}:
        return BitsAndBytesConfig(load_in_8bit=True)
    if value in {"int4", "4bit", "4-bit"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    raise ValueError(f"Unsupported precision: {precision}")


def _build_torch_dtype(precision: str, device: str) -> torch.dtype:
    value = precision.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if value in {"fp16", "float16"}:
        return torch.float16 if device == "cuda" else torch.float32
    if value in {"fp32", "float32"}:
        return torch.float32
    return _preferred_dtype(device)


def _load_model(model_name: str, precision: str, config: Dict[str, Any]):
    device = _requested_device(config)
    trust_remote_code = bool(config.get("trust_remote_code", False))
    compute_dtype = _preferred_dtype(device)
    quantization_config = _build_quantization_config(precision, compute_dtype)

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        if device == "cuda":
            load_kwargs["device_map"] = config.get("device_map", "auto")
        if precision.lower() in {"int8", "8bit", "8-bit"}:
            load_kwargs["dtype"] = torch.float16
    else:
        load_kwargs["dtype"] = _build_torch_dtype(precision, device)
        if device == "cuda":
            load_kwargs["device_map"] = config.get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    if device == "cpu" and quantization_config is None:
        model = model.to("cpu")

    model.eval()
    model.config.use_cache = True
    return model


def _finalize_generation_config(model, tokenizer) -> None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50


def load_target_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    model_name = config["target_model"]
    precision = str(config.get("target_precision", "bf16"))
    tokenizer = load_tokenizer(model_name, trust_remote_code=bool(config.get("trust_remote_code", False)))
    model = _load_model(model_name, precision, config)
    _finalize_generation_config(model, tokenizer)
    return model, tokenizer


def load_draft_model(config: Dict[str, Any], precision: str):
    model_name = config["draft_model"]
    tokenizer = load_tokenizer(model_name, trust_remote_code=bool(config.get("trust_remote_code", False)))
    model = _load_model(model_name, precision, config)
    _finalize_generation_config(model, tokenizer)
    return model, tokenizer
