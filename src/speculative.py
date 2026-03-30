from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Sequence, Tuple
import inspect
from dataclasses import dataclass
from transformers import DynamicCache

import torch
from torch.profiler import record_function

from src.models import get_model_device


@dataclass
class CausalState:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    cache: DynamicCache
    next_logits: torch.Tensor
    device: torch.device


def _model_forward(model, input_ids, attention_mask, cache, last_token_only: bool):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": cache,
        "use_cache": True,
    }

    if last_token_only:
        try:
            kwargs["logits_to_keep"] = 1
        except Exception:
            pass

    with torch.inference_mode():
        return model(**kwargs)


def _prefill_state(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> CausalState:
    device = get_model_device(model)
    cache = DynamicCache(config=model.config)

    model_input_ids = input_ids.to(device)
    model_attention_mask = attention_mask.to(device)

    outputs = _model_forward(
        model=model,
        input_ids=model_input_ids,
        attention_mask=model_attention_mask,
        cache=cache,
        last_token_only=True,
    )

    return CausalState(
        input_ids=input_ids.detach().cpu(),
        attention_mask=attention_mask.detach().cpu(),
        cache=cache,
        next_logits=outputs.logits[:, -1, :].detach(),
        device=device,
    )


def _advance_state(model, state: CausalState, new_token_ids: torch.Tensor) -> CausalState:
    if new_token_ids.ndim != 2:
        raise ValueError("new_token_ids must have shape [batch, seq_len]")

    if new_token_ids.shape[1] == 0:
        return state

    device_new_ids = new_token_ids.to(state.device)
    old_attention_mask = state.attention_mask.to(state.device)
    new_attention_mask = torch.cat(
        [old_attention_mask, old_attention_mask.new_ones((old_attention_mask.shape[0], new_token_ids.shape[1]))],
        dim=-1,
    )

    outputs = _model_forward(
        model=model,
        input_ids=device_new_ids,
        attention_mask=new_attention_mask,
        cache=state.cache,
        last_token_only=True,
    )

    return CausalState(
        input_ids=torch.cat([state.input_ids, new_token_ids.detach().cpu()], dim=-1),
        attention_mask=new_attention_mask.detach().cpu(),
        cache=state.cache,
        next_logits=outputs.logits[:, -1, :].detach(),
        device=state.device,
    )


def prompt_lookup_generate(model, tokenizer, prompt: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    use_chat_template = bool(config.get("use_chat_template", True))
    generation_config = config.get("generation", {})
    max_new_tokens = int(generation_config.get("max_new_tokens", 128))
    prompt_lookup_num_tokens = int(generation_config.get("prompt_lookup_num_tokens", 4))
    max_matching_ngram_size = int(generation_config.get("max_matching_ngram_size", 2))

    encoded = prepare_inputs(tokenizer, prompt, use_chat_template=use_chat_template)
    prompt_ids = encoded["input_ids"].cpu()

    model_device = get_model_device(model)
    model_inputs = {key: value.to(model_device) for key, value in encoded.items()}

    resident_vram_bytes = None
    if model_device.type == "cuda" and torch.cuda.is_available():
        resident_vram_bytes = int(torch.cuda.memory_allocated(model_device))
        torch.cuda.reset_peak_memory_stats(model_device)

    _sync(model_device)
    started_at = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prompt_lookup_num_tokens=prompt_lookup_num_tokens,
            max_matching_ngram_size=max_matching_ngram_size,
        )
    _sync(model_device)
    finished_at = time.perf_counter()

    sequences = outputs.sequences.detach().cpu()
    generated_ids = sequences[:, prompt_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(sequences[0], skip_special_tokens=True)

    generated_token_count = int(generated_ids.shape[1])
    wall_time_ms = (finished_at - started_at) * 1000.0
    ms_per_token = wall_time_ms / max(generated_token_count, 1)

    peak_runtime_vram_bytes = None
    if model_device.type == "cuda" and torch.cuda.is_available():
        peak_runtime_vram_bytes = int(torch.cuda.max_memory_allocated(model_device))

    return {
        "mode": "prompt_lookup",
        "prompt": prompt,
        "prompt_token_count": int(prompt_ids.shape[1]),
        "generated_token_count": generated_token_count,
        "generated_token_ids": generated_ids[0].tolist(),
        "generated_text": generated_text,
        "full_text": full_text,
        "prompt_lookup_num_tokens": prompt_lookup_num_tokens,
        "max_matching_ngram_size": max_matching_ngram_size,
        "wall_time_ms": wall_time_ms,
        "ms_per_token": ms_per_token,
        "resident_vram_bytes": resident_vram_bytes,
        "peak_runtime_vram_bytes": peak_runtime_vram_bytes,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def prepare_inputs(tokenizer, prompt: Any, use_chat_template: bool = True) -> Dict[str, torch.Tensor]:
    if isinstance(prompt, str):
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = prompt
    elif isinstance(prompt, Sequence):
        if not use_chat_template or not getattr(tokenizer, "chat_template", None):
            raise ValueError("A list-like prompt requires a tokenizer chat_template.")
        rendered = tokenizer.apply_chat_template(
            list(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        raise TypeError("Prompt must be a string or a chat message list.")

    encoded = tokenizer(rendered, return_tensors="pt")

    if encoded["input_ids"].shape[1] == 0:
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        if bos_token_id is None:
            raise ValueError("Prompt is empty and tokenizer does not define bos_token_id or eos_token_id.")
        encoded = {
            "input_ids": torch.tensor([[bos_token_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    return encoded


def _generation_kwargs(tokenizer, max_new_tokens: int) -> Dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _call_generate(model, **kwargs):
    generate_fn = model.generate
    try:
        signature = inspect.signature(generate_fn)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
    except (TypeError, ValueError):
        filtered_kwargs = kwargs

    return generate_fn(**filtered_kwargs)


def baseline_generate(model, tokenizer, prompt: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    use_chat_template = bool(config.get("use_chat_template", True))
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 128))

    encoded = prepare_inputs(tokenizer, prompt, use_chat_template=use_chat_template)
    prompt_ids = encoded["input_ids"].cpu()
    model_device = get_model_device(model)
    model_inputs = {key: value.to(model_device) for key, value in encoded.items()}

    resident_vram_bytes = None
    if model_device.type == "cuda" and torch.cuda.is_available():
        resident_vram_bytes = int(torch.cuda.memory_allocated(model_device))
        torch.cuda.reset_peak_memory_stats(model_device)

    _sync(model_device)
    started_at = time.perf_counter()
    with torch.inference_mode():
        outputs = _call_generate(
            model,
            **model_inputs,
            **_generation_kwargs(tokenizer, max_new_tokens=max_new_tokens),
        )
    _sync(model_device)
    finished_at = time.perf_counter()

    sequences = outputs.sequences.detach().cpu()
    generated_ids = sequences[:, prompt_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(sequences[0], skip_special_tokens=True)

    generated_token_count = int(generated_ids.shape[1])
    wall_time_ms = (finished_at - started_at) * 1000.0
    ms_per_token = wall_time_ms / max(generated_token_count, 1)

    peak_runtime_vram_bytes = None
    if model_device.type == "cuda" and torch.cuda.is_available():
        peak_runtime_vram_bytes = int(torch.cuda.max_memory_allocated(model_device))

    return {
        "mode": "baseline",
        "prompt": prompt,
        "prompt_token_count": int(prompt_ids.shape[1]),
        "generated_token_count": generated_token_count,
        "generated_token_ids": generated_ids[0].tolist(),
        "generated_text": generated_text,
        "full_text": full_text,
        "wall_time_ms": wall_time_ms,
        "ms_per_token": ms_per_token,
        "resident_vram_bytes": resident_vram_bytes,
        "peak_runtime_vram_bytes": peak_runtime_vram_bytes,
    }


def _generate_draft_block_cached(draft_model, state: CausalState, max_new_tokens: int):
    base_len = state.cache.get_seq_length()
    running_attention_mask = state.attention_mask.to(state.device)
    current_logits = state.next_logits
    proposed_tokens: List[int] = []

    _sync(state.device)
    started_at = time.perf_counter()

    with record_function("draft_generate_block"):
        for _ in range(max_new_tokens):
            next_token = current_logits.argmax(dim=-1, keepdim=True)
            proposed_tokens.append(int(next_token[0, 0].item()))

            running_attention_mask = torch.cat(
                [running_attention_mask, running_attention_mask.new_ones((1, 1))],
                dim=-1,
            )

            outputs = _model_forward(
                model=draft_model,
                input_ids=next_token.to(state.device),
                attention_mask=running_attention_mask,
                cache=state.cache,
                last_token_only=True,
            )
            current_logits = outputs.logits[:, -1, :].detach()

    state.cache.crop(base_len)

    _sync(state.device)
    finished_at = time.perf_counter()

    draft_ids = torch.tensor([proposed_tokens], dtype=torch.long)
    return draft_ids, (finished_at - started_at) * 1000.0


def _verify_with_target_cached(target_model, state: CausalState, draft_ids: torch.Tensor):
    if draft_ids.shape[1] == 0:
        raise ValueError("draft_ids must be non-empty")

    base_len = state.cache.get_seq_length()

    draft_ids_device = draft_ids.to(state.device)
    running_attention_mask = torch.cat(
        [
            state.attention_mask.to(state.device),
            state.attention_mask.new_ones((state.attention_mask.shape[0], draft_ids.shape[1])).to(state.device),
        ],
        dim=-1,
    )

    _sync(state.device)
    started_at = time.perf_counter()

    with record_function("target_verify_block"):
        with torch.inference_mode():
            outputs = target_model(
                input_ids=draft_ids_device,
                attention_mask=running_attention_mask,
                past_key_values=state.cache,
                use_cache=True,
            )

    verify_logits = outputs.logits.detach().cpu()
    state.cache.crop(base_len)

    _sync(state.device)
    finished_at = time.perf_counter()

    predictions: List[int] = [int(state.next_logits.argmax(dim=-1).item())]
    predictions.extend(int(verify_logits[0, i].argmax(dim=-1).item()) for i in range(draft_ids.shape[1]))

    proposed = draft_ids[0].tolist()

    accepted_len = 0
    for index, token_id in enumerate(proposed):
        if predictions[index] != token_id:
            break
        accepted_len += 1

    correction_token = torch.tensor([[predictions[accepted_len]]], dtype=torch.long)

    return accepted_len, correction_token, (finished_at - started_at) * 1000.0


def speculative_generate(target_model, draft_model, tokenizer, prompt: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    use_chat_template = bool(config.get("use_chat_template", True))
    generation_config = config.get("generation", {})
    max_new_tokens = int(generation_config.get("max_new_tokens", 128))
    speculative_k = int(generation_config.get("speculative_k", 4))

    encoded = prepare_inputs(tokenizer, prompt, use_chat_template=use_chat_template)
    prompt_ids = encoded["input_ids"].cpu()
    prompt_attention_mask = encoded["attention_mask"].cpu()

    target_state = _prefill_state(target_model, prompt_ids, prompt_attention_mask)
    draft_state = _prefill_state(draft_model, prompt_ids, prompt_attention_mask)

    eos_token_id = tokenizer.eos_token_id
    generated_ids: List[int] = []
    accepted_lengths: List[int] = []
    proposed_token_count = 0
    accepted_token_count = 0
    draft_time_ms = 0.0
    target_time_ms = 0.0
    iterations = 0

    resident_vram_bytes = None
    if target_state.device.type == "cuda" and torch.cuda.is_available():
        resident_vram_bytes = int(torch.cuda.memory_allocated(target_state.device))
        torch.cuda.reset_peak_memory_stats(target_state.device)

    _sync(target_state.device)
    wall_started_at = time.perf_counter()

    while len(generated_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_ids)
        draft_budget = min(speculative_k, remaining)

        draft_block, current_draft_time_ms = _generate_draft_block_cached(
            draft_model=draft_model,
            state=draft_state,
            max_new_tokens=draft_budget,
        )
        draft_time_ms += current_draft_time_ms

        accepted_len, correction_token, current_target_time_ms = _verify_with_target_cached(
            target_model=target_model,
            state=target_state,
            draft_ids=draft_block,
        )
        target_time_ms += current_target_time_ms

        iterations += 1
        proposed_token_count += int(draft_block.shape[1])
        accepted_token_count += accepted_len
        accepted_lengths.append(accepted_len)

        accepted_prefix = draft_block[:, :accepted_len]
        commit_tokens = torch.cat([accepted_prefix, correction_token], dim=-1)

        target_state = _advance_state(target_model, target_state, commit_tokens)
        draft_state = _advance_state(draft_model, draft_state, commit_tokens)

        commit_list = commit_tokens[0].tolist()
        for token_id in commit_list:
            if len(generated_ids) >= max_new_tokens:
                break
            generated_ids.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break

        if eos_token_id is not None and generated_ids and generated_ids[-1] == eos_token_id:
            break

    _sync(target_state.device)
    wall_finished_at = time.perf_counter()

    generated_tensor = torch.tensor([generated_ids], dtype=torch.long)
    generated_text = tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
    full_ids = torch.cat([prompt_ids, generated_tensor], dim=-1)
    full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

    wall_time_ms = (wall_finished_at - wall_started_at) * 1000.0
    generated_token_count = int(generated_tensor.shape[1])
    ms_per_token = wall_time_ms / max(generated_token_count, 1)
    acceptance_rate = accepted_token_count / max(proposed_token_count, 1)
    mean_acceptance_length = sum(accepted_lengths) / max(len(accepted_lengths), 1)

    peak_runtime_vram_bytes = None
    if target_state.device.type == "cuda" and torch.cuda.is_available():
        peak_runtime_vram_bytes = int(torch.cuda.max_memory_allocated(target_state.device))

    return {
        "mode": "speculative",
        "prompt": prompt,
        "prompt_token_count": int(prompt_ids.shape[1]),
        "generated_token_count": generated_token_count,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "full_text": full_text,
        "speculative_k": speculative_k,
        "iterations": iterations,
        "proposed_token_count": proposed_token_count,
        "accepted_token_count": accepted_token_count,
        "accepted_lengths": accepted_lengths,
        "acceptance_rate": acceptance_rate,
        "mean_acceptance_length": mean_acceptance_length,
        "draft_time_ms": draft_time_ms,
        "target_time_ms": target_time_ms,
        "wall_time_ms": wall_time_ms,
        "ms_per_token": ms_per_token,
        "resident_vram_bytes": resident_vram_bytes,
        "peak_runtime_vram_bytes": peak_runtime_vram_bytes,
    }
