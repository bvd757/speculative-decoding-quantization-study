from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.speculative import baseline_generate, speculative_generate


class TinyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.vocab_size = 9
        self.chat_template = None

    def encode(self, text, add_special_tokens=False):
        words = text.strip().split()
        ids = [3 + (sum(ord(char) for char in word) % 6) for word in words]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def __call__(self, text, return_tensors="pt"):
        ids = self.encode(text, add_special_tokens=True)
        if not ids:
            ids = [self.bos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        special_ids = {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            pieces.append(f"tok{token_id}")
        return " ".join(pieces)


class ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int, mistake_period: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.mistake_period = mistake_period
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(use_cache=True)
        self.generation_config = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    def _true_next_token(self, prev_token: int) -> int:
        if 3 <= prev_token <= 8:
            return 3 + ((prev_token - 3 + 1) % 6)
        return 3

    def _wrong_token(self, true_token: int) -> int:
        return 3 + ((true_token - 3 + 2) % 6)

    def _draft_next_token(self, current_length: int, prev_token: int) -> int:
        true_token = self._true_next_token(prev_token)
        if self.mistake_period is not None and current_length % self.mistake_period == 0:
            return self._wrong_token(true_token)
        return true_token

    def forward(self, input_ids, use_cache=False):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -1e9, dtype=torch.float32, device=input_ids.device)
        for batch_index in range(batch_size):
            for position in range(seq_len):
                prev_token = int(input_ids[batch_index, position].item())
                next_token = self._true_next_token(prev_token)
                logits[batch_index, position, next_token] = 0.0
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids,
        max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        pad_token_id=None,
        eos_token_id=None,
    ):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            prev_token = int(generated[0, -1].item())
            current_length = int(generated.shape[1])
            next_token = self._draft_next_token(current_length=current_length, prev_token=prev_token)
            next_token_tensor = torch.tensor([[next_token]], dtype=generated.dtype, device=generated.device)
            generated = torch.cat([generated, next_token_tensor], dim=-1)
        return SimpleNamespace(sequences=generated)


def test_speculative_matches_baseline_with_perfect_draft():
    tokenizer = TinyTokenizer()
    target_model = ToyCausalLM(vocab_size=tokenizer.vocab_size)
    draft_model = ToyCausalLM(vocab_size=tokenizer.vocab_size)

    config = {
        "use_chat_template": False,
        "generation": {
            "max_new_tokens": 12,
            "speculative_k": 4,
        },
    }

    prompt = "alpha beta gamma"

    baseline = baseline_generate(
        model=target_model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )
    speculative = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )

    assert speculative["generated_token_ids"] == baseline["generated_token_ids"]
    assert speculative["generated_text"] == baseline["generated_text"]
    assert speculative["generated_token_count"] == baseline["generated_token_count"]
    assert speculative["acceptance_rate"] == 1.0
    assert all(value >= 0 for value in speculative["accepted_lengths"])


def test_speculative_matches_baseline_with_imperfect_draft():
    tokenizer = TinyTokenizer()
    target_model = ToyCausalLM(vocab_size=tokenizer.vocab_size)
    draft_model = ToyCausalLM(vocab_size=tokenizer.vocab_size, mistake_period=3)

    config = {
        "use_chat_template": False,
        "generation": {
            "max_new_tokens": 12,
            "speculative_k": 4,
        },
    }

    prompt = "delta epsilon zeta"

    baseline = baseline_generate(
        model=target_model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )
    speculative = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )

    assert speculative["generated_token_ids"] == baseline["generated_token_ids"]
    assert speculative["generated_text"] == baseline["generated_text"]
    assert speculative["generated_token_count"] == baseline["generated_token_count"]
    assert 0.0 <= speculative["acceptance_rate"] < 1.0
    assert any(value < config["generation"]["speculative_k"] for value in speculative["accepted_lengths"])
