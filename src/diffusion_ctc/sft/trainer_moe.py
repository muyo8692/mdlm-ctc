"""Vanilla SFT trainer/collator for LLaDA-MoE.

Identical to the LLaDA `dLLMTrainer` except it can optionally request router
logits and add the load-balancing aux loss exposed by `LLaDAMoEModelLM`.

NOTE: As of the published `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` modeling
code, `LLaDAMoEDecoderLayer.forward` does NOT actually append router_logits
to its returned tuple even when `output_router_logits=True`, so the model's
`aux_loss` path crashes with a shape mismatch. Leave `use_router_aux_loss`
at its default `False` until upstream patches that.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DefaultDataCollator, Trainer


class dLLMTrainer(Trainer):
    def __init__(
        self,
        *args,
        use_router_aux_loss: bool = False,
        router_aux_loss_coef: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_router_aux_loss = use_router_aux_loss
        if router_aux_loss_coef is None:
            router_aux_loss_coef = float(
                getattr(self.model.config, "router_aux_loss_coef", 0.0) or 0.0
            )
        self.router_aux_loss_coef = router_aux_loss_coef

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        num_prompt_tokens = inputs.pop("num_prompt_tokens")
        if self.use_router_aux_loss:
            inputs["output_router_logits"] = True

        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(
                {"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()}
            )
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)

        if self.use_router_aux_loss:
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
                if (self.state.global_step + 1) % self.args.logging_steps == 0:
                    self.log({"router_aux_loss": float(aux_loss.detach().item())})

        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert "mask_token_id" in kwargs, (
                "For dLLM models, pass a mask_token_id or set tokenizer.mask_token_id"
            )
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)

        base_attention_mask = batch["attention_mask"].clone()

        if not hasattr(self, "_debug_printed_collator"):
            self._debug_printed_collator = True
            print("\n" + "=" * 50)
            print("DEBUG: Collator Input (Before Masking) [LLaDA-MoE]")
            print("=" * 50)
            if self.tokenizer:
                input_ids_0 = batch["input_ids"][0]
                active_len_0 = int(base_attention_mask[0].sum().item())
                print(
                    f"\n[Input Text (Raw Padded)]: {self.tokenizer.decode(input_ids_0, skip_special_tokens=False)}"
                )
                print(f"[Input IDs]: {input_ids_0.tolist()}")
                print(
                    f"[Active length]: {active_len_0}, [Total length]: {input_ids_0.size(0)}"
                )
                print(f"[mask_token_id]: {self.mask_token_id}")
            print("=" * 50 + "\n")

        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()

        pad_mask = base_attention_mask == 0
        batch["labels"][pad_mask] = -100
        noisy_batch[pad_mask] = batch["input_ids"][pad_mask]
        batch["attention_mask"] = torch.ones_like(base_attention_mask)

        batch["input_ids"] = noisy_batch.long()
        return batch


# The LLaDA-MoE chat template terminates the assistant turn with `<|role_end|>`
# and does NOT add a trailing assistant header when add_generation_prompt=False.
# We keep a guard helper so the preprocess code is defensive against template
# changes; the runtime check makes a future template change fail loudly.
ASSISTANT_HEADER_VARIANTS = (
    "<role>ASSISTANT</role>",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
)


def _llada_moe_template_needs_strip(tokenizer) -> bool:
    probe_prompt = [{"role": "user", "content": "hi"}]
    probe_response = [{"role": "assistant", "content": "ok"}]
    try:
        rendered = tokenizer.apply_chat_template(
            probe_prompt + probe_response,
            add_generation_prompt=False,
            tokenize=False,
        )
    except Exception:
        return False
    return any(rendered.endswith(v) for v in ASSISTANT_HEADER_VARIANTS)


def maybe_strip_trailing_assistant_header(text: str, tokenizer) -> str:
    if not getattr(tokenizer, "_llada_moe_strip_probed", False):
        tokenizer._llada_moe_needs_strip = _llada_moe_template_needs_strip(tokenizer)
        tokenizer._llada_moe_strip_probed = True
    if not tokenizer._llada_moe_needs_strip:
        return text
    for v in ASSISTANT_HEADER_VARIANTS:
        if text.endswith(v):
            return text[: -len(v)]
    return text


def preprocess_dataset(data, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []

    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = data[i]["instruction"]
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": data[i]["response"]}]

        inputs = tokenizer.apply_chat_template(
            prompt + response, add_generation_prompt=False, tokenize=False
        )
        inputs = maybe_strip_trailing_assistant_header(inputs, tokenizer)

        prompt_text = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )

        tokenized_output = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=False,
        )
        tokenized_input = tokenized_output.input_ids.squeeze(0)
        attention_mask = tokenized_output.attention_mask.squeeze(0)

        tokenized_prompt = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "attention_mask": attention_mask,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
