#!/usr/bin/env python3
"""Build the LLaDA CTC training dataset from Magpie-Pro-300K-Filtered.

Inserts a system prompt that primes the model to use the CTC blank token
during generation, tokenises with the LLaDA-Instruct tokenizer, computes the
response start index and the clean (un-blanked) target ids needed by the CTC
loss, and (optionally) pushes the result to the Hugging Face Hub.

Source repo IDs and the destination namespace come from `config.yaml`."""

from __future__ import annotations

import os

import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from diffusion_ctc.config import load_config
from diffusion_ctc.tokens import LLADA

load_dotenv()

PAD_TOKEN_ID = LLADA.pad
BLANK_TOKEN_ID = LLADA.blank
EOT_TOKEN_ID = LLADA.eot

SYSTEM_PROMPT = """
Use <|reserved_token_0|> to adjust spacing during generation while response to the following question.
"""

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
ASSISTANT_HEADER_SUFFIXES = f"{ASSISTANT_HEADER}\n\n"


def strip_trailing_assistant_header(text: str) -> str:
    if text.endswith(ASSISTANT_HEADER_SUFFIXES):
        return text[: -len(ASSISTANT_HEADER_SUFFIXES)]
    return text


def main(
    min_tokens: int = 512,
    max_tokens: int = 1024,
    max_seq_length: int = 2048,
    num_proc: int = 32,
    push_to_hub: bool = False,
    repo_namespace: str | None = None,
    dataset_name: str | None = None,
    tokenizer_name: str | None = None,
    set_gated: bool = False,
    slack_priming: bool = True,
):
    """Args:
        repo_namespace: When pushing, override the destination HF namespace.
            Otherwise the script uses the namespace it can authenticate with.
        dataset_name: Override the upstream dataset (defaults to
            `cfg.datasets.source_magpie`).
        tokenizer_name: Override the LLaDA tokenizer (defaults to
            `cfg.models.llada`).
        slack_priming: If True (default), prepend a short system prompt
            instructing the model to use the <slack> token to adjust spacing
            during generation. This is the dataset variant consumed by the
            Jitter and CE+CTC-S training runs. Pass `--noslack-priming` to
            produce the raw variant used by the CE-Only baseline (no
            system prompt prepended). The two variants are uploaded under
            distinct names so both can coexist on the Hub.
    """
    cfg = load_config()
    dataset_name = dataset_name or cfg.datasets.source_magpie
    tokenizer_name = tokenizer_name or cfg.models.resolve("llada")
    if slack_priming:
        upload_dataset_name = f"Magpie-Pro-300K-Filtered-CTC-{min_tokens}-{max_tokens}"
    else:
        upload_dataset_name = f"Magpie-Pro-300K-Filtered-{min_tokens}-{max_tokens}"

    hf_token = cfg.api_keys.hf_up_token
    if push_to_hub and not hf_token:
        raise SystemExit(
            "push_to_hub=True requires HF_UP_TOKEN (or `api_keys.hf_up_token` in config.yaml)."
        )

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Initial dataset size: {len(ds)}")
    print(f"Slack priming: {slack_priming} -> upload name: {upload_dataset_name}")

    def _prepare_single_example(example):
        instruction = None
        response = None

        if "conversations" in example:
            for turn in example["conversations"]:
                if turn["from"] == "human":
                    instruction = turn["value"]
                elif turn["from"] == "gpt":
                    response = turn["value"]
        else:
            instruction = example.get("instruction")
            response = example.get("response")

        if not instruction or not response:
            return None

        # The slack-priming system prompt is what teaches the (Jitter / CE+CTC-S)
        # model that it can emit <slack> at generation time. The CE-Only
        # baseline is trained on the raw instruction with no priming.
        question = (SYSTEM_PROMPT + "\n\n" + instruction) if slack_priming else instruction
        prompt = [{"role": "user", "content": question}]
        response_msg = [{"role": "assistant", "content": response}]

        try:
            inputs = tokenizer.apply_chat_template(prompt + response_msg, tokenize=False)
            inputs = strip_trailing_assistant_header(inputs)
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"

            full_ids = tokenizer(inputs, return_tensors="pt", truncation=False).input_ids
            original_len = full_ids.size(1)

            tokenized_inputs = tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
            tokenized_input = tokenized_inputs.input_ids.squeeze(0)
            input_mask = tokenized_inputs.attention_mask.squeeze(0)

            tokenized_prompt = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )

            response_boundary = tokenizer(
                ASSISTANT_HEADER, return_tensors="pt"
            ).input_ids.squeeze(0)

            header_idx = -1
            pat_len = response_boundary.size(0)
            seq_len = tokenized_input.size(0)

            start_search = max(0, tokenized_prompt.input_ids.size(1) - 20)

            for start in range(start_search, seq_len - pat_len + 1):
                if torch.equal(
                    tokenized_input[start : start + pat_len], response_boundary
                ):
                    header_idx = start
                    break

            if header_idx == -1:
                return None

            response_start = header_idx + pat_len
            while response_start < seq_len and tokenized_input[response_start] == 198:
                response_start += 1

            response_ids = tokenized_input[response_start:]

            pad_mask = input_mask == 0
            if pad_mask.any():
                tokenized_input[pad_mask] = PAD_TOKEN_ID
                first_pad_idx = torch.where(pad_mask)[0][0]
                if first_pad_idx < tokenized_input.size(0):
                    tokenized_input[first_pad_idx] = EOT_TOKEN_ID
                    if first_pad_idx + 1 < tokenized_input.size(0):
                        tokenized_input[first_pad_idx + 1] = tokenizer.eos_token_id
                    else:
                        tokenized_input[first_pad_idx - 1] = tokenizer.eos_token_id
                elif tokenized_input.size(0) >= 2:
                    tokenized_input[-2] = EOT_TOKEN_ID
                    tokenized_input[-1] = tokenizer.eos_token_id
            else:
                if tokenized_input.size(0) >= 2:
                    tokenized_input[-2] = EOT_TOKEN_ID
                    tokenized_input[-1] = tokenizer.eos_token_id

            response_mask = input_mask[response_start:]
            is_pad = response_mask == 0
            is_special = (response_ids == tokenizer.eos_token_id) | (
                response_ids == EOT_TOKEN_ID
            )
            keep_mask = ~(is_pad | is_special)
            clean_response_ids = response_ids[keep_mask]

            real_response_len = input_mask[response_start:].sum().item()
            clean_len = len(clean_response_ids)

            if clean_len < min_tokens or clean_len > max_tokens:
                return None

            return {
                "input_ids": tokenized_input.tolist(),
                "attention_mask": input_mask.tolist(),
                "clean_response_ids": clean_response_ids.tolist(),
                "clean_response_length": clean_len,
                "blanked_response_length": real_response_len,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1).tolist(),
                "response_start_idx": int(response_start),
                "original_length": original_len,
            }
        except Exception:
            return None

    def _prepare_example(batch):
        variants = []
        keys = list(batch.keys())
        num_examples = len(batch[keys[0]])
        for i in range(num_examples):
            example = {k: batch[k][i] for k in keys}
            res = _prepare_single_example(example)
            if res:
                variants.append(res)

        if not variants:
            return {k: [] for k in ["input_ids"]}

        collated = {key: [] for key in variants[0]}
        for variant in variants:
            for key, value in variant.items():
                collated[key].append(value)
        return collated

    print("Applying CTC Preprocessing & Filtering...")
    ds_processed = ds.map(
        _prepare_example,
        batched=True,
        batch_size=5000,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="CTC Preprocessing",
    )

    print("Filtering invalid CTC examples...")

    def is_valid_ctc(example):
        return example["blanked_response_length"] >= example["clean_response_length"]

    ds_final = ds_processed.filter(is_valid_ctc, num_proc=num_proc)
    print(f"Final Dataset Size: {len(ds_final)}")

    lengths = ds_final["clean_response_length"]
    if lengths:
        print(f"Avg Response Length: {sum(lengths) / len(lengths):.2f}")
        print(f"Max Response Length: {max(lengths)}")

    if push_to_hub:
        api = HfApi(token=hf_token)
        username = repo_namespace or api.whoami()["name"]
        repo_id = f"{username}/{upload_dataset_name}"
        print(f"Uploading to Hugging Face Hub as {repo_id}...")
        ds_final.push_to_hub(repo_id, token=hf_token, private=False)

        if set_gated:
            try:
                api.update_repo_settings(
                    repo_id=repo_id,
                    repo_type="dataset",
                    gated="manual",
                    private=False,
                )
                print("Repository is now gated.")
            except Exception as gate_err:
                print(f"Warning: Could not set gated status: {gate_err}")

        print("Upload complete.")
    else:
        save_path = f"./{upload_dataset_name}"
        print(f"Saving to local disk: {save_path}")
        ds_final.save_to_disk(save_path)


if __name__ == "__main__":
    fire.Fire(main)
