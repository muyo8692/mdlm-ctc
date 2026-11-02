#!/usr/bin/env python3
"""Build the LLaDA-MoE CTC training dataset from Magpie-Pro-300K-Filtered.

Same shape as `preprocess_magpie.py` but uses the LLaDA-MoE tokenizer and
constants. The chat template differs: LLaDA-MoE uses `<|role_end|>` as the
turn boundary and does not require post-hoc header stripping when called with
`add_generation_prompt=False`.
"""

from __future__ import annotations

import fire
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from diffusion_ctc.config import load_config
from diffusion_ctc.tokens import LLADA_MOE

load_dotenv()

PAD_TOKEN_ID = LLADA_MOE.pad
BLANK_TOKEN_ID = LLADA_MOE.blank
ROLE_END_TOKEN_ID = LLADA_MOE.eot
EOS_TOKEN_ID = LLADA_MOE.eos

SYSTEM_PROMPT = """
Use <|reserved_token_6|> to adjust spacing during generation while response to the following question.
"""


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
    """Build the LLaDA-MoE preprocessed Magpie subset.

    `slack_priming=True` (default) prepends the slack-priming system prompt
    that teaches the Jitter / CE+CTC-S MoE models to emit <slack> at
    generation time. Pass `--noslack-priming` to produce the raw variant
    used by the CE-Only MoE baseline (no system prompt prepended). The two
    variants are uploaded under distinct names so both can coexist on the
    Hub.
    """
    cfg = load_config()
    dataset_name = dataset_name or cfg.datasets.source_magpie
    tokenizer_name = tokenizer_name or cfg.models.resolve("moe")
    if slack_priming:
        upload_dataset_name = (
            f"Magpie-Pro-300K-Filtered-CTC-{min_tokens}-{max_tokens}-MoE"
        )
    else:
        upload_dataset_name = (
            f"Magpie-Pro-300K-Filtered-{min_tokens}-{max_tokens}-MoE"
        )

    hf_token = cfg.api_keys.hf_up_token
    if push_to_hub and not hf_token:
        raise SystemExit(
            "push_to_hub=True requires HF_UP_TOKEN (or `api_keys.hf_up_token` in config.yaml)."
        )

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Sanity-check the constants against the actual tokenizer.
    assert tokenizer.mask_token_id == LLADA_MOE.mask, (
        f"Unexpected mask_token_id: {tokenizer.mask_token_id}"
    )
    assert tokenizer.eos_token_id == EOS_TOKEN_ID, (
        f"Unexpected eos_token_id: {tokenizer.eos_token_id}"
    )
    assert tokenizer.convert_tokens_to_ids("<|role_end|>") == ROLE_END_TOKEN_ID
    assert tokenizer.convert_tokens_to_ids("<|reserved_token_6|>") == BLANK_TOKEN_ID

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

        # See preprocess_magpie.py for the rationale behind the two variants.
        question = (SYSTEM_PROMPT + "\n\n" + instruction) if slack_priming else instruction
        prompt = [{"role": "user", "content": question}]
        response_msg = [{"role": "assistant", "content": response}]

        try:
            # No assistant-header strip: with add_generation_prompt=False the
            # MoE chat template terminates the assistant turn with <|role_end|>
            # and adds no trailing generation marker.
            inputs = tokenizer.apply_chat_template(
                prompt + response_msg,
                add_generation_prompt=False,
                tokenize=False,
            )

            prompt_text = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )

            full_ids = tokenizer(
                inputs, return_tensors="pt", truncation=False, add_special_tokens=False
            ).input_ids
            original_len = full_ids.size(1)

            tokenized_inputs = tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                add_special_tokens=False,
            )
            tokenized_input = tokenized_inputs.input_ids.squeeze(0)
            input_mask = tokenized_inputs.attention_mask.squeeze(0)

            tokenized_prompt = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                add_special_tokens=False,
            )

            prompt_len = int(tokenized_prompt.attention_mask.sum(-1).item())
            response_start = prompt_len

            seq_len = tokenized_input.size(0)
            if response_start >= seq_len:
                return None

            response_ids = tokenized_input[response_start:]

            # Convert padding region to PAD; force-place EOT then EOS at the
            # boundary so the response always ends with [..., EOT, EOS, PAD*].
            import torch

            pad_mask = input_mask == 0
            if pad_mask.any():
                tokenized_input[pad_mask] = PAD_TOKEN_ID
                first_pad_idx = torch.where(pad_mask)[0][0]
                if first_pad_idx < tokenized_input.size(0):
                    tokenized_input[first_pad_idx] = ROLE_END_TOKEN_ID
                    if first_pad_idx + 1 < tokenized_input.size(0):
                        tokenized_input[first_pad_idx + 1] = EOS_TOKEN_ID
                    else:
                        tokenized_input[first_pad_idx - 1] = EOS_TOKEN_ID
                elif tokenized_input.size(0) >= 2:
                    tokenized_input[-2] = ROLE_END_TOKEN_ID
                    tokenized_input[-1] = EOS_TOKEN_ID
            else:
                if tokenized_input.size(0) >= 2:
                    tokenized_input[-2] = ROLE_END_TOKEN_ID
                    tokenized_input[-1] = EOS_TOKEN_ID

            response_mask = input_mask[response_start:]
            is_pad = response_mask == 0
            is_special = (response_ids == EOS_TOKEN_ID) | (
                response_ids == ROLE_END_TOKEN_ID
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
                "prompt_lengths": [prompt_len],
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
