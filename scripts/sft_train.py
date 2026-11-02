#!/usr/bin/env python3
"""CE-Only baseline SFT for LLaDA.

Standard absorbing-state cross-entropy on the pre-tokenized Magpie-Pro
subset (the same data the Jitter and CE+CTC-S variants consume); the only
difference is that the vanilla collator does not augment responses with
<slack> tokens at training time.

Defaults are taken from `config.yaml`; hyperparameters match the paper's
Implementation Details (Appendix B.2).
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from diffusion_ctc.config import load_config
from diffusion_ctc.sft.trainer import dLLMDataCollator, dLLMSFTDataset, dLLMTrainer

load_dotenv()


# Columns produced by `scripts/preprocess_magpie.py` that the CTC variants
# need but the vanilla CE-only baseline ignores. They are dropped before the
# DefaultDataCollator runs because some (e.g. clean_response_ids) are
# variable-length and would otherwise fail to stack.
_CTC_BOOKKEEPING_COLUMNS = [
    "clean_response_ids",
    "clean_response_length",
    "blanked_response_length",
    "response_start_idx",
    "original_length",
]


def init_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=cfg.models.resolve("llada"))
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=cfg.paths.output_dir)
    parser.add_argument("--job_name", type=str, default="llada-ce-only")
    parser.add_argument("--train_data", type=str, default=cfg.datasets.sft_baseline_llada,
                        help="Pre-tokenized Magpie-Pro subset, RAW variant (no slack-priming "
                             "system prompt). Produced by `preprocess_magpie.py --noslack-priming`.")
    parser.add_argument("--debugging", action="store_true",
                        help="Disable wandb logging.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "wte", "ff_out"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    return tokenizer, model


def load_data(args: argparse.Namespace, tokenizer):
    if not args.train_data:
        raise ValueError(
            "No baseline dataset configured. Set `datasets.sft_baseline_llada` in "
            "config.yaml (the raw, no-prompt Magpie subset produced by "
            "`scripts/preprocess_magpie.py --noslack-priming`) or pass "
            "--train_data explicitly."
        )
    print(f"Loading pre-tokenized data from: {args.train_data}")
    dataset = load_dataset(args.train_data)
    if "train" not in dataset:
        raise ValueError(f"Expected a 'train' split in {args.train_data!r}.")
    train = dataset["train"]

    drop_cols = [c for c in _CTC_BOOKKEEPING_COLUMNS if c in train.column_names]
    if drop_cols:
        train = train.remove_columns(drop_cols)

    if len(train) > 0:
        example_len = len(train[0]["input_ids"])
        if example_len != args.max_length:
            raise ValueError(
                f"Processed data max_length ({example_len}) mismatches training "
                f"max_length ({args.max_length}). Re-run preprocess_magpie.py "
                "with the right length or pass --max_length to match."
            )

    split = train.train_test_split(test_size=0.01)
    train_split, eval_split = split["train"], split["test"]
    print("Train data length:", len(train_split))
    print("Eval data length:", len(eval_split))
    train_dataset = dLLMSFTDataset(train_split, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_split, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset


def train_model(args: argparse.Namespace, tokenizer, model) -> None:
    train_dataset, eval_dataset = load_data(args, tokenizer)

    n_devices = max(1, torch.cuda.device_count())
    num_train_steps = int(
        len(train_dataset) * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * n_devices)
    )
    eval_save_every = max(1, num_train_steps // 10)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=eval_save_every,
        logging_steps=1,
        save_steps=eval_save_every,
        save_total_limit=10,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )
    print(f"Total training steps: {num_train_steps}")

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    args = parse_args()
    init_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)
