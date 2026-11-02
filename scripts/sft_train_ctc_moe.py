#!/usr/bin/env python3
"""CTC-augmented SFT for LLaDA-MoE. Consumes a pre-processed dataset built by
`scripts/preprocess_magpie_moe.py`."""

from __future__ import annotations

import os
import random
from pathlib import Path

import fire
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Subset
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from diffusion_ctc.config import load_config
from diffusion_ctc.sft.trainer_ctc_moe import (
    dLLMDataCollator,
    dLLMSFTDataset,
    dLLMTrainer,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()


def init_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )
    # AutoModel resolves to LLaDAMoEModelLM via the auto_map.
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    # See trainer_moe.py for why router/expert modules are intentionally excluded.
    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "embed_tokens", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    return tokenizer, model


def load_data(train_data_path: str, tokenizer, max_length: int):
    print(f"Loading pre-processed CTC data from: {train_data_path}")
    dataset = load_dataset(train_data_path)

    if len(dataset["train"]) > 0:
        example_len = len(dataset["train"][0]["input_ids"])
        if example_len != max_length:
            raise ValueError(
                f"Processed data max_length ({example_len}) mismatches training "
                f"max_length ({max_length}). Re-run preprocess with the right length."
            )

    train_data = dataset["train"]
    split = train_data.train_test_split(test_size=0.01)
    train_data = split["train"]
    eval_data = split["test"]

    print("Train data length:", len(train_data))
    print("Eval data length:", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, max_length, eval=True)
    return train_dataset, eval_dataset


def main(
    model_name: str | None = None,
    batch_size: int = 3,
    max_length: int = 2048,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    grad_accum_steps: int = 1,
    output_dir: str | None = None,
    job_name: str = "llada-moe-ce-ctcs",
    train_data: str | None = None,
    debugging: bool = False,
    ctc_weight: float = 0.1,
    ce_blank_weight: float = 1.0,
    slack_max_ratio: float = 0.5,
    seed: int = 42,
    quick_test: bool = False,
    use_router_aux_loss: bool = False,
    router_aux_loss_coef: float | None = None,
):
    cfg = load_config()
    model_name = model_name or cfg.models.resolve("moe")
    output_dir = output_dir or cfg.paths.output_dir
    train_data = train_data or cfg.datasets.sft_slack_moe
    if not train_data:
        raise ValueError(
            "No slack-primed dataset configured. Set `datasets.sft_slack_moe` "
            "in config.yaml (the slack-primed Magpie subset produced by "
            "`scripts/preprocess_magpie_moe.py`) or pass --train_data explicitly."
        )

    init_seed(seed)

    tokenizer, model = load_model_and_tokenizer(model_name)
    train_dataset, eval_dataset = load_data(train_data, tokenizer, max_length)

    if quick_test:
        train_dataset = Subset(train_dataset, [0])
        eval_dataset = None

    n_devices = max(1, torch.cuda.device_count())
    num_train_steps = int(
        len(train_dataset) * num_epochs
        / (batch_size * grad_accum_steps * n_devices)
    )
    eval_save_every = max(1, num_train_steps // 10)

    output_path = Path(output_dir) / job_name
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_save_every,
        logging_steps=1,
        save_steps=eval_save_every,
        save_total_limit=10,
        learning_rate=learning_rate,
        load_best_model_at_end=True if eval_dataset is not None else False,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not debugging else "none",
        remove_unused_columns=False,
    )

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=dLLMDataCollator(
            processing_class=tokenizer,
            max_length=max_length,
            slack_max_ratio=slack_max_ratio,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ctc_weight=ctc_weight,
        ce_blank_weight=ce_blank_weight,
        use_router_aux_loss=use_router_aux_loss,
        router_aux_loss_coef=router_aux_loss_coef,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
