#!/usr/bin/env python3
"""Self-contained sample of the diffusion sampler.

Loads a LLaDA / LLaDA-MoE checkpoint with `transformers.AutoModel`, builds a
chat-template prompt, and runs either the standard diffusion sampler or the
shift-intervention variant used in the paper's Sec. 3.3 / Sec. 6.1 experiments.

Examples:
    # standard sampler, LLaDA-8B-Instruct
    uv run python scripts/run_generate.py \\
        --variant llada --prompt "Explain inertia like I'm five." \\
        --steps 256 --gen-length 512 --block-length 512

    # shift intervention with thresholds matching the paper (t-values, Sec. 3.3)
    uv run python scripts/run_generate.py \\
        --variant llada --prompt "..." --preliminary-exp \\
        --insert-n 2 --insert-fractions "[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]" \\
        --steps 512 --gen-length 1024 --block-length 1024
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from diffusion_ctc.config import load_config
from diffusion_ctc.generate import generate, generate_preliminary_exp
from diffusion_ctc.tokens import spec_for

load_dotenv()


def _parse_insert_fractions(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()
    if not s:
        return None
    parts = [p for p in s.replace(" ", ",").split(",") if p]
    out = [float(p) for p in parts]
    return out or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("llada", "moe"), default="llada",
                        help="Which backbone family to load.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override the HF repo / local path (otherwise from config.yaml).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The user prompt to feed through the chat template.")

    # Diffusion sampler knobs.
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence",
                        choices=("low_confidence", "random"))

    # EOS handling.
    parser.add_argument("--suppress-eos", action=argparse.BooleanOptionalAction, default=True,
                        help="Block EOS-equivalent tokens from being unmasked.")
    parser.add_argument("--logits-eos-inf", action="store_true",
                        help="Force EOS logits to -inf before Gumbel sampling (ablation).")

    # Shift intervention.
    parser.add_argument("--preliminary-exp", action="store_true",
                        help="Use the shift-intervention sampler instead of the standard one.")
    parser.add_argument("--insert-n", type=int, default=0,
                        help="Number of revealed-token / adjacent-mask pairs to swap at each threshold.")
    parser.add_argument("--insert-fractions", type=str, default="",
                        help="Diffusion-time (t) thresholds at which to apply the shift, e.g. '[0.95,0.5]'. "
                             "Events fire as t descends past each value.")

    parser.add_argument("--device", default=None,
                        help="Force device. Default: cuda if available, else cpu.")
    parser.add_argument("--dtype", default="auto",
                        choices=("auto", "bf16", "fp16", "fp32"),
                        help="Model dtype. 'auto' picks bf16 on GPU, fp32 on CPU.")
    return parser.parse_args()


def _resolve_dtype(arg: str, device: torch.device) -> torch.dtype:
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    if arg == "fp32":
        return torch.float32
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def main() -> None:
    args = parse_args()
    cfg = load_config()
    spec = spec_for(args.variant)

    model_path = args.model_path or cfg.models.resolve(args.variant)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = _resolve_dtype(args.dtype, device)
    print(f"Loading model {model_path!r} -> device={device}, dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    chat = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    if args.preliminary_exp:
        fractions = _parse_insert_fractions(args.insert_fractions)
        out = generate_preliminary_exp(
            model,
            input_ids,
            mask_id=spec.mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            insert_n=args.insert_n,
            insert_at_mask_fractions=fractions,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
        )
    else:
        out = generate(
            model,
            input_ids,
            mask_id=spec.mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            suppress_eos=args.suppress_eos,
            suppress_token_ids=spec.suppress_when_no_eos,
            logits_eos_inf=args.logits_eos_inf,
            logits_eos_token_id=spec.eos,
        )

    prompt_len = input_ids.shape[1]
    gen_ids = out[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("\n=== Generation ===\n" + text)


if __name__ == "__main__":
    main()
