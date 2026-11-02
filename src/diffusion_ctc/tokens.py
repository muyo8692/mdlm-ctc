"""Token-id constants for the LLaDA and LLaDA-MoE tokenizers.

These IDs are verified against the publicly-released tokenizers for
`GSAI-ML/LLaDA-8B-Instruct` and `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`.
Centralising them here keeps the trainers, collators, and generation code in
sync. Each `TokenSpec` records exactly which id plays each role for that
backbone family.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSpec:
    """Token-id roles needed by the trainers and the diffusion sampler."""

    mask: int
    blank: int           # CTC blank
    pad: int
    eos: int
    eot: int             # turn / role boundary
    # Token IDs whose logits/confidence should be suppressed when running with
    # --suppress-eos. Some backbones treat EOT and EOS as separate tokens.
    suppress_when_no_eos: tuple[int, ...]


# LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct)
LLADA = TokenSpec(
    mask=126336,
    blank=126084,         # <|reserved_token_0|>
    pad=126339,
    eos=126081,
    eot=126348,
    suppress_when_no_eos=(126081, 126348),
)


# LLaDA-MoE-7B-A1B-Instruct (inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
LLADA_MOE = TokenSpec(
    mask=156895,
    blank=156901,         # <|reserved_token_6|>
    pad=156892,           # EOS == PAD for the MoE tokenizer
    eos=156892,
    eot=156900,           # <|role_end|>
    suppress_when_no_eos=(156892,),
)


def spec_for(variant: str) -> TokenSpec:
    if variant == "llada":
        return LLADA
    if variant in ("moe", "llada_moe"):
        return LLADA_MOE
    raise ValueError(f"Unknown variant: {variant!r}")
