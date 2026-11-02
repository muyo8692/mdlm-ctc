"""Diffusion sampler with optional shift-intervention.

`generate()` runs the standard semi-autoregressive masked-diffusion sampler
(Gumbel-noise, low-confidence remasking, block-wise unmasking).

`generate_preliminary_exp()` is the same sampler with the controlled shift
intervention of the paper (Sec. 3.3): at each diffusion-time threshold in
`insert_at_mask_fractions`, it uniformly samples `insert_n` revealed tokens
that have a masked neighbour and swaps each with that neighbour, displacing
exactly `insert_n` tokens by one position while preserving their identity.
This is the probe used to measure how sensitive a model is to small
positional shifts under iterative parallel decoding.
"""

from .core import (
    generate,
    generate_preliminary_exp,
    swap_revealed_with_masked_neighbors,
)

__all__ = [
    "generate",
    "generate_preliminary_exp",
    "swap_revealed_with_masked_neighbors",
]
