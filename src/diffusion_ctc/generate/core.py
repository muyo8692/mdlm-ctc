"""Core diffusion sampler shared between LLaDA and LLaDA-MoE.

The two backbones differ only in `mask_id` and which tokens to suppress when
running with `--no-suppress-eos`; both are passed in by the caller (see
`scripts/run_generate.py` or the `tokens` module for canonical values).
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

import torch


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling perturbation. We use float64 because low-precision
    Gumbel hurts generation quality for masked diffusion (arXiv:2409.02908)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    log_gumbel_noise = temperature * torch.log(-torch.log(noise))
    return logits - log_gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Even split of mask_count across `steps`, distributing remainder to early steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    *,
    mask_id: int,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    suppress_eos: bool = False,
    suppress_token_ids: Sequence[int] = (),
    logits_eos_inf: bool = False,
    logits_eos_token_id: Optional[int] = None,
    return_history: bool = False,
    return_trace: bool = False,
    return_timing: bool = False,
):
    """Standard masked-diffusion sampler.

    Args:
        model: Mask predictor; called as `model(x).logits`.
        prompt: (1, L) prompt token ids.
        mask_id: Token id used to denote masked positions.
        suppress_eos: If True, set the confidence of every id in
            `suppress_token_ids` to -inf so it is never unmasked. (Pass the
            full list of EOS / EOT-equivalent ids for the active backbone.)
        logits_eos_inf: If True, force a single id's logits to -inf BEFORE
            Gumbel sampling, killing it as a candidate even with high noise.
            Used for `--logits-eos-inf` ablations.
    """
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    history: Optional[list] = [] if return_history else None
    trace: Optional[list] = [] if return_trace else None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if logits_eos_inf and logits_eos_token_id is not None:
                logits[:, :, logits_eos_token_id] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                x0_logits = torch.squeeze(
                    torch.gather(logits, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                x0_p = (x0_logits - logits.logsumexp(dim=-1)).exp()
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            if suppress_eos and suppress_token_ids:
                eos_mask = torch.zeros_like(x0, dtype=torch.bool)
                for tok_id in suppress_token_ids:
                    eos_mask |= x0 == tok_id
                x0_p[eos_mask] = -torch.inf

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = torch.finfo(
                x0_p.dtype
            ).min

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.finfo(x0_p.dtype).min)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if return_trace:
                revealed_abs = torch.where(transfer_index[0])[0]
                revealed_ids = x[0, revealed_abs]
                prompt_len = int(prompt.shape[1])
                revealed_gen = revealed_abs - prompt_len
                keep = revealed_gen >= 0
                revealed_abs = revealed_abs[keep]
                revealed_gen = revealed_gen[keep]
                revealed_ids = revealed_ids[keep]

                trace.append(
                    {
                        "block_idx": int(num_block),
                        "step_in_block": int(i),
                        "global_step": int(num_block * steps + i),
                        "positions_abs": revealed_abs.detach().cpu().tolist(),
                        "positions_gen": revealed_gen.detach().cpu().tolist(),
                        "token_ids": revealed_ids.detach().cpu().tolist(),
                    }
                )

            if return_history:
                history.append(x.detach().cpu().clone())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    if return_timing:
        if return_history and return_trace:
            return x, history, trace, elapsed
        if return_history:
            return x, history, elapsed
        if return_trace:
            return x, trace, elapsed
        return x, elapsed

    if return_history and return_trace:
        return x, history, trace
    if return_history:
        return x, history
    if return_trace:
        return x, trace
    return x


def swap_revealed_with_masked_neighbors(
    x: torch.Tensor, *, prompt_len: int, mask_id: int, k: int
) -> int:
    """One shift-intervention event (paper Sec. 3.3), applied in place.

    Among response positions (index >= `prompt_len`) that are already
    revealed and have at least one masked neighbour, build disjoint
    (revealed, masked-neighbour) pairs, uniformly sample up to `k` of them,
    and swap each pair: the revealed token moves one position into the
    masked slot and the slot it left becomes masked again. Token identity
    and the number of revealed tokens are preserved; exactly `min(k, #pairs)`
    tokens are displaced by one position.

    Returns the number of pairs swapped (0 when no candidate exists, in which
    case `x` is left untouched).
    """
    ok_tok = x[0] != mask_id
    mask_tok = x[0] == mask_id
    gen_ok = ok_tok[prompt_len:]
    right_is_mask = torch.zeros_like(gen_ok)
    right_is_mask[:-1] = gen_ok[:-1] & mask_tok[prompt_len + 1 :]
    left_is_mask = torch.zeros_like(gen_ok)
    left_is_mask[1:] = gen_ok[1:] & mask_tok[prompt_len : x.shape[1] - 1]
    candidate_mask = right_is_mask | left_is_mask
    candidate_idx = torch.nonzero(candidate_mask).flatten()
    if candidate_idx.numel() == 0:
        return 0

    build_order = torch.randperm(candidate_idx.numel(), device=candidate_idx.device)
    pairs = []
    reserved = set()
    for c in candidate_idx[build_order].tolist():
        pos = c + prompt_len
        options = []
        if pos + 1 < x.shape[1] and bool(mask_tok[pos + 1].item()):
            options.append(pos + 1)
        if pos - 1 >= prompt_len and bool(mask_tok[pos - 1].item()):
            options.append(pos - 1)
        if len(options) == 2 and torch.rand(1).item() < 0.5:
            options.reverse()
        chosen = next((n for n in options if n not in reserved), None)
        if chosen is None:
            continue
        reserved.add(chosen)
        pairs.append((pos, chosen))

    if not pairs:
        return 0

    k_swap = min(int(k), len(pairs))
    sample_idx = torch.randperm(len(pairs))[:k_swap].tolist()
    swap_pos = [pairs[j][0] for j in sample_idx]
    swap_neigh = [pairs[j][1] for j in sample_idx]
    pos_t = torch.tensor(swap_pos, device=x.device, dtype=torch.long)
    neigh_t = torch.tensor(swap_neigh, device=x.device, dtype=torch.long)
    vals = x[0, pos_t].clone()
    x[0, pos_t] = mask_id
    x[0, neigh_t] = vals
    return k_swap


@torch.no_grad()
def generate_preliminary_exp(
    model,
    prompt: torch.Tensor,
    *,
    mask_id: int,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    insert_n: int = 0,
    insert_at_mask_fractions: Optional[Iterable[float]] = None,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
):
    """Sampler with a shift intervention.

    At each diffusion-time threshold in `insert_at_mask_fractions`, uniformly
    sample `insert_n` revealed tokens that have at least one masked neighbor
    and swap each with one such neighbor. Each event displaces exactly K
    tokens by one position, preserving token identity and immediate lexical
    context. Thresholds are interpreted in t-units (mask fraction under
    LLaDA's linear schedule); under that schedule t ≈ remaining_masks /
    initial_masks. Single-block sampler (no semi-AR).
    """
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_len = prompt.shape[1]

    shift_n = int(insert_n) if insert_n is not None else 0
    if shift_n < 0:
        raise ValueError("insert_n must be >= 0.")

    shift_thresholds = None
    shift_ptr = 0
    if insert_at_mask_fractions:
        # Thresholds are t-values; events fire as t descends past each one.
        shift_thresholds = sorted(
            {float(v) for v in insert_at_mask_fractions if v is not None},
            reverse=True,
        )
        for f in shift_thresholds:
            if f < 0.0 or f > 1.0:
                raise ValueError("insert_at_mask_fractions must be within [0, 1].")

    steps = int(steps)
    initial_mask_index = x == mask_id
    initial_masks = int(initial_mask_index.sum().item())
    if initial_masks == 0:
        return x
    num_transfer_tokens = get_num_transfer_tokens(initial_mask_index, steps)

    for i in range(steps):
        if shift_thresholds is not None and shift_n > 0:
            while shift_ptr < len(shift_thresholds):
                remaining_masks = int((x == mask_id).sum().item())
                current_t = remaining_masks / float(initial_masks)
                if current_t > shift_thresholds[shift_ptr]:
                    break

                swapped = swap_revealed_with_masked_neighbors(
                    x, prompt_len=prompt_len, mask_id=mask_id, k=shift_n
                )
                if swapped == 0:
                    # No revealed token has a masked neighbour yet; retry the
                    # same threshold at the next step.
                    break

                shift_ptr += 1

        mask_index = x == mask_id
        num_masks = int(mask_index.sum().item())
        if num_masks == 0:
            break

        k = int(num_transfer_tokens[0, i].item())
        if k <= 0:
            continue
        k = min(k, num_masks)

        prompt_index = x != mask_id

        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == "low_confidence":
            x0_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )
            x0_p = (x0_logits - logits.logsumexp(dim=-1)).exp()
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.finfo(x0_p.dtype).min)

        _, select_index = torch.topk(confidence[0], k=k)
        x[0, select_index] = x0[0, select_index]

    return x
