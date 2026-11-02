"""Blank-only CTC loss for masked-diffusion SFT.

Tries the on-demand CUDA build first; falls back to a pure-torch JIT
implementation that runs on CPU so the recurrence is inspectable without a
GPU.
"""

from __future__ import annotations

import torch

try:
    from .ctc_blank_only_cuda import ctc_loss_blank_only as _ctc_loss_cuda

    _HAVE_CUDA_KERNEL = True
except Exception:  # pragma: no cover - depends on host toolchain
    _HAVE_CUDA_KERNEL = False


@torch.jit.script
def _ctc_loss_blank_only_jit(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = True,
) -> torch.Tensor:
    device = log_probs.device
    dtype = log_probs.dtype
    T, B, V = log_probs.shape
    NEGINF = torch.tensor(-1e9, device=device, dtype=dtype)

    if targets.dim() == 1:
        max_s = int(target_lengths.max().item())
        tgts = targets.new_zeros(B, max_s)
        offset = 0
        for b in range(B):
            tl = int(target_lengths[b].item())
            tgts[b, :tl] = targets[offset : offset + tl]
            offset += tl
    else:
        tgts = targets
        max_s = tgts.size(1)

    max_ext = 2 * max_s + 1

    ext_labels = tgts.new_full((B, max_ext), blank)
    for s in range(max_s):
        ext_labels[:, 2 * s + 1] = tgts[:, s]

    ext_lengths = 2 * target_lengths + 1

    ext_exp = ext_labels.unsqueeze(0).expand(T, B, max_ext)
    emit = log_probs.gather(2, ext_exp)

    state_valid = (
        torch.arange(max_ext, device=device).unsqueeze(0) < ext_lengths.unsqueeze(1)
    )

    is_blank_state = torch.arange(max_ext, device=device) % 2 == 0
    is_label_state = ~is_blank_state

    alpha = torch.full((B, max_ext), -1e9, device=device, dtype=dtype)
    alpha[:, 0] = emit[0, :, 0]
    if max_ext > 1:
        alpha[:, 1] = torch.where(target_lengths > 0, emit[0, :, 1], NEGINF)

    all_alpha = torch.full((T, B, max_ext), -1e9, device=device, dtype=dtype)
    all_alpha[0] = alpha

    for t in range(1, T):
        prev = torch.full_like(alpha, -1e9)
        prev[:, 1:] = alpha[:, :-1]

        skip = torch.full_like(alpha, -1e9)
        skip[:, 2:] = alpha[:, :-2]
        skip = torch.where(is_label_state, skip, NEGINF)

        self_loop = torch.where(is_blank_state, alpha, NEGINF)
        alpha = torch.logaddexp(torch.logaddexp(self_loop, prev), skip) + emit[t]
        alpha = torch.where(state_valid, alpha, NEGINF)
        all_alpha[t] = alpha

    t_idx = (input_lengths - 1).clamp(min=0)
    b_idx = torch.arange(B, device=device)
    final_alpha = all_alpha[t_idx, b_idx]

    last_blank = final_alpha[b_idx, (ext_lengths - 1).clamp(min=0)]
    last_label = torch.where(
        ext_lengths >= 2,
        final_alpha[b_idx, (ext_lengths - 2).clamp(min=0)],
        NEGINF,
    )
    result_log_prob = torch.logaddexp(last_blank, last_label)

    loss = -result_log_prob

    if zero_infinity:
        loss = torch.where(loss > 1e8, torch.zeros_like(loss), loss)

    return (loss / target_lengths.to(dtype=loss.dtype).clamp(min=1)).mean()


def ctc_loss_blank_only(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = True,
) -> torch.Tensor:
    """Blank-only CTC loss. Uses CUDA kernel if available, JIT fallback otherwise."""
    if _HAVE_CUDA_KERNEL and log_probs.is_cuda:
        return _ctc_loss_cuda(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            zero_infinity=zero_infinity,
        )
    return _ctc_loss_blank_only_jit(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        zero_infinity=zero_infinity,
    )
