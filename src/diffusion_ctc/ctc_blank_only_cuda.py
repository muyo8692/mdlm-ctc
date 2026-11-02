"""On-demand CUDA build of the blank-only CTC kernel.

Imported lazily by `ctc_loss.py`; if CUDA / nvcc / a working host compiler are
unavailable, the import fails cleanly and callers fall back to the pure-torch
JIT implementation in `ctc_loss_jit.py`.
"""

import glob
import os

import torch
import torch.nn.functional as F  # noqa: F401  (kept for API parity with upstream)


_ext = None


def _setup_build_env():
    if not os.environ.get("CUDA_HOME"):
        import shutil

        candidates = ["/usr/local/cuda"] + sorted(
            glob.glob("/usr/local/cuda-*"), reverse=True
        )
        for path in candidates:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "bin", "nvcc")):
                os.environ["CUDA_HOME"] = path
                break
        else:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                os.environ["CUDA_HOME"] = os.path.dirname(
                    os.path.dirname(os.path.realpath(nvcc_path))
                )

    if not os.environ.get("CC"):
        for ver in ["13", "12", "11"]:
            gcc = f"/opt/rh/gcc-toolset-{ver}/root/usr/bin/gcc"
            gpp = f"/opt/rh/gcc-toolset-{ver}/root/usr/bin/g++"
            if os.path.isfile(gcc) and os.path.isfile(gpp):
                os.environ["CC"] = gcc
                os.environ["CXX"] = gpp
                break


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext
    _setup_build_env()
    from torch.utils.cpp_extension import load

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    _ext = load(
        name="ctc_blank_only_cuda",
        sources=[os.path.join(src_dir, "ctc_blank_only.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return _ext


class CTCBlankOnlyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity):
        ext = _load_extension()
        input_lengths_long = input_lengths.to(dtype=torch.long)
        target_lengths_long = target_lengths.to(dtype=torch.long)
        neg_log_likelihood, log_alpha = ext.forward(
            log_probs,
            targets,
            input_lengths_long,
            target_lengths_long,
            blank,
            zero_infinity,
        )
        ctx.save_for_backward(
            log_probs,
            targets,
            neg_log_likelihood,
            log_alpha,
            input_lengths_long,
            target_lengths_long,
        )
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity
        return neg_log_likelihood

    @staticmethod
    def backward(ctx, grad_output):
        (
            log_probs,
            targets,
            neg_log_likelihood,
            log_alpha,
            input_lengths,
            target_lengths,
        ) = ctx.saved_tensors
        ext = _load_extension()
        grad = ext.backward(
            grad_output,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            neg_log_likelihood,
            log_alpha,
            ctx.blank,
            ctx.zero_infinity,
        )
        return grad, None, None, None, None, None


def ctc_loss_blank_only(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    if targets.dim() == 2:
        targets_flat = targets.reshape(-1)
    else:
        targets_flat = targets

    neg_log_likelihood = CTCBlankOnlyFunction.apply(
        log_probs, targets_flat, input_lengths, target_lengths, blank, zero_infinity
    )

    if zero_infinity:
        neg_log_likelihood = torch.where(
            neg_log_likelihood == float("inf"),
            torch.zeros_like(neg_log_likelihood),
            neg_log_likelihood,
        )

    if reduction == "mean":
        return (neg_log_likelihood / target_lengths.clamp(min=1)).mean()
    if reduction == "sum":
        return neg_log_likelihood.sum()
    return neg_log_likelihood
