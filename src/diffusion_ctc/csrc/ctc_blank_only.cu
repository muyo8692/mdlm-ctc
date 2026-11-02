// CTC loss variant: blank-only removal (no duplicate merging).
// Based on PyTorch v2.6.0 LossCTC.cu (Copyright (c) 2018 MathInf GmbH, Thomas Viehmann, BSD-3-Clause).
//
// Modifications from standard CTC (4 lines changed):
//   1. Self-loop (staying in the same state) only allowed for BLANK states (even s).
//   2. Skip transitions (s-2 -> s) allowed for ALL label states unconditionally.
// These make the decoding rule "remove blanks only" (no duplicate merging).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/macros/Macros.h>

namespace {

template <typename target_t>
__device__ static inline int64_t get_target_prime(
    const target_t* __restrict__ target,
    int64_t offset,
    int64_t stride,
    int64_t idx,
    int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

// ==================== FORWARD (alpha) kernel ====================
template <typename scalar_t, typename target_t>
__global__ void ctc_blank_only_log_alpha_kernel(
    scalar_t* __restrict__ log_alpha_data,
    const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths,
    int64_t max_input_length,
    const target_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths,
    int64_t max_target_length,
    scalar_t* __restrict__ neg_log_likelihood_data,
    int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
    const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
    int64_t batch_size, int64_t BLANK) {

  constexpr scalar_t neginf = -INFINITY;

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= batch_size)
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b * lp_batch_stride;
  int64_t la_batch_offset = b * la_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (input_length == 0) {
    if (threadIdx.x == 0) {
      scalar_t log_likelihood = target_length == 0 ? 0 : neginf;
      neg_log_likelihood_data[b] = -log_likelihood;
    }
    return;
  }

  for (int64_t block_s = 0; block_s < 2 * max_target_length + 1; block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t la;
    switch (s) {
      case 0:
        la = log_probs_data[lp_batch_offset + lp_char_stride * BLANK];
        break;
      case 1:
        la = target_length == 0 ? neginf
            : log_probs_data[lp_batch_offset + lp_char_stride *
                  get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK)];
        break;
      default:
        la = neginf;
    }
    if (s < 2 * max_target_length + 1)
      log_alpha_data[la_batch_offset + la_target_stride * s] = la;
  }

  for (int64_t block_s = 0; block_s < 2 * max_target_length + 1; block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;

    int64_t current_char;
    bool have_three;
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_char = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      // MODIFIED: skip allowed for ALL label states (odd s), unconditionally
      have_three = ((s > 1) && (s % 2 == 1));
    } else {
      current_char = BLANK;
      have_three = false;
    }

    for (int64_t t = 1; t < max_input_length; t++) {
      __syncthreads();
      if ((t < input_length) && (s < 2 * target_length + 1)) {
        // MODIFIED: self-loop only for blank states (even s)
        scalar_t la1 = (s % 2 == 0)
            ? log_alpha_data[la_batch_offset + la_input_stride * (t - 1) + la_target_stride * s]
            : neginf;
        scalar_t lamax = la1;
        scalar_t la2, la3;
        if (s > 0) {
          la2 = log_alpha_data[la_batch_offset + la_input_stride * (t - 1) + la_target_stride * (s - 1)];
          if (la2 > lamax) lamax = la2;
        } else {
          la2 = neginf;
        }
        if (have_three) {
          la3 = log_alpha_data[la_batch_offset + la_input_stride * (t - 1) + la_target_stride * (s - 2)];
          if (la3 > lamax) lamax = la3;
        } else {
          la3 = neginf;
        }
        if (lamax == neginf) lamax = 0;

        log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] =
            std::log(std::exp(la1 - lamax) + std::exp(la2 - lamax) + std::exp(la3 - lamax)) + lamax
            + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * current_char];
      } else {
        if (s < 2 * max_target_length + 1)
          log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = neginf;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    scalar_t l1 = log_alpha_data[la_batch_offset + la_input_stride * (input_length - 1) + la_target_stride * (target_length * 2)];
    scalar_t l2 = target_length > 0
        ? log_alpha_data[la_batch_offset + la_input_stride * (input_length - 1) + la_target_stride * (target_length * 2 - 1)]
        : neginf;
    scalar_t m = ((l1 > l2) ? l1 : l2);
    m = ((m == neginf) ? 0 : m);
    scalar_t log_likelihood = std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
    neg_log_likelihood_data[b] = -log_likelihood;
  }
}


// ==================== BACKWARD (beta) kernel ====================
template <typename scalar_t, typename target_t>
__global__ void
C10_LAUNCH_BOUNDS_2((std::is_same_v<scalar_t, float> ? 1024 : 896), 1)
ctc_blank_only_log_beta_kernel(
    scalar_t* __restrict__ log_beta_data,
    const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths,
    int64_t max_input_length,
    const target_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths,
    int64_t max_target_length,
    int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
    const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
    int64_t batch_size, int64_t BLANK) {

  constexpr scalar_t neginf = -INFINITY;

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= batch_size)
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b * lp_batch_stride;
  int64_t lb_batch_offset = b * lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (input_length == 0)
    return;

  for (int64_t block_s = 2 * max_target_length - (2 * max_target_length % blockDim.x);
       block_s >= 0; block_s -= blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t lb;
    if (s == 2 * target_length) {
      lb = log_probs_data[lp_batch_offset + (input_length - 1) * lp_input_stride + lp_char_stride * BLANK];
    } else if (s == 2 * target_length - 1) {
      int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      lb = log_probs_data[lp_batch_offset + (input_length - 1) * lp_input_stride + lp_char_stride * current_target_prime];
    } else {
      lb = neginf;
    }
    if (s < 2 * max_target_length + 1) {
      log_beta_data[lb_batch_offset + (input_length - 1) * lb_input_stride + lb_target_stride * s] = lb;
    }
  }

  for (int64_t block_s = 2 * max_target_length - (2 * max_target_length % blockDim.x);
       block_s >= 0; block_s -= blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    int64_t current_target_prime;
    bool have_three;
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      // MODIFIED: skip allowed for ALL label states (odd s), unconditionally
      have_three = ((s < 2 * target_length - 1) && (s % 2 == 1));
    } else {
      current_target_prime = BLANK;
      have_three = false;
    }

    for (int64_t t = max_input_length - 2; t >= 0; t--) {
      __syncthreads();
      if ((t < input_length - 1) && (s < 2 * target_length + 1)) {
        // MODIFIED: self-loop only for blank states (even s)
        scalar_t lb1 = (s % 2 == 0)
            ? log_beta_data[lb_batch_offset + lb_input_stride * (t + 1) + lb_target_stride * s]
            : neginf;
        scalar_t lbmax = lb1;
        scalar_t lb2, lb3;

        if (s < 2 * target_length) {
          lb2 = log_beta_data[lb_batch_offset + lb_input_stride * (t + 1) + lb_target_stride * (s + 1)];
          if (lb2 > lbmax) lbmax = lb2;
        } else {
          lb2 = neginf;
        }
        if (have_three) {
          lb3 = log_beta_data[lb_batch_offset + lb_input_stride * (t + 1) + lb_target_stride * (s + 2)];
          if (lb3 > lbmax) lbmax = lb3;
        } else {
          lb3 = neginf;
        }
        if (lbmax == neginf) lbmax = 0;

        scalar_t lb = std::log(std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) + std::exp(lb3 - lbmax)) + lbmax
            + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * current_target_prime];

        log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s] = lb;
      } else if (
          (s < 2 * max_target_length + 1) &&
          (((target_length == 0) && (s > 0)) || (s >= 2 * target_length + 1) || (t >= input_length))) {
        log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s] = neginf;
      }
    }
  }
}


// ==================== BACKWARD: collect nonblank gradients (large path) ====================
template <typename scalar_t, typename target_t>
__global__ void ctc_blank_only_backward_collect_nonblank_kernel(
    scalar_t* __restrict__ gradient_data,
    const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
    const scalar_t* __restrict__ log_alpha_data,
    const scalar_t* __restrict__ log_beta_data,
    const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths,
    const target_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths,
    const scalar_t* __restrict__ neg_log_likelihood_data,
    int64_t gr_input_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
    int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
    int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
    const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
    int64_t batch_size, bool zero_infinity) {

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t s = threadIdx.x + blockIdx.x * blockDim.x;

  if (b >= batch_size)
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t gr_batch_offset = b * gr_batch_stride;
  int64_t lp_batch_offset = b * lp_batch_stride;
  int64_t la_batch_offset = b * la_batch_stride;
  int64_t lb_batch_offset = b * lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (s >= target_length)
    return;

  int64_t target = targets_data[tg_batch_offset + s * tg_target_stride];
  scalar_t nll = neg_log_likelihood_data[b];
  scalar_t gr = grad_out_data[b * grad_out_batch_stride];

  if (zero_infinity && nll == INFINITY)
    return;

  for (int64_t t = 0; t < input_length; t++) {
    scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * target];
    gpuAtomicAddNoReturn(
        &gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * target],
        -std::exp(log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * (s * 2 + 1)]
            + log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * (s * 2 + 1)]
            + nll - lp) * gr);
  }
}


// ==================== BACKWARD: collect all gradients (small/naive path) ====================
template <typename scalar_t, typename target_t>
__global__ void ctc_blank_only_backward_collect_kernel(
    scalar_t* __restrict__ gradient_data,
    const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
    const scalar_t* __restrict__ log_alpha_data,
    const scalar_t* __restrict__ log_beta_data,
    const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths, int64_t max_input_length,
    const target_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths, int64_t max_target_length,
    const scalar_t* __restrict__ neg_log_likelihood_data,
    int64_t gr_input_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
    int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
    int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
    const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
    int64_t batch_size, int64_t num_labels, int64_t BLANK, bool zero_infinity) {

  constexpr scalar_t neginf = -INFINITY;
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t t = threadIdx.x + blockIdx.x * blockDim.x;

  if ((t >= max_input_length) || (b >= batch_size))
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t gr_batch_offset = b * gr_batch_stride;
  int64_t lp_batch_offset = b * lp_batch_stride;
  int64_t la_batch_offset = b * la_batch_stride;
  int64_t lb_batch_offset = b * lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  for (int s = 0; s < 2 * max_target_length + 1; s++) {
    if (s < 2 * target_length + 1) {
      int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      scalar_t log_alpha_beta =
          log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s]
          + log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s];
      scalar_t& lcab = gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * current_target_prime];
      if (lcab == neginf) {
        lcab = log_alpha_beta;
      } else {
        scalar_t max = ((lcab > log_alpha_beta) ? lcab : log_alpha_beta);
        lcab = std::log(std::exp(lcab - max) + std::exp(log_alpha_beta - max)) + max;
      }
    }
  }

  scalar_t nll = neg_log_likelihood_data[b];
  scalar_t gr = grad_out_data[b * grad_out_batch_stride];

  for (int64_t c = 0; c < num_labels; c++) {
    scalar_t& res = gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * c];
    if (t < input_length && (!zero_infinity || nll != INFINITY)) {
      scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * c];
      res = (std::exp(lp) - std::exp(res + nll - lp)) * gr;
    } else {
      res = 0.;
    }
  }
}


// ==================== Zero padded gradients kernel ====================
template <typename scalar_t>
__global__ void ctc_blank_only_zero_padded_gradients(
    scalar_t* __restrict__ gradient_data,
    const int64_t* __restrict__ input_lengths,
    int64_t gr_timestep_stride, int64_t gr_batch_stride, int64_t gr_label_stride,
    int64_t max_input_length, int64_t batch_size, int64_t num_labels) {

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t t = threadIdx.x + blockIdx.x * blockDim.x;

  if (b >= batch_size || t >= max_input_length)
    return;

  if (t >= input_lengths[b]) {
    for (int l = 0; l < num_labels; l++)
      gradient_data[t * gr_timestep_stride + b * gr_batch_stride + l * gr_label_stride] = 0.0f;
  }
}


// ==================== C++ wrapper: forward ====================
template <typename scalar_t, at::ScalarType target_scalar_type>
std::tuple<at::Tensor, at::Tensor> ctc_blank_only_forward_impl(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths_t,
    const at::Tensor& target_lengths_t,
    int64_t BLANK) {

  using target_t = typename std::conditional<target_scalar_type == at::kInt, int, int64_t>::type;

  int64_t batch_size = log_probs.size(1);
  int64_t max_input_length = log_probs.size(0);

  auto il_cpu = input_lengths_t.to(at::kCPU, at::kLong).contiguous();
  auto tl_cpu = target_lengths_t.to(at::kCPU, at::kLong).contiguous();
  auto il_data = il_cpu.data_ptr<int64_t>();
  auto tl_data = tl_cpu.data_ptr<int64_t>();

  int64_t tg_target_stride;
  int64_t max_target_length = 0;
  auto tg_batch_offsets = at::empty({batch_size}, at::TensorOptions(at::kCPU).dtype(at::kLong));
  auto tg_bo_data = tg_batch_offsets.mutable_data_ptr<int64_t>();

  if (targets.dim() == 1) {
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_bo_data[i] = pos;
      pos += tl_data[i];
      if (max_target_length < tl_data[i]) max_target_length = tl_data[i];
    }
    tg_target_stride = targets.stride(0);
  } else {
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_bo_data[i] = i * tg_batch_stride;
      if (max_target_length < tl_data[i]) max_target_length = tl_data[i];
    }
    tg_target_stride = targets.stride(1);
  }

  auto input_lengths_gpu = input_lengths_t.to(log_probs.device(), at::kLong).contiguous();
  auto target_lengths_gpu = target_lengths_t.to(log_probs.device(), at::kLong).contiguous();
  tg_batch_offsets = tg_batch_offsets.to(log_probs.device());

  at::Tensor log_alpha = at::empty({batch_size, max_input_length, 2 * max_target_length + 1}, log_probs.options());
  at::Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  constexpr int max_threads = std::is_same_v<scalar_t, float> ? 1024 : 768;
  int threads_target = max_threads;
  while (threads_target / 2 >= 2 * max_target_length + 1) threads_target /= 2;
  int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
  dim3 block(threads_target, threads_batch);
  dim3 grid(1, (batch_size + threads_batch - 1) / threads_batch);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ctc_blank_only_log_alpha_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>(
      log_alpha.mutable_data_ptr<scalar_t>(),
      log_probs.const_data_ptr<scalar_t>(),
      input_lengths_gpu.const_data_ptr<int64_t>(),
      max_input_length,
      targets.const_data_ptr<target_t>(),
      target_lengths_gpu.const_data_ptr<int64_t>(),
      max_target_length,
      neg_log_likelihood.mutable_data_ptr<scalar_t>(),
      log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
      log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
      tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
      batch_size, BLANK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(neg_log_likelihood, log_alpha);
}


// ==================== C++ wrapper: backward ====================
template <typename scalar_t, at::ScalarType target_scalar_type>
at::Tensor ctc_blank_only_backward_impl(
    const at::Tensor& grad_out,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths_t,
    const at::Tensor& target_lengths_t,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {

  using target_t = typename std::conditional<target_scalar_type == at::kInt, int, int64_t>::type;
  constexpr scalar_t neginf = -INFINITY;

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);

  auto il_cpu = input_lengths_t.to(at::kCPU, at::kLong).contiguous();
  auto tl_cpu = target_lengths_t.to(at::kCPU, at::kLong).contiguous();
  auto tl_data = tl_cpu.data_ptr<int64_t>();

  int64_t tg_target_stride;
  int64_t max_target_length;
  auto tg_batch_offsets = at::empty({batch_size}, at::TensorOptions(at::kCPU).dtype(at::kLong));
  auto tg_bo_data = tg_batch_offsets.mutable_data_ptr<int64_t>();

  if (targets.dim() == 1) {
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_bo_data[i] = pos;
      pos += tl_data[i];
      if (max_target_length < tl_data[i]) max_target_length = tl_data[i];
    }
    tg_target_stride = targets.stride(0);
  } else {
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_bo_data[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length = log_alpha.size(2) / 2;
  }

  auto input_lengths_gpu = input_lengths_t.to(log_probs.device(), at::kLong).contiguous();
  auto target_lengths_gpu = target_lengths_t.to(log_probs.device(), at::kLong).contiguous();
  tg_batch_offsets = tg_batch_offsets.to(log_probs.device());

  at::Tensor log_beta = at::empty_like(log_alpha);
  log_beta.fill_(neginf);
  at::Tensor grad = at::full_like(log_probs, neginf);

  constexpr int max_threads = std::is_same_v<scalar_t, float> ? 1024 : 896;
  int threads_target = max_threads;
  while (threads_target / 2 >= 2 * max_target_length + 1) threads_target /= 2;
  int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  {
    dim3 block(threads_target, threads_batch);
    dim3 grid(1, (batch_size + threads_batch - 1) / threads_batch);
    ctc_blank_only_log_beta_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>(
        log_beta.mutable_data_ptr<scalar_t>(),
        log_probs.const_data_ptr<scalar_t>(),
        input_lengths_gpu.const_data_ptr<int64_t>(),
        log_probs.size(0),
        targets.const_data_ptr<target_t>(),
        target_lengths_gpu.const_data_ptr<int64_t>(),
        max_target_length,
        log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
        log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
        tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
        batch_size, BLANK);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  bool is_large = (2 * log_probs.size(0) + (24 * batch_size) / 10 + (2 * num_labels) / 10) > 450;
  if (is_large) {
    at::exp_out(grad, log_probs);
    auto grad_blank = grad.narrow(2, BLANK, 1);
    grad_blank -= (at::logsumexp(
        log_alpha.as_strided(
            {batch_size, log_alpha.size(1), max_target_length + 1},
            {log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2) * 2})
        + log_beta.as_strided(
            {batch_size, log_beta.size(1), max_target_length + 1},
            {log_beta.stride(0), log_beta.stride(1), log_beta.stride(2) * 2}),
        2, true)
        .permute({1, 0, 2})
        .add_(neg_log_likelihood.view({1, batch_size, 1}))
        .sub_(log_probs.narrow(2, BLANK, 1))
        .exp_());
    grad *= grad_out.view({1, batch_size, 1});
    if (zero_infinity) {
      grad = at::where(
          neg_log_likelihood.view({1, batch_size, 1}) == at::Scalar(INFINITY),
          at::zeros({}, grad.options()), grad);
    }

    int threads_tgt = max_threads;
    while (threads_tgt / 2 >= max_target_length && threads_tgt > 1) threads_tgt /= 2;
    int threads_bat = std::min(max_threads / threads_tgt, (int)batch_size);
    dim3 block2(threads_tgt, threads_bat);
    dim3 grid2(
        std::max((int64_t)((max_target_length + threads_tgt - 1) / threads_tgt), (int64_t)1),
        (batch_size + threads_bat - 1) / threads_bat, 1);
    ctc_blank_only_backward_collect_nonblank_kernel<scalar_t, target_t><<<grid2, block2, 0, stream>>>(
        grad.mutable_data_ptr<scalar_t>(),
        grad_out.const_data_ptr<scalar_t>(), grad_out.stride(0),
        log_alpha.const_data_ptr<scalar_t>(), log_beta.const_data_ptr<scalar_t>(),
        log_probs.const_data_ptr<scalar_t>(),
        input_lengths_gpu.const_data_ptr<int64_t>(),
        targets.const_data_ptr<target_t>(),
        target_lengths_gpu.const_data_ptr<int64_t>(),
        neg_log_likelihood.const_data_ptr<scalar_t>(),
        grad.stride(0), grad.stride(1), grad.stride(2),
        log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
        log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
        log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
        tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
        batch_size, zero_infinity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int threads_input = max_threads;
    while (threads_input / 2 >= log_probs.size(0) && threads_input > 1) threads_input /= 2;
    threads_batch = std::min(max_threads / threads_input, (int)batch_size);
    dim3 block2(threads_input, threads_batch);
    dim3 grid2(
        (log_probs.size(0) + threads_input - 1) / threads_input,
        (batch_size + threads_batch - 1) / threads_batch);
    ctc_blank_only_backward_collect_kernel<scalar_t, target_t><<<grid2, block2, 0, stream>>>(
        grad.mutable_data_ptr<scalar_t>(),
        grad_out.const_data_ptr<scalar_t>(), grad_out.stride(0),
        log_alpha.const_data_ptr<scalar_t>(), log_beta.const_data_ptr<scalar_t>(),
        log_probs.const_data_ptr<scalar_t>(),
        input_lengths_gpu.const_data_ptr<int64_t>(), log_probs.size(0),
        targets.const_data_ptr<target_t>(),
        target_lengths_gpu.const_data_ptr<int64_t>(), max_target_length,
        neg_log_likelihood.const_data_ptr<scalar_t>(),
        grad.stride(0), grad.stride(1), grad.stride(2),
        log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
        log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
        log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
        tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
        batch_size, num_labels, BLANK, zero_infinity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  {
    int threads_input = max_threads;
    while (threads_input / 2 >= log_probs.size(0)) threads_input /= 2;
    threads_batch = std::min(max_threads / threads_input, (int)batch_size);
    dim3 block3(threads_input, threads_batch);
    dim3 grid3(
        (log_probs.size(0) + threads_input - 1) / threads_input,
        (batch_size + threads_batch - 1) / threads_batch);
    ctc_blank_only_zero_padded_gradients<scalar_t><<<grid3, block3, 0, stream>>>(
        grad.mutable_data_ptr<scalar_t>(),
        input_lengths_gpu.const_data_ptr<int64_t>(),
        grad.stride(0), grad.stride(1), grad.stride(2),
        grad.size(0), grad.size(1), grad.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return grad;
}


// ==================== Python-facing dispatch functions ====================
std::tuple<at::Tensor, at::Tensor> ctc_blank_only_forward(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t BLANK,
    bool zero_infinity) {

  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_blank_only_forward", [&] {
    if (targets.scalar_type() == at::kLong) {
      return ctc_blank_only_forward_impl<scalar_t, at::kLong>(log_probs, targets, input_lengths, target_lengths, BLANK);
    } else {
      return ctc_blank_only_forward_impl<scalar_t, at::kInt>(log_probs, targets, input_lengths, target_lengths, BLANK);
    }
  });
}

at::Tensor ctc_blank_only_backward(
    const at::Tensor& grad_out,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {

  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_blank_only_backward", [&] {
    if (targets.scalar_type() == at::kLong) {
      return ctc_blank_only_backward_impl<scalar_t, at::kLong>(
          grad_out, log_probs, targets, input_lengths, target_lengths,
          neg_log_likelihood, log_alpha, BLANK, zero_infinity);
    } else {
      return ctc_blank_only_backward_impl<scalar_t, at::kInt>(
          grad_out, log_probs, targets, input_lengths, target_lengths,
          neg_log_likelihood, log_alpha, BLANK, zero_infinity);
    }
  });
}

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ctc_blank_only_forward, "CTC blank-only forward (CUDA)");
  m.def("backward", &ctc_blank_only_backward, "CTC blank-only backward (CUDA)");
}
