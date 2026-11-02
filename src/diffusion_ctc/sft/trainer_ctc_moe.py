"""CTC-augmented SFT trainer/collator for LLaDA-MoE.

Same algorithm as `trainer_ctc.py`: a single per-example
``t ~ U[0, 1]`` drives both the slack insertion
``n = min(floor(s_max * (1 - t) * R), A)`` (``s_max`` = ``slack_max_ratio``,
default 0.5) and the diffusion noising. The
trailing EOT/EOS are excluded from <slack> insertion but are noised
normally. Token-id constants come from ``tokens.LLADA_MOE``. Optionally
adds the load-balancing aux loss exposed by ``LLaDAMoEModelLM`` (see note
on ``use_router_aux_loss`` in ``trainer_moe``).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import wandb
from transformers import DefaultDataCollator, Trainer

from ..ctc_loss import ctc_loss_blank_only
from ..tokens import LLADA_MOE

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BLANK_TOKEN_ID = LLADA_MOE.blank
PAD_TOKEN_ID = LLADA_MOE.pad
EOT_TOKEN_ID = LLADA_MOE.eot
EOS_TOKEN_ID = LLADA_MOE.eos


class dLLMTrainer(Trainer):
    def __init__(
        self,
        *args,
        ctc_weight=0.1,
        ce_blank_weight=1.0,
        use_router_aux_loss: bool = False,
        router_aux_loss_coef: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ctc_weight = ctc_weight
        # ce_blank_weight=1.0 means uniform per-class weighting in CE — the
        # "standard absorbing-state formulation" used by the paper.
        self.ce_blank_weight = ce_blank_weight
        self.eval_ce_losses = []
        self.eval_ctc_losses = []
        self.use_router_aux_loss = use_router_aux_loss
        if router_aux_loss_coef is None:
            router_aux_loss_coef = float(
                getattr(self.model.config, "router_aux_loss_coef", 0.0) or 0.0
            )
        self.router_aux_loss_coef = router_aux_loss_coef

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        num_prompt_tokens = inputs.pop("num_prompt_tokens")

        clean_targets = inputs.pop("clean_targets")
        target_lengths = inputs.pop("target_lengths")
        input_lengths = inputs.pop("input_lengths")
        response_start_idx = inputs.pop("response_start_idx")

        self._debug_print_first_batch(
            inputs, clean_targets, target_lengths, input_lengths, response_start_idx, labels
        )

        if self.use_router_aux_loss:
            inputs["output_router_logits"] = True

        outputs = model(**inputs)
        logits = outputs.logits

        ce_weights = torch.ones(
            logits.size(-1), device=logits.device, dtype=logits.dtype
        )
        ce_weights[BLANK_TOKEN_ID] = self.ce_blank_weight

        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            weight=ce_weights,
            reduction="none",
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(
                {"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()}
            )
        loss = unscaled_loss / t
        ce_loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)

        # --- CTC loss ---
        batch_size, _, vocab_size = logits.shape
        device = logits.device
        clean_targets = clean_targets.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = input_lengths.to(device)
        response_start_idx = response_start_idx.to(device)

        max_resp_len = int(input_lengths.max().item()) if input_lengths.numel() else 0
        if max_resp_len > 0:
            response_logits = logits.new_zeros(
                (batch_size, max_resp_len, vocab_size), dtype=logits.dtype
            )
            for i in range(batch_size):
                start = int(response_start_idx[i].item())
                length = int(input_lengths[i].item())
                if length == 0:
                    continue
                end = start + length
                response_slice = logits[i, start:end]
                response_logits[i, : response_slice.size(0)] = response_slice

            log_probs = response_logits.log_softmax(dim=-1).transpose(0, 1)
            log_probs = log_probs.to(torch.float32)
            ctc_loss = ctc_loss_blank_only(
                log_probs,
                clean_targets,
                input_lengths,
                target_lengths,
                blank=BLANK_TOKEN_ID,
                zero_infinity=True,
            )
            ctc_skipped = False
        else:
            ctc_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
            ctc_skipped = True

        total_loss = ce_loss + self.ctc_weight * ctc_loss

        aux_loss_value = None
        if self.use_router_aux_loss:
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                total_loss = total_loss + self.router_aux_loss_coef * aux_loss.to(
                    total_loss.device
                )
                aux_loss_value = float(aux_loss.detach().item())

        if not model.training:
            self.eval_ce_losses.append(ce_loss.detach().item())
            self.eval_ctc_losses.append(ctc_loss.detach().item())

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self._log_metrics_and_text(
                ce_loss,
                logits,
                ctc_loss,
                ctc_skipped,
                blank_prob=None,
                clean_targets=clean_targets,
                target_lengths=target_lengths,
                input_lengths=input_lengths,
                response_start_idx=response_start_idx,
                attention_mask=inputs["attention_mask"],
                aux_loss_value=aux_loss_value,
            )

        return total_loss if not return_outputs else (total_loss, outputs)

    def _log_metrics_and_text(
        self,
        ce_loss,
        logits,
        ctc_loss,
        ctc_skipped,
        blank_prob,
        clean_targets,
        target_lengths,
        input_lengths,
        response_start_idx,
        attention_mask,
        aux_loss_value=None,
    ):
        with torch.no_grad():
            if blank_prob is None:
                blank_prob = F.softmax(logits, dim=-1)[..., BLANK_TOKEN_ID].mean()

            pad_mask = attention_mask == 0
            vocab_size = logits.size(-1)
            if not hasattr(self, "_debug_pad_vocab_printed"):
                self._debug_pad_vocab_printed = True
                print(
                    f"[PAD DEBUG] logits.shape={tuple(logits.shape)}, vocab_size={vocab_size}, PAD_TOKEN_ID={PAD_TOKEN_ID}"
                )
            if pad_mask.any():
                pad_logits = logits[..., PAD_TOKEN_ID]
                pad_probs = F.softmax(logits, dim=-1)[..., PAD_TOKEN_ID]
                pad_token_logit = pad_logits[pad_mask].mean()
                pad_token_prob = pad_probs[pad_mask].mean()
            else:
                pad_token_logit = torch.tensor(0.0, device=logits.device)
                pad_token_prob = torch.tensor(0.0, device=logits.device)

            preds = logits.argmax(dim=-1)
            blank_top1_ratio = (preds == BLANK_TOKEN_ID).float().mean()

            text_log_freq = 500
            if (
                self.state.global_step + 1
            ) % text_log_freq == 0 or self.state.global_step == 0:
                if self.processing_class and wandb.run is not None:
                    start = int(response_start_idx[0].item())
                    length = int(input_lengths[0].item())
                    ctc_input_pred_slice = preds[0, start : start + length]
                    ctc_input_text = self.processing_class.decode(
                        ctc_input_pred_slice, skip_special_tokens=False
                    )

                    t_len = int(target_lengths[0].item())
                    ctc_target_slice = clean_targets[:t_len]
                    ctc_target_text = self.processing_class.decode(
                        ctc_target_slice, skip_special_tokens=False
                    )

                    wandb.log(
                        {
                            "generated_samples": wandb.Table(
                                columns=[
                                    "Step",
                                    "CTC Input (Predicted w/ Blanks)",
                                    "CTC Target (Clean Text)",
                                ],
                                data=[
                                    [
                                        self.state.global_step,
                                        ctc_input_text,
                                        ctc_target_text,
                                    ]
                                ],
                            )
                        },
                        commit=False,
                    )

        log_payload = {
            "ce_loss": ce_loss.item(),
            "blank_prob": blank_prob.item(),
            "blank_top1_ratio": blank_top1_ratio.item(),
            "pad_token_logit": pad_token_logit.item(),
            "pad_token_prob": pad_token_prob.item(),
        }
        if not ctc_skipped:
            log_payload["ctc_loss"] = ctc_loss.item()
        else:
            log_payload["ctc_skipped"] = 1
        if aux_loss_value is not None:
            log_payload["router_aux_loss"] = aux_loss_value
        self.log(log_payload)

    def _debug_print_first_batch(
        self,
        inputs,
        clean_targets,
        target_lengths,
        input_lengths,
        response_start_idx,
        labels,
    ):
        if self.state.global_step == 0 and not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print("\n" + "=" * 50)
            print("DEBUG: First Batch Inspection [LLaDA-MoE]")
            print("=" * 50)

            tokenizer = self.processing_class

            if tokenizer:
                input_ids_0 = inputs["input_ids"][0]
                print(
                    f"\n[Input Text (with blanks)]: {tokenizer.decode(input_ids_0, skip_special_tokens=False)}"
                )

                target_len_0 = target_lengths[0]
                target_ids_0 = clean_targets[:target_len_0]
                print(
                    f"\n[CTC Target Text]: {tokenizer.decode(target_ids_0, skip_special_tokens=False)}"
                )

                start = response_start_idx[0].item()
                length = input_lengths[0].item()
                response_slice = input_ids_0[start : start + length]
                print(
                    f"\n[Response Slice (Input to CTC)]: {tokenizer.decode(response_slice, skip_special_tokens=False)}"
                )

                print(
                    f"\n[Lengths]: Input Length={length}, Target Length={target_len_0}"
                )
                print(f"[Response Start Index]: {start}")

                labels_0 = labels[0]
                masked_count = (labels_0 != -100).sum().item()
                print(f"[CE Labels]: {masked_count} tokens supervised ( != -100)")
            else:
                print(
                    "WARNING: No tokenizer/processing_class found for debug printing."
                )
            print("=" * 50 + "\n")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.eval_ce_losses = []
        self.eval_ctc_losses = []
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.eval_ce_losses:
            metrics[f"{metric_key_prefix}_ce_loss"] = sum(self.eval_ce_losses) / len(
                self.eval_ce_losses
            )
        if self.eval_ctc_losses:
            metrics[f"{metric_key_prefix}_ctc_loss"] = sum(self.eval_ctc_losses) / len(
                self.eval_ctc_losses
            )

        self.log(metrics)
        return metrics


class dLLMSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = kwargs.get("processing_class")

        if self.tokenizer is None:
            raise ValueError(
                "dLLMDataCollator requires a tokenizer (passed as 'processing_class')"
            )

        self.mask_token_id = self.tokenizer.mask_token_id

        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]

        # Maximum <slack> ratio s_max in n = min(floor(s_max * (1 - t) * R), A).
        # 0.5 reproduces the main experiments; 0.2 / 0.8 reproduce the
        # slack-ratio ablation (paper Sec. 6.4).
        self.slack_max_ratio = float(kwargs.get("slack_max_ratio", 0.5))
        if not 0.0 <= self.slack_max_ratio <= 1.0:
            raise ValueError("slack_max_ratio must be within [0, 1].")

        if self.mask_token_id is None:
            assert "mask_token_id" in kwargs, (
                "For dLLM models, pass a mask_token_id or set tokenizer.mask_token_id"
            )
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]

        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, features):
        features = [feature.copy() for feature in features]
        clean_responses = []
        for feature in features:
            item = feature.pop("clean_response_ids")
            if isinstance(item, list):
                item = torch.tensor(item, dtype=torch.long)
            clean_responses.append(item)

        clean_response_lengths = torch.tensor(
            [
                (
                    feature.pop("clean_response_length").item()
                    if hasattr(feature["clean_response_length"], "item")
                    else feature.pop("clean_response_length")
                )
                for feature in features
            ],
            dtype=torch.long,
        )
        for feature in features:
            feature.pop("original_length", None)
            feature.pop("blanked_response_length", None)

        batch = super().__call__(features)
        base_attention_mask = batch["attention_mask"].clone()
        input_ids = batch["input_ids"]
        B, Lmax = input_ids.shape

        # Sample t once per example and reuse it for both <slack> insertion and
        # diffusion noising. The paper specifies a single per-example
        # t ~ U[0, 1]; storing it in batch["t"] makes `forward_process`
        # reuse exactly the same value below.
        if "t" in batch:
            t_scalar = batch["t"].to(input_ids.device).float()
        else:
            t_scalar = torch.rand((B,), device=input_ids.device)
            batch["t"] = t_scalar

        s_max = self.slack_max_ratio
        slack_ratio = s_max * (1 - t_scalar)

        response_start_idx = batch["response_start_idx"].to(input_ids.device)

        new_input_lengths = []
        ce_input_lengths = []
        debug_info = None
        for i in range(B):
            start = int(response_start_idx[i].item())

            base_valid_len = int(base_attention_mask[i].sum().item())

            base_resp_len = max(0, base_valid_len - start)

            resp = input_ids[i, start : start + base_resp_len]

            # Strip trailing EOS/EOT (any order); we re-emit them in canonical
            # order at the end.
            if resp.numel() > 0 and resp[-1].item() == EOS_TOKEN_ID:
                resp = resp[:-1]
            if resp.numel() > 0 and resp[-1].item() == EOT_TOKEN_ID:
                resp = resp[:-1]

            stripped_resp_len = resp.size(0)

            pre_blanks = (resp == BLANK_TOKEN_ID).sum().item()

            cap = Lmax - start
            reserved_suffix = 2  # EOT + EOS
            avail = cap - stripped_resp_len - reserved_suffix

            if avail <= 0:
                num_blanks = 0
                desired = 0
            else:
                # Paper formula: n = min(floor(s_max * (1 - t) * R), A).
                desired = int(slack_ratio[i].item() * stripped_resp_len)
                num_blanks = min(desired, avail)

            new_resp = self._insert_blanks_into_response(resp, num_blanks)

            new_len = int(new_resp.size(0))
            new_input_lengths.append(new_len)

            if i == 0:
                debug_info = {
                    "start": start,
                    "base_resp_len": base_resp_len,
                    "stripped_resp_len": stripped_resp_len,
                    "pre_blanks": pre_blanks,
                    "desired": desired,
                    "num_blanks": num_blanks,
                    "avail": avail,
                    "cap": cap,
                    "t_scalar": float(t_scalar[i].item()),
                }

            input_ids[i, start : start + new_len] = new_resp

            curr_pos = start + new_len
            extra_tokens = 0

            if curr_pos < Lmax:
                input_ids[i, curr_pos] = EOT_TOKEN_ID
                curr_pos += 1
                extra_tokens += 1

            if curr_pos < Lmax:
                input_ids[i, curr_pos] = EOS_TOKEN_ID
                curr_pos += 1
                extra_tokens += 1

            ce_input_lengths.append(new_len + extra_tokens)

            if curr_pos < Lmax:
                input_ids[i, curr_pos:] = EOS_TOKEN_ID

        batch["input_ids"] = input_ids
        batch["input_lengths"] = torch.tensor(
            new_input_lengths, device=input_ids.device, dtype=torch.long
        )

        ce_input_lengths_tensor = torch.tensor(
            ce_input_lengths, device=input_ids.device, dtype=torch.long
        )

        active_len = response_start_idx + ce_input_lengths_tensor
        idx = torch.arange(Lmax, device=input_ids.device).unsqueeze(0)
        active_mask = idx < active_len.unsqueeze(1)

        batch["attention_mask"] = torch.ones_like(batch["attention_mask"])

        if not hasattr(self, "_debug_printed_collator"):
            self._debug_printed_collator = True
            print("\n" + "=" * 50)
            print("DEBUG: Collator Input (Before Masking) [LLaDA-MoE CTC]")
            print("=" * 50)
            if self.tokenizer:
                input_ids_0 = batch["input_ids"][0]
                print(
                    f"\n[Input Text (Raw Padded)]: {self.tokenizer.decode(input_ids_0, skip_special_tokens=False)}"
                )
                print(f"[Input IDs]: {input_ids_0.tolist()}")
                print(
                    f"[Lmax]: {Lmax}, [active_len[0]]: {int(active_len[0].item())}, [max active_len]: {int(active_len.max().item())}"
                )
                start0 = int(response_start_idx[0].item())
                resp_len0 = int(batch["input_lengths"][0].item())
                resp_slice0 = input_ids_0[start0 : start0 + resp_len0]
                if resp_len0 > 0:
                    blank_ratio0 = (resp_slice0 == BLANK_TOKEN_ID).float().mean().item()
                else:
                    blank_ratio0 = 0.0
                print(
                    f"[Blank ratio in response slice 0]: {blank_ratio0:.4f} (len={resp_len0})"
                )
                if debug_info is not None:
                    print(
                        f"[Slack debug] start={debug_info['start']}, base_resp_len={debug_info['base_resp_len']}, "
                        f"pre_blanks={debug_info['pre_blanks']}, desired={debug_info['desired']}, "
                        f"num_blanks={debug_info['num_blanks']}, avail={debug_info['avail']}, cap={debug_info['cap']}, "
                        f"t_scalar={debug_info['t_scalar']:.4f}"
                    )
            print("=" * 50 + "\n")

        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        # Standard absorbing-state CE: only positions actually masked by the
        # forward process contribute to CE. The trailing EOT/EOS are excluded
        # from <slack> insertion (so the response always ends with them prior
        # to noising) but are noised normally — same as any other token — to
        # match the paper description.
        batch["labels"][~mask_indices] = -100

        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_lengths = torch.as_tensor(prompt_lengths, device=input_ids.device)
            prompt_lengths = prompt_lengths.view(-1)
            prompt_length_indices = torch.arange(
                noisy_batch.shape[1], device=input_ids.device
            ).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths.unsqueeze(1)
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()

        batch["labels"][~active_mask] = -100

        batch["input_ids"] = noisy_batch.long()

        batch["clean_targets"] = torch.cat(clean_responses, dim=0)
        batch["target_lengths"] = clean_response_lengths
        return batch

    def _insert_blanks_into_response(self, resp_ids: torch.Tensor, num_blanks: int):
        if num_blanks <= 0:
            return resp_ids
        R = resp_ids.size(0)

        slots = torch.randperm(R + 1, device=resp_ids.device)[:num_blanks]
        slots, _ = torch.sort(slots)

        out = []
        prev = 0
        for s in slots.tolist():
            out.append(resp_ids[prev:s])
            out.append(
                torch.tensor(
                    [BLANK_TOKEN_ID], device=resp_ids.device, dtype=resp_ids.dtype
                )
            )
            prev = s
        out.append(resp_ids[prev:])
        return torch.cat(out, dim=0)
