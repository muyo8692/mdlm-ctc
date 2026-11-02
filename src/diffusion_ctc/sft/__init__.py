"""SFT trainers and data collators for the four supported variants:

* `trainer`        — vanilla SFT for LLaDA.
* `trainer_moe`    — vanilla SFT for LLaDA-MoE (optional router-load-balancing aux loss).
* `trainer_ctc`    — CTC-augmented SFT for LLaDA.
* `trainer_ctc_moe` — CTC-augmented SFT for LLaDA-MoE.
"""
