# Qwen Image 2.1 Speedup

Sampling accelerator for Qwen Image 2.1 in ComfyUI.

## Node

- Display name: `Qwen Image 2.1 Speedup`
- Class ID: `QwenImage21Speedup`
- Category: `model/patch`

## What it does

Two orthogonal speedups in one node, both compatible with the model's built-in
prefix K/V cache (text + reference image K/V computed once per sampling run):

1. **Residual caching (default on)** — TeaCache-style: while the accumulated
   relative drift of the timestep embedding stays under `cache_threshold`, the
   whole transformer forward is skipped and a cached `output - input`
   residual is applied instead. With `enable_forecast` (default on) the
   applied residual is a linear extrapolation in sigma space from the last two
   measured residuals (TaylorCache-style), clamped to 1.5 measured intervals.
   The first/last stretch of the schedule (`cache_start_percent` /
   `cache_end_percent`) and `max_consecutive_skips` bound the approximation.
   CFG positive/negative streams are cached independently.
2. **Sparse attention (optional)** — a preset wrapper around ComfyUI's
   official Sol-Attn block-sparse attention (`Model Sparse Attention` node
   internals). Only image segments go sparse; text segments keep their causal
   mask and stay dense. Gains grow with resolution; `sparse_min_tokens`
   (default 2048) engages it from roughly 1MP images up.

## Recommended chain

```
UNETLoader → Qwen Image 2.1 Speedup → KSampler
```

Add the official `Qwen Image 2.1 Cache` node after this one when VRAM is
tight (int8 prefix K/V). Nothing else is required: the mixed-granularity
attention and the prefix K/V reuse from the Qwen Image 2.1 architecture are
already built into ComfyUI.

## Tuning

- **Adaptive mode (recommended)**: set `target_error` (0.05 is strict,
  0.06-0.08 trades a little fidelity for clearly more speed) and the node
  closes the loop on quality: after every skip run it measures the actual
  error of the replayed residual against a real forward and multiplies the
  effective threshold by 0.7x-1.3x to hold the error near the target. If the
  controller shrinks the threshold below one step's drift so no skip is
  possible, three starved steps raise it by 1.5x to re-probe. This adapts to
  step count, resolution and prompt automatically. `cache_threshold` is the
  starting point in this mode.
- **Keep `enable_forecast` on.** Measured on 25-step runs: with forecast a
  single skip costs ~0.045 relative error, without it ~0.07-0.11 — forecast
  is what makes skipping affordable at low step counts, and in adaptive mode
  it is the difference between the controller holding at ~44% skipped and
  starving down to 20%.
- **Fixed mode** (`target_error = 0`): `cache_threshold` defaults to 0.40,
  calibrated from measured drift on this model (~0.13 per step mid-schedule
  at 40 steps): the threshold is roughly 0.13 x the skip run length, so 0.3
  skips ~2 steps per refresh, 0.5 ~3-4, 0.8 ~6.
- Keep `cache_start_percent` >= 0.15: measured replay error peaks right at
  the schedule start.
- The node logs the skip rate at the end of each run; with `debug_log` (or
  adaptive mode) it also reports the measured replay error mean/max and the
  final effective threshold.
- `sparse_tau` 1.3 keeps ~11% of key blocks; 1.0 is closer to dense, 1.5+ is
  aggressive. Sparse attention only pays off on long sequences (2K+
  resolutions, long multi-image prefixes); at 1MP it is usually slower than
  dense, hence the 8192 default for `sparse_min_tokens`.
