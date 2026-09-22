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

- `cache_threshold` defaults to 0.40, calibrated from measured drift on this
  model (~0.13 per step mid-schedule): the threshold is roughly 0.13 x the
  skip run length, so 0.3 skips ~2 steps per refresh, 0.5 ~3-4, 0.8 ~6. The
  node logs how many forwards were skipped at the end of each run; with
  `debug_log` on it also reports the measured error of replayed residuals vs
  actual full forwards (mean/max), which is the quantity to watch while
  raising the threshold.
- `sparse_tau` 1.3 keeps ~11% of key blocks; 1.0 is closer to dense, 1.5+ is
  aggressive. Sparse attention only pays off on long sequences (2K+
  resolutions, long multi-image prefixes); at 1MP it is usually slower than
  dense, hence the 8192 default for `sparse_min_tokens`.
