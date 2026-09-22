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
   whole transformer forward is skipped and the cached `output - input`
   residual is replayed instead. The first/last stretch of the schedule
   (`cache_start_percent` / `cache_end_percent`) and
   `max_consecutive_skips` bound the approximation. CFG positive/negative
   streams are cached independently.
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

- `cache_threshold` 0.10 is a conservative start; raise toward 0.2 for more
  speed, lower for fidelity. The node logs how many forwards were skipped at
  the end of each run.
- `sparse_tau` 1.3 keeps ~11% of key blocks; 1.0 is closer to dense, 1.5+ is
  aggressive.
