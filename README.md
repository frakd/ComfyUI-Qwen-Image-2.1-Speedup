# Qwen Image 2.1 Speedup

Sampling accelerator for Qwen Image 2.1 in ComfyUI.

## Node

- Display name: `Qwen Image 2.1 Speedup`
- Class ID: `QwenImage21Speedup`
- Category: `model/patch`

## What it does

Step-level residual caching, compatible with the model's built-in prefix K/V
cache (text + reference image K/V computed once per sampling run):

- **Skip gate (TeaCache-style)**: while the accumulated relative drift of the
  timestep embedding stays under `cache_threshold`, the whole transformer
  forward is skipped. The first/last stretch of the schedule
  (`cache_start_percent` / `cache_end_percent`) and `max_consecutive_skips`
  bound the approximation. CFG positive/negative streams are cached
  independently.
- **Residual forecast (TaylorCache-style)**: the residual applied on a cached
  step is extrapolated in sigma space from the measured residual history —
  `first` uses the last two residuals (secant), `second` adds the quadratic
  term from the last three (clamped to the linear term's magnitude), `off`
  replays the latest residual verbatim.
- **Adaptive mode**: with `target_error` > 0, the actual error of each skip
  run is measured against the next real forward and the effective threshold
  is adjusted (0.7x-1.3x per correction) to hold the error near the target.
  Three starved steps raise the threshold by 1.5x so the controller can
  always re-probe upward.

## Recommended chain

```
UNETLoader → Qwen Image 2.1 Speedup → KSampler
```

Add the official `Qwen Image 2.1 Cache` node after this one when VRAM is
tight (int8 prefix K/V). Nothing else is required: the mixed-granularity
attention and the prefix K/V reuse from the Qwen Image 2.1 architecture are
already built into ComfyUI.

## Tuning

Designed and calibrated for 40-step runs; the defaults are the measured
sweet spot (adaptive `target_error` 0.05, first-order forecast, ~50% of
forwards skipped at ~1.7x model speedup).

- **Adaptive mode (default, `target_error = 0.05`)**: after every skip run
  the replay error is measured against the next real forward and the
  effective threshold adjusted (0.7x-1.3x per correction). 0.06-0.08 is
  faster with slightly lower fidelity. `cache_threshold` (0.30) is the
  controller's starting point.
- **Fixed mode** (`target_error = 0`): measured drift is ~0.13 per step at
  40 steps, so the threshold is roughly 0.13 x the skip run length: 0.3
  skips ~2 steps per refresh, 0.5 ~3-4, 0.8 ~6.
- Keep `cache_start_percent` >= 0.15: measured replay error peaks right at
  the schedule start.
- `forecast` `first` is the measured sweet spot: at equal skip rate it cuts
  the replay error roughly in half vs `off`. `second` showed no further gain
  in testing and costs extra time; it stays available for comparison.
- The node logs the skip rate at the end of each run; with `debug_log` (or
  adaptive mode) it also reports the measured replay error mean/max and the
  final effective threshold.
