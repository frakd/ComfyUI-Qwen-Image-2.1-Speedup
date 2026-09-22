# Qwen Image 2.1 Speedup

Qwen Image 2.1 的 ComfyUI 采样加速节点：TeaCache 式跳步门控 + TaylorCache 式 sigma 空间残差外推 + 误差闭环自适应控制。与模型内置的前缀 KV Cache（文本与参考图 K/V 每次采样只算一次）完全兼容，可叠加生效。

实测 40 步迭代下约 50% 的前向被缓存跳过，模型部分提速约 1.7×。

---

## 推荐用法

- **分辨率：≥ 220 万像素**（如 2048×1088、1920×1152），配合 **40 步**迭代——节点参数即按此场景标定。
- 链路：`UNETLoader → Qwen Image 2.1 Speedup → KSampler`，夹在模型加载与采样器之间即可，默认参数无需调整。
- 显存紧张时可在其后追加官方 `Qwen Image 2.1 Cache` 节点（int8 前缀 K/V）。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> qwen_image_2_1_speedup
```

重启 ComfyUI 后在 `model/patch` 分类下找到 **Qwen Image 2.1 Speedup**。

## 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `enable_cache` | true | 总开关；关闭时模型原样透传 |
| `cache_threshold` | 0.30 | 累积漂移阈值；自适应模式下作为控制器起点。固定模式下约等于 0.13 × 连跳步数（0.3≈跳2步、0.5≈跳3~4步） |
| `target_error` | 0.05 | 自适应模式：每次跳跃段结束后实测回放误差，按 0.7×~1.3× 乘性反馈调节有效阈值，把误差维持在目标附近；0.06~0.08 更快、保真略降；0 关闭自适应 |
| `cache_start_percent` / `cache_end_percent` | 0.15 / 0.90 | 只在调度中段启用缓存，首尾保持全量（实测起始段误差最大） |
| `max_consecutive_skips` | 3 | 连续跳步上限 |
| `forecast` | first | 残差外推阶数：`first` 一阶（最近 2 个实测残差，实测最优）；`second` 二阶（3 个残差，二次项有界，实测无额外收益，仅供对比）；`off` 原样重放 |
| `cache_device` | auto | 残差存储位置：auto 优先显存、不足时 pinned 内存 |
| `debug_log` | false | 逐步输出 sigma/漂移/决策原因 |

每次采样结束会输出跳过率；自适应模式（或开启 debug）还会输出回放误差 mean/max 与最终有效阈值。

## 原理

1. **跳步门控**：timestep 嵌入的相对漂移累积低于阈值时，整个 DiT 前向被跳过——门控挂在 `DIFFUSION_MODEL` wrapper 上，不打任何 block patch，因此官方前缀 KV Cache 不受影响。
2. **残差外推**：跳步时输出的不是上一步残差的原样重放，而是按 sigma 间隔从最近实测残差外推的预测值，外推幅度 clamp 在 1.5 个实测区间内。实测相同跳过率下误差约为直接重放的一半。
3. **误差闭环**：全量步实测上一段回放残差与真实前向的相对误差，驱动阈值上下调节；连续 3 步无法跳跃时阈值 ×1.5 向上重探，避免控制器锁死。

CFG 正负流独立缓存；模型内置的混合粒度注意力（文本因果掩码 + 图像块级掩码）与前缀 KV 复用为 ComfyUI 官方实现，本节点只在其上做采样层加速。

---
---

# Qwen Image 2.1 Speedup (English)

A ComfyUI sampling accelerator for Qwen Image 2.1: TeaCache-style skip gating + TaylorCache-style sigma-space residual extrapolation + a closed-loop error controller. Fully compatible with the model's built-in prefix K/V cache (text and reference-image K/V computed once per sampling run).

Measured at 40 steps: ~50% of transformer forwards are skipped, ~1.7x speedup on the model side.

## Recommended usage

- **Resolution ≥ 2.2 megapixels** (e.g. 2048×1088, 1920×1152) with **40 sampling steps** — the node is calibrated for this regime.
- Chain: `UNETLoader → Qwen Image 2.1 Speedup → KSampler`. Defaults are the measured sweet spot.
- When VRAM is tight, append the official `Qwen Image 2.1 Cache` node (int8 prefix K/V).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> qwen_image_2_1_speedup
```

Restart ComfyUI; the node appears as **Qwen Image 2.1 Speedup** under `model/patch`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `enable_cache` | true | Master switch; the model passes through unchanged when off |
| `cache_threshold` | 0.30 | Accumulated drift threshold; the controller's starting point in adaptive mode. In fixed mode it is roughly 0.13 x the skip run length |
| `target_error` | 0.05 | Adaptive mode: the actual replay error is measured after every skip run and the effective threshold adjusted (0.7x-1.3x) to hold the error near this target. 0.06-0.08 is faster with slightly lower fidelity; 0 disables adaptation |
| `cache_start_percent` / `cache_end_percent` | 0.15 / 0.90 | Caching only applies mid-schedule; the head and tail always run full forwards |
| `max_consecutive_skips` | 3 | Upper bound on cached forwards in a row |
| `forecast` | first | Residual extrapolation order: `first` (last two measured residuals; measured best), `second` (quadratic term from three, bounded; no measured gain, kept for comparison), `off` (verbatim replay) |
| `cache_device` | auto | Where cached residuals live: spare VRAM first, then pinned RAM |
| `debug_log` | false | Per-step sigma/drift/decision logging |

The node logs the skip rate after every run; adaptive mode (or `debug_log`) also reports the measured replay error mean/max and the final effective threshold.

## How it works

1. **Skip gate**: when the accumulated relative drift of the timestep embedding stays under the threshold, the whole transformer forward is skipped. The gate rides on the `DIFFUSION_MODEL` wrapper and installs no block patches, so the official prefix K/V cache keeps working on full steps.
2. **Residual forecast**: cached steps apply a residual extrapolated in sigma space from recently measured residuals (clamped to 1.5 measured intervals) instead of a verbatim replay — about half the error at the same skip rate.
3. **Error feedback**: full steps measure the actual error of the previous skip run and steer the effective threshold; three starved steps raise it 1.5x so the controller can never deadlock.

CFG positive/negative streams are cached independently. The mixed-granularity attention (causal text mask, chunk-level image mask) and prefix K/V reuse are ComfyUI's own implementations; this node only accelerates the sampling loop on top of them.
