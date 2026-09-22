"""Qwen Image 2.1 Speedup node.

Step-level residual caching (TeaCache-style) for the Qwen Image 2.1
transformer, plus an optional preset wrapper around ComfyUI's official
block-sparse attention. The cache rides on the DIFFUSION_MODEL wrapper so the
model's built-in prefix K/V cache (text + reference images, computed once per
sampling run) stays active on full steps.
"""

import logging

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention
from typing_extensions import override

_MIB = 1024 * 1024
_PATCH_KEY = "qwen_image_2_1_speedup"
_MODEL_NAME = "QwenImage21Transformer2DModel"


class _StreamState:
    def __init__(self):
        self.residual = None
        self.prev_indicator = None
        self.last_sigma = None
        self.accumulated = 0.0
        self.consecutive_skips = 0
        self.full_steps = 0
        self.cache_hits = 0

    def clear_tensors(self):
        self.residual = None
        self.prev_indicator = None


class _TurboCache:
    """Caches the whole-forward residual (out - x) per conditioning stream and
    replays it while the accumulated drift of the timestep embedding stays
    under the threshold."""

    def __init__(self, threshold, start_percent, end_percent, max_consecutive_skips, cache_device):
        self.threshold = threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_consecutive_skips = max_consecutive_skips
        self.cache_device = cache_device
        self.streams = {}

    def reset(self):
        self.streams = {}

    def finish(self):
        full = sum(s.full_steps for s in self.streams.values())
        hits = sum(s.cache_hits for s in self.streams.values())
        if full + hits > 0:
            logging.info(
                "QwenImage21Speedup: %d cached of %d model forwards (%.1f%% skipped)",
                hits, full + hits, hits / (full + hits) * 100)
        for s in self.streams.values():
            s.clear_tensors()
        self.streams = {}

    @staticmethod
    def _stream_key(x, context, ref_latents, transformer_options):
        uuids = transformer_options.get("uuids")
        stream = tuple(str(u) for u in uuids) if uuids else ("default",)
        refs = tuple(tuple(r.shape) for r in (ref_latents or []))
        return (stream, tuple(x.shape), str(x.dtype), tuple(context.shape), refs)

    @staticmethod
    def _step_info(transformer_options):
        sigmas = transformer_options.get("sigmas")
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sigmas is None or sample_sigmas is None:
            return None
        sigma = float(sigmas[0])
        schedule = [float(s) for s in sample_sigmas]
        step = min(range(len(schedule)), key=lambda i: abs(schedule[i] - sigma))
        percent = min(1.0, step / max(1, len(schedule) - 2))
        return sigma, percent

    @staticmethod
    def _indicator(model, timestep, dtype):
        # same rounding as the model: target rows of the timestep embedding
        t = ((timestep * 1000).to(dtype) / 1000).to(dtype)
        temb = model.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        return temb[:-1].detach().float().flatten()

    def _store_residual(self, state, residual):
        # the residual outlives the forward, keep it out of the malloc graph
        with comfy.model_prefetch.pause_malloc_graph():
            residual = residual.detach()
            location = self.cache_device
            if residual.device.type == "cpu":
                location = "cpu"
            elif location == "auto":
                free = comfy.model_management.get_free_memory(residual.device)
                location = "gpu" if free > 10 * residual.numel() * residual.element_size() + 256 * _MIB else "cpu"
            if location == "gpu":
                state.residual = residual.clone()
            else:
                pinned = torch.empty(residual.shape, dtype=residual.dtype, device="cpu",
                                     pin_memory=torch.cuda.is_available())
                pinned.copy_(residual, non_blocking=False)
                state.residual = pinned

    def __call__(self, executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        model = executor.class_obj
        state = self.streams.setdefault(
            self._stream_key(x, context, ref_latents, transformer_options), _StreamState())
        step_info = self._step_info(transformer_options)

        with comfy.model_prefetch.pause_malloc_graph():
            indicator = self._indicator(model, timestep, x.dtype)

        eligible = False
        if step_info is not None and state.residual is not None and state.prev_indicator is not None:
            sigma, percent = step_info
            if state.last_sigma is None or sigma <= state.last_sigma + 1e-6:
                diff = float((indicator - state.prev_indicator).abs().mean()
                             / state.prev_indicator.abs().mean().clamp_min(1e-6))
                state.accumulated += diff
                eligible = (self.start_percent <= percent <= self.end_percent
                            and state.accumulated < self.threshold
                            and state.consecutive_skips < self.max_consecutive_skips)

        if eligible:
            out = x + state.residual.to(device=x.device, dtype=x.dtype)
            state.cache_hits += 1
            state.consecutive_skips += 1
        else:
            out = executor(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)
            self._store_residual(state, out - x)
            state.full_steps += 1
            state.consecutive_skips = 0
            state.accumulated = 0.0

        with comfy.model_prefetch.pause_malloc_graph():
            state.prev_indicator = indicator
        if step_info is not None:
            state.last_sigma = step_info[0]
        return out


class _SamplingScope:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, executor, *args, **kwargs):
        self.cache.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            self.cache.finish()


class QwenImage21Speedup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21Speedup",
            display_name="Qwen Image 2.1 Speedup",
            category="model/patch",
            is_experimental=True,
            description="Sampling accelerator for Qwen Image 2.1: caches the whole-model residual and replays it on "
                        "low-drift steps (TeaCache-style), optionally combined with official block-sparse attention. "
                        "Compatible with the model's built-in prefix K/V cache.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enable_cache", default=True,
                                 tooltip="Skip whole model forwards by replaying the cached residual while the "
                                         "accumulated timestep-embedding drift stays under the threshold."),
                io.Float.Input("cache_threshold", default=0.10, min=0.0, max=1.0, step=0.01,
                               tooltip="Accumulated relative drift allowed before forcing a full forward. "
                                       "Higher skips more steps and drifts further from the uncached result."),
                io.Float.Input("cache_start_percent", default=0.10, min=0.0, max=1.0, step=0.01,
                               tooltip="Caching only kicks in after this point in the sampling schedule."),
                io.Float.Input("cache_end_percent", default=0.90, min=0.0, max=1.0, step=0.01,
                               tooltip="Caching stops after this point; the tail always runs full forwards."),
                io.Int.Input("max_consecutive_skips", default=3, min=1, max=10, step=1,
                             tooltip="Upper bound on cached forwards in a row before a full refresh."),
                io.Combo.Input("cache_device", options=["auto", "gpu", "cpu"], default="auto",
                               tooltip="Where the cached residual lives. auto uses spare VRAM, else pinned RAM."),
                io.Boolean.Input("enable_sparse_attention", default=False,
                                 tooltip="Apply ComfyUI's official Sol-Attn block-sparse attention to the image "
                                         "segments. Helps most at high resolution; text segments stay dense."),
                io.Float.Input("sparse_tau", default=1.3, min=0.0, max=4.0, step=0.05,
                               tooltip="Sparsity threshold in score-distribution sigmas. Higher is sparser: "
                                       "1.0 keeps ~16% of key blocks, 1.5 ~7%, 2.0 ~2.7%."),
                io.Int.Input("sparse_min_tokens", default=2048, min=0, max=1 << 20, step=512,
                             tooltip="Attention calls with fewer query tokens stay dense. 2048 engages sparse "
                                     "attention from roughly 1MP images up."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enable_cache, cache_threshold, cache_start_percent, cache_end_percent,
                max_consecutive_skips, cache_device, enable_sparse_attention, sparse_tau, sparse_min_tokens):
        diffusion_model = model.get_model_object("diffusion_model")
        if type(diffusion_model).__name__ != _MODEL_NAME:
            raise ValueError(f"QwenImage21Speedup only supports Qwen Image 2.1 ({_MODEL_NAME}), "
                             f"got {type(diffusion_model).__name__}")

        m = model
        if enable_sparse_attention:
            m = apply_block_sparse_attention(
                m, tau=sparse_tau, topk_ratio=0.0, vsa=False,
                start_percent=0.2, end_percent=1.0, min_tokens=sparse_min_tokens,
                dense_blocks=set(), sink_conditioning="off", extra_tokens=256, verbose=False)

        if enable_cache:
            m = m.clone()
            cache = _TurboCache(cache_threshold, cache_start_percent, cache_end_percent,
                                max_consecutive_skips, cache_device)
            m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY)
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY, cache)
            m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY)
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY, _SamplingScope(cache))

        if m is model:
            logging.info("QwenImage21Speedup: every option is off, model passed through unchanged")
        return io.NodeOutput(m)


class QwenImage21SpeedupExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [QwenImage21Speedup]


async def comfy_entrypoint():
    return QwenImage21SpeedupExtension()


NODE_CLASS_MAPPINGS = {"QwenImage21Speedup": QwenImage21Speedup}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage21Speedup": "Qwen Image 2.1 Speedup"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
