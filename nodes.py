"""Qwen Image 2.1 Speedup node.

Step-level residual caching (TeaCache-style) for the Qwen Image 2.1
transformer with sigma-space residual extrapolation (TaylorCache-style) and
an optional error-feedback controller. The cache rides on the
DIFFUSION_MODEL wrapper so the model's built-in prefix K/V cache (text +
reference images, computed once per sampling run) stays active on full steps.
"""

import logging

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

_MIB = 1024 * 1024
_PATCH_KEY = "qwen_image_2_1_speedup"
_MODEL_NAME = "QwenImage21Transformer2DModel"


class _StreamState:
    def __init__(self):
        self.residuals = []  # newest-first [(sigma, tensor)], up to 3 for second-order forecast
        self.prev_indicator = None
        self.last_sigma = None
        self.last_used = None
        self.threshold_eff = None
        self.starve_count = 0
        self.accumulated = 0.0
        self.consecutive_skips = 0
        self.full_steps = 0
        self.cache_hits = 0
        self.drift_sum = 0.0
        self.drift_max = 0.0
        self.drift_count = 0
        self.pred_err_sum = 0.0
        self.pred_err_max = 0.0
        self.pred_err_count = 0

    def clear_tensors(self):
        self.residuals = []
        self.prev_indicator = None
        self.last_used = None


class _TurboCache:
    """Caches the whole-forward residual (out - x) per conditioning stream and
    replays it while the accumulated drift of the timestep embedding stays
    under the threshold. The replayed residual is a sigma-space extrapolation
    from the measured residual history: first-order secant, or second-order
    Newton form with the quadratic term clamped to the linear term's
    magnitude. With a target error set, measured replay errors steer the
    effective threshold between corrections."""

    _MAX_EXTRAPOLATE = 1.5
    _HISTORY = 3

    def __init__(self, threshold, start_percent, end_percent, max_consecutive_skips, cache_device,
                 forecast="first", target_error=0.0, debug=False):
        self.threshold = threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_consecutive_skips = max_consecutive_skips
        self.cache_device = cache_device
        self.forecast = forecast
        self.target_error = target_error
        self.debug = debug
        self.streams = {}

    def reset(self):
        self.streams = {}

    def finish(self):
        full = sum(s.full_steps for s in self.streams.values())
        hits = sum(s.cache_hits for s in self.streams.values())
        drift_count = sum(s.drift_count for s in self.streams.values())
        if full + hits > 0:
            logging.info(
                "QwenImage21Speedup: %d cached of %d model forwards (%.1f%% skipped)",
                hits, full + hits, hits / (full + hits) * 100)
        if drift_count > 0 and (self.debug or hits == 0):
            drift_sum = sum(s.drift_sum for s in self.streams.values())
            drift_max = max(s.drift_max for s in self.streams.values())
            logging.info(
                "QwenImage21Speedup: per-step indicator drift mean %.4f, max %.4f over %d steps "
                "(a useful cache_threshold must exceed the mean; current %.3f)",
                drift_sum / drift_count, drift_max, drift_count, self.threshold)
        pred_err_count = sum(s.pred_err_count for s in self.streams.values())
        if pred_err_count > 0 and (self.debug or self.target_error > 0):
            err_sum = sum(s.pred_err_sum for s in self.streams.values())
            err_max = max(s.pred_err_max for s in self.streams.values())
            eff = [s.threshold_eff for s in self.streams.values() if s.threshold_eff is not None]
            logging.info(
                "QwenImage21Speedup: replayed-residual error vs actual forward mean %.4f, max %.4f over %d checks%s",
                err_sum / pred_err_count, err_max, pred_err_count,
                f", final effective threshold {sum(eff) / len(eff):.3f}" if eff else "")
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

    def _store_residual(self, state, residual, sigma):
        # residuals outlive the forward, keep them out of the malloc graph
        with comfy.model_prefetch.pause_malloc_graph():
            residual = residual.detach()
            location = self.cache_device
            if residual.device.type == "cpu":
                location = "cpu"
            elif location == "auto":
                free = comfy.model_management.get_free_memory(residual.device)
                location = "gpu" if free > 10 * residual.numel() * residual.element_size() + 256 * _MIB else "cpu"
            if location == "gpu":
                stored = residual.clone()
            else:
                stored = torch.empty(residual.shape, dtype=residual.dtype, device="cpu",
                                     pin_memory=torch.cuda.is_available())
                stored.copy_(residual, non_blocking=False)
            state.residuals.insert(0, (sigma, stored))
            del state.residuals[self._HISTORY:]

    def _replay(self, state, x, sigma):
        """Residual for a cached step: plain replay, or sigma-space
        extrapolation from the measured residual history. The horizon is
        clamped to _MAX_EXTRAPOLATE measured intervals, and the second-order
        term elementwise to the linear term's magnitude."""
        hist = state.residuals
        t2 = hist[0][1].to(device=x.device, dtype=x.dtype)
        r1 = t2
        f = 0.0
        if self.forecast != "off" and sigma is not None and len(hist) > 1:
            s1, s0 = hist[0][0], hist[1][0]
            if s1 is not None and s0 is not None and abs(s1 - s0) > 1e-8:
                f = min(max((sigma - s1) / (s1 - s0), 0.0), self._MAX_EXTRAPOLATE)
        if f > 0.0:
            t1 = hist[1][1].to(device=x.device, dtype=x.dtype)
            r1 = torch.lerp(t1, t2, 1.0 + f)
            if self.forecast == "second" and len(hist) > 2 and hist[2][0] is not None and abs(hist[0][0] - hist[2][0]) > 1e-8:
                s2, s1, s0 = hist[0][0], hist[1][0], hist[2][0]
                t0 = hist[2][1].to(device=x.device, dtype=x.dtype)
                d1 = (t2 - t1) / (s2 - s1)
                d0 = (t1 - t0) / (s1 - s0)
                linear = d1 * (sigma - s2)
                quad = (d1 - d0) / (s2 - s0) * ((sigma - s2) * (sigma - s1))
                r1 = r1 + torch.clamp(quad, -linear.abs(), linear.abs())
        if self.debug or self.target_error > 0:
            with comfy.model_prefetch.pause_malloc_graph():
                state.last_used = r1.detach().clone()
        if self.debug:
            logging.info("QwenImage21Speedup: replayed residual, extrapolation factor %.2f", f)
        return x + r1

    def __call__(self, executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        model = executor.class_obj
        state = self.streams.setdefault(
            self._stream_key(x, context, ref_latents, transformer_options), _StreamState())
        step_info = self._step_info(transformer_options)
        sigma = step_info[0] if step_info is not None else None

        with comfy.model_prefetch.pause_malloc_graph():
            indicator = self._indicator(model, timestep, x.dtype)

        eligible = False
        reason = "no cached residual yet"
        if step_info is not None and state.residuals and state.prev_indicator is not None:
            sigma, percent = step_info
            if state.last_sigma is not None and sigma > state.last_sigma + 1e-6:
                reason = "sigma moved backwards (new run), forcing full"
            else:
                threshold = state.threshold_eff if state.threshold_eff is not None else self.threshold
                diff = float((indicator - state.prev_indicator).abs().mean()
                             / state.prev_indicator.abs().mean().clamp_min(1e-6))
                state.accumulated += diff
                state.drift_sum += diff
                state.drift_max = max(state.drift_max, diff)
                state.drift_count += 1
                if not (self.start_percent <= percent <= self.end_percent):
                    reason = f"outside window ({percent:.2f})"
                elif state.accumulated >= threshold:
                    reason = f"accumulated drift {state.accumulated:.4f} >= threshold {threshold:.3f}"
                    if self.target_error > 0:
                        # starvation recovery: without skips there are no error
                        # measurements, so a shrunk threshold could never climb back
                        state.starve_count += 1
                        if state.starve_count >= 3:
                            state.threshold_eff = min(threshold * 1.5, 2.0)
                            state.starve_count = 0
                            reason += f"; no skips for 3 steps, effective threshold raised to {state.threshold_eff:.3f}"
                elif state.consecutive_skips >= self.max_consecutive_skips:
                    reason = "max consecutive skips reached"
                else:
                    eligible = True
                    state.starve_count = 0
                if self.debug:
                    logging.info(
                        "QwenImage21Speedup: sigma %.4f, percent %.2f, drift %.4f, accumulated %.4f -> %s",
                        sigma, percent, diff, state.accumulated, "cached" if eligible else f"full ({reason})")

        if eligible:
            out = self._replay(state, x, sigma)
            state.cache_hits += 1
            state.consecutive_skips += 1
        else:
            out = executor(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)
            residual = out - x
            if state.last_used is not None:
                # measured error of the last replay, the feedback signal for adaptive mode
                err = float((residual - state.last_used).abs().mean() / out.abs().mean().clamp_min(1e-6))
                state.pred_err_sum += err
                state.pred_err_max = max(state.pred_err_max, err)
                state.pred_err_count += 1
                state.last_used = None
                if self.target_error > 0:
                    base = state.threshold_eff if state.threshold_eff is not None else self.threshold
                    ratio = min(max(self.target_error / max(err, 1e-4), 0.7), 1.3)
                    state.threshold_eff = min(max(base * ratio, 0.05), 2.0)
                if self.debug:
                    logging.info(
                        "QwenImage21Speedup: actual forward vs last replayed residual, relative error %.4f%s",
                        err, f", effective threshold -> {state.threshold_eff:.3f}"
                        if state.threshold_eff is not None else "")
            self._store_residual(state, residual, sigma)
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
                        "low-drift steps (TeaCache-style) with sigma-space extrapolation (TaylorCache-style). "
                        "Compatible with the model's built-in prefix K/V cache.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enable_cache", default=True,
                                 tooltip="Skip whole model forwards by replaying the cached residual while the "
                                         "accumulated timestep-embedding drift stays under the threshold."),
                io.Float.Input("cache_threshold", default=0.40, min=0.0, max=2.0, step=0.01,
                               tooltip="Accumulated relative drift allowed before forcing a full forward. "
                                       "Measured drift on this model is ~0.13 per step at 40 steps, so the "
                                       "threshold is roughly 0.13 x the skip run length: 0.3 skips ~2 steps, "
                                       "0.5 ~3-4, 0.8 ~6."),
                io.Float.Input("target_error", default=0.0, min=0.0, max=1.0, step=0.005,
                               tooltip="Adaptive mode: if > 0 (e.g. 0.05), the effective threshold is adjusted "
                                       "after every measured replay error to hold the error near this target "
                                       "(multiplicative feedback, 0.7x-1.3x per correction). Adapts to step "
                                       "count, resolution and prompt automatically. 0 uses the fixed threshold."),
                io.Float.Input("cache_start_percent", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Caching only kicks in after this point in the sampling schedule. "
                                       "Measured replay error is highest right after the start, keep >= 0.15."),
                io.Float.Input("cache_end_percent", default=0.90, min=0.0, max=1.0, step=0.01,
                               tooltip="Caching stops after this point; the tail always runs full forwards."),
                io.Int.Input("max_consecutive_skips", default=3, min=1, max=10, step=1,
                             tooltip="Upper bound on cached forwards in a row before a full refresh."),
                io.Combo.Input("forecast", options=["first", "second", "off"], default="first",
                               tooltip="Residual extrapolation order for cached steps. first: sigma-space linear "
                                       "extrapolation from the last two measured residuals. second: adds the "
                                       "quadratic term from the last three, clamped to the linear term's "
                                       "magnitude. off: replay the latest residual verbatim."),
                io.Combo.Input("cache_device", options=["auto", "gpu", "cpu"], default="auto",
                               tooltip="Where the cached residual lives. auto uses spare VRAM, else pinned RAM."),
                io.Boolean.Input("debug_log", default=False,
                                 tooltip="Log per-step sigma, drift and cache decisions to the console."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enable_cache, cache_threshold, target_error, cache_start_percent, cache_end_percent,
                max_consecutive_skips, forecast, cache_device, debug_log=False):
        diffusion_model = model.get_model_object("diffusion_model")
        if type(diffusion_model).__name__ != _MODEL_NAME:
            raise ValueError(f"QwenImage21Speedup only supports Qwen Image 2.1 ({_MODEL_NAME}), "
                             f"got {type(diffusion_model).__name__}")
        if not enable_cache:
            logging.info("QwenImage21Speedup: cache disabled, model passed through unchanged")
            return io.NodeOutput(model)

        m = model.clone()
        cache = _TurboCache(cache_threshold, cache_start_percent, cache_end_percent,
                            max_consecutive_skips, cache_device, forecast=forecast,
                            target_error=target_error, debug=debug_log)
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY, cache)
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY, _SamplingScope(cache))
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
