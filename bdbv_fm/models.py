"""Thin wrappers around each time-series foundation model.

Every wrapper has the same interface:

    model = Wrapper()
    levels, preds = model.predict(series, horizon)

`series` maps a key (location|nowcast path) -> pd.Series of weekly values (DatetimeIndex, Saturdays,
no gaps, no NaN). `levels` is the array of quantile levels the model natively
produces and `preds` maps location -> array of shape (horizon, len(levels)).

Imports of the heavy model packages are deferred to __init__ so a missing or
broken package only takes that one model out of the ensemble.
"""
from __future__ import annotations

import os
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as C

FREQ = "W-SAT"
# 0.01, 0.05, 0.1, ..., 0.9, 0.95, 0.99
LEVELS_21 = np.array([0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99])
LEVELS_9 = np.round(np.arange(0.1, 0.91, 0.1), 2)


def device() -> str:
    if C.DEVICE != "auto":
        return C.DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _long_frame(series: dict[str, pd.Series], id_col: str, time_col: str, target_col: str) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame({id_col: loc, time_col: s.index, target_col: s.to_numpy()}) for loc, s in series.items()],
        ignore_index=True,
    )


def _match_columns(df: pd.DataFrame, levels) -> list[str]:
    """Find the output column for each quantile level, tolerating '0.1' vs '0.10' style names."""
    by_value = {}
    for c in df.columns:
        try:
            by_value[round(float(c), 6)] = c
        except (TypeError, ValueError):
            pass
    return [by_value[round(float(q), 6)] for q in levels]


class Baseline:
    """Reference model, not part of the ensemble: a random walk on log(x + 1).

    The median is the last value; the spread grows with sqrt(horizon) from the s.d. of the
    last 8 weekly log changes (at least 0.25). Needs no model weights.
    """

    levels = LEVELS_21

    def predict(self, series, horizon):
        z = np.array([NormalDist().inv_cdf(float(q)) for q in self.levels])
        steps = np.sqrt(np.arange(1, horizon + 1))[:, None]
        out = {}
        for key, s in series.items():
            y = np.log1p(np.maximum(s.to_numpy(float), 0))
            sd = max(float(np.std(np.diff(y[-9:]))) if len(y) > 2 else 0.0, 0.25)
            out[key] = np.expm1(y[-1] + sd * steps * z[None, :])
        return self.levels, out


class Chronos2:
    """Amazon Chronos-2 (https://huggingface.co/amazon/chronos-2)."""

    levels = LEVELS_21

    def __init__(self, model_id: str = "amazon/chronos-2"):
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device())

    def predict(self, series, horizon):
        ctx = _long_frame(series, "id", "timestamp", "target")
        pred = self.pipe.predict_df(
            ctx,
            prediction_length=horizon,
            quantile_levels=[float(q) for q in self.levels],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        ).sort_values(["id", "timestamp"])
        cols = _match_columns(pred, self.levels)
        return self.levels, {loc: g[cols].to_numpy(float) for loc, g in pred.groupby("id")}


class TimesFM:
    """Google TimesFM 2.5 (Apache-2.0 weights). TimesFM 3.0 weights are non-commercial/non-production."""

    levels = LEVELS_9
    patch = 32
    max_context = 1024

    def __init__(self, model_id: str = "google/timesfm-2.5-200m-pytorch"):
        import timesfm

        self.timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self.context = None

    def _compile(self, context: int):
        if context == self.context:
            return
        self.model.compile(
            self.timesfm.ForecastConfig(
                max_context=context,
                max_horizon=128,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self.context = context

    def predict(self, series, horizon):
        # A context window with a whole patch of padding gives all-NaN output on torch < 2.5
        # (which Moirai's uni2ts pins), so size the window to each series, rounded up to a patch.
        groups: dict[int, list[str]] = {}
        for loc, s in series.items():
            n = min(len(s), self.max_context)
            groups.setdefault(-(-n // self.patch) * self.patch, []).append(loc)
        out = {}
        for context, locs in sorted(groups.items()):
            self._compile(context)
            inputs = [series[loc].to_numpy(np.float32)[-context:] for loc in locs]
            _, quant = self.model.forecast(horizon=horizon, inputs=inputs)
            quant = _to_numpy(quant)  # (n, horizon, 10): mean, then 0.1 ... 0.9
            out.update({loc: quant[i, :horizon, 1:] for i, loc in enumerate(locs)})
        return self.levels, out


class Moirai2:
    """Salesforce Moirai 2.0 (https://huggingface.co/Salesforce/moirai-2.0-R-small)."""

    levels = LEVELS_9

    def __init__(self, model_id: str = "Salesforce/moirai-2.0-R-small"):
        from uni2ts.model.moirai2 import Moirai2Module

        self.module = Moirai2Module.from_pretrained(model_id)

    def predict(self, series, horizon):
        from gluonts.dataset.pandas import PandasDataset
        from uni2ts.model.moirai2 import Moirai2Forecast

        context_length = max(len(s) for s in series.values())
        model = Moirai2Forecast(
            module=self.module,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=32)
        ds = PandasDataset({loc: s.to_frame("target") for loc, s in series.items()}, target="target", freq=FREQ)
        preds = {}
        for f in predictor.predict(ds):
            preds[f.item_id] = np.stack([np.asarray(f.quantile(q), float) for q in self.levels], axis=-1)
        return self.levels, preds


class Timestral:
    """Mistral Timestral (https://huggingface.co/mistralai/Timestral-209M)."""

    def __init__(self, model_id: str | None = None):
        from timestral import TimestralForecaster

        model_id = model_id or os.environ.get("TIMESTRAL_MODEL", "mistralai/Timestral-209M")
        self.model = TimestralForecaster.from_pretrained(model_id, device=device())

    def predict(self, series, horizon):
        locs = list(series)
        # Batch as (B, T); shorter series are left-padded with NaN, which Timestral treats as missing.
        T = max(len(s) for s in series.values())
        ctx = np.full((len(locs), T), np.nan, dtype=np.float32)
        for i, loc in enumerate(locs):
            v = series[loc].to_numpy(np.float32)
            ctx[i, T - len(v):] = v
        fc = self.model.forecast(ctx, prediction_length=horizon)
        q = _to_numpy(fc.quantiles)
        if q.ndim == 4:  # (B, variates, horizon, Q)
            q = q[:, 0]
        levels = _to_numpy(fc.quantile_levels)
        return levels, {loc: q[i, :horizon] for i, loc in enumerate(locs)}


MODELS = {
    "baseline": Baseline,
    "chronos2": Chronos2,
    "timesfm": TimesFM,
    "moirai2": Moirai2,
    "timestral": Timestral,
}


class ModelPool:
    """Loads each model once and keeps it, so backtests don't reload weights every week."""

    def __init__(self):
        self._models, self._errors = {}, {}

    def get(self, name: str):
        if name in self._errors:
            raise RuntimeError(f"{name} failed to load earlier") from self._errors[name]
        if name not in self._models:
            try:
                self._models[name] = MODELS[name]()
            except Exception as e:
                self._errors[name] = e
                raise
        return self._models[name]
