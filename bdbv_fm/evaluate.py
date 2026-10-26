"""Scoring past submissions against the latest data.

Every model's 100 sample trajectories are saved next to each submission
(components/<ref>.parquet, with the ensemble under the model id), so all are scored
the same way:

- CRPS from samples, on counts and on log(x + 1)
- 50% and 90% central interval coverage
- CRPS relative to the random-walk baseline, on the forecasts both made

Weeks are scored only once they are REVISION_SETTLE_WEEKS old in the latest data;
before that the "truth" is itself still incomplete. Horizons are counted from the
reference date (negative = nowcast of a week already in the data).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data

log = logging.getLogger(__name__)
KEYS = ["model", "reference_date", "location", "target_end_date"]


def truth_table(target: pd.DataFrame | None = None) -> pd.DataFrame:
    target = data.load_target() if target is None else target
    _, final = data.latest(target)
    final = final.drop_duplicates(["location", "date"], keep="last")
    settled = final["date"].max() - pd.Timedelta(weeks=C.REVISION_SETTLE_WEEKS)
    return final.rename(columns={"date": "target_end_date", "value": "truth"}).assign(
        settled=lambda d: d["target_end_date"] <= settled)


def load_samples(root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((Path(root) / "components").glob("*.parquet")):
        df = pd.read_parquet(f)
        df["reference_date"] = pd.Timestamp(f.stem)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    df["model"] = df["model"].replace({"ensemble": C.model_id()})
    return df


def crps_samples(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of each row of samples X (groups, n) for outcomes y (groups,)."""
    n = X.shape[1]
    Xs = np.sort(X, axis=1)
    spread = 2 * ((2 * np.arange(1, n + 1) - n - 1) * Xs).sum(axis=1) / n**2
    return np.abs(X - y[:, None]).mean(axis=1) - 0.5 * spread


def score(samples: pd.DataFrame, truth: pd.DataFrame, locations: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (model, reference_date, location, target_end_date)."""
    wide = samples.pivot_table(index=KEYS, columns="sample", values="value")
    df = wide.reset_index()[KEYS].merge(truth, on=["location", "target_end_date"], how="left")
    ok = df["truth"].notna().to_numpy()
    X, y = wide.to_numpy(float)[ok], df.loc[ok, "truth"].to_numpy(float)
    df = df[ok].reset_index(drop=True)
    df["crps"] = crps_samples(X, y)
    df["crps_log"] = crps_samples(np.log1p(np.maximum(X, 0)), np.log1p(y))
    for p in (50, 90):
        lo, hi = np.quantile(X, [0.5 - p / 200, 0.5 + p / 200], axis=1)
        df[f"cov{p}"] = ((lo <= y) & (y <= hi)).astype(float)
    df["median"] = np.median(X, axis=1)
    df["horizon"] = (df["target_end_date"] - df["reference_date"]).dt.days // 7
    if locations is not None:
        df["level"] = df["location"].map(dict(zip(locations["location"], locations["location_type"])))
    return df


def relative_summary(scores: pd.DataFrame, by=("horizon",)) -> pd.DataFrame:
    """Mean scores per model, with CRPS relative to the baseline on the forecasts both made."""
    by = list(by)
    rows = []
    base = scores[scores["model"] == C.BASELINE_MODEL].set_index(KEYS[1:])[["crps", "crps_log"]]
    for model, g in scores.groupby("model"):
        j = g.join(base, on=KEYS[1:], rsuffix="_base")
        for key, gg in j.groupby(by):
            both = gg.dropna(subset=["crps_base"])
            rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,))) | {
                "model": model, "n": len(gg), "crps": gg["crps"].mean(), "crps_log": gg["crps_log"].mean(),
                "rel_crps": _ratio(both["crps"].sum(), both["crps_base"].sum()),
                "rel_crps_log": _ratio(both["crps_log"].sum(), both["crps_log_base"].sum()),
                "cov50": gg["cov50"].mean(), "cov90": gg["cov90"].mean(),
            })
    return pd.DataFrame(rows)


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else np.nan


def evaluate(root: Path) -> dict[str, pd.DataFrame]:
    """Score everything under `root` (forecasts/ or a backtest folder); writes CSVs to root/evaluation/."""
    root = Path(root)
    samples = load_samples(root)
    if samples.empty:
        log.info("No saved samples under %s to evaluate", root)
        return {}
    locations = data.load_locations(refresh=False) if (C.CACHE_DIR / "hub" / C.LOCATIONS_PATH).exists() \
        else data.load_locations()
    scores = score(samples, truth_table(), locations)
    scores = scores[scores["settled"]]
    if scores.empty:
        log.info("No forecast weeks under %s are settled in the data yet", root)
        return {}
    out = {
        "scores": scores,
        "by_horizon": relative_summary(scores),
        "by_level": relative_summary(scores, by=("level", "horizon")),
        "by_week": relative_summary(scores, by=("reference_date",)),
    }
    d = root / "evaluation"
    d.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(d / f"{name}.csv", index=False)
    log.info("Wrote evaluation tables to %s", d)
    return out
