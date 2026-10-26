"""One week's submission: nowcast, component models, ensemble, and 100 sample trajectories per location.

How the pieces fit:

1. The latest data are turned into one weekly series per location (country, provinces,
   and health zones with recent cases).
2. The last NOWCAST_WEEKS weeks are nowcast (nowcast.py) and NOWCAST_PATHS joint draws
   of those weeks are taken. Each draw is a complete series: settled history plus one
   plausible version of the incomplete weeks.
3. Every model forecasts every draw, from the week after the data end through
   reference_date + MAX_HORIZON. Their quantiles are mapped onto a common grid and
   combined across models (median by default), separately for each draw.
4. Sample i uses draw i mod NOWCAST_PATHS: its nowcast weeks come from that draw, and its
   future weeks are sampled from that draw's ensemble quantiles, tied together over
   weeks by a Gaussian copula with AR(1) correlation TRAJ_RHO.

The submission covers the nowcast weeks and all later weeks through reference_date + 3.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as C
from . import nowcast as nc
from .models import FREQ, ModelPool

log = logging.getLogger(__name__)

_TRANSFORMS = {
    "none": (lambda x: x, lambda x: x),
    "sqrt": (np.sqrt, lambda x: np.square(np.maximum(x, 0))),
    "log1p": (np.log1p, np.expm1),
}
COLUMNS = ["reference_date", "target", "target_end_date", "location", "output_type", "output_type_id", "value"]


def _z(levels) -> np.ndarray:
    return np.array([NormalDist().inv_cdf(float(p)) for p in levels])


@dataclasses.dataclass
class Result:
    ref_date: dt.date
    as_of: pd.Timestamp              # release date of the data used
    submission: pd.DataFrame
    problems: list[str]
    warnings: list[str]
    last_obs: pd.Timestamp           # last week in the data
    weeks: pd.DatetimeIndex          # weeks in the submission
    reported: dict                   # {location: Series as reported}
    revised: dict                    # {location: Series with nowcast medians}
    nowcasts: pd.DataFrame           # table from nowcast.nowcast
    samples: dict                    # {model or "ensemble": {location: array (N_SAMPLES, len(weeks))}}
    models: list[str]                # component models that ran (excluding the baseline)


# ------------------------------------------------------------------------- data
def build_series(frame: pd.DataFrame, locations: pd.DataFrame, active_only: bool = True):
    """One gap-free weekly series per location from one data version.

    Returns (series, last week, locations left out because they had no recent cases).
    Weeks missing from the data count as 0 cases.
    """
    levels = locations.loc[locations["location_type"].isin(C.LEVELS), "location"]
    df = frame.dropna(subset=["value"])
    df = df[df["location"].isin(levels)]
    if (df["date"].dt.dayofweek != 5).any():
        raise ValueError("target data contains dates that are not Saturdays")
    last_obs = df["date"].max()
    start = max(pd.Timestamp(C.CONTEXT_START or C.OUTBREAK_START), df["date"].min())
    index = pd.date_range(start, last_obs, freq=FREQ)
    series, inactive = {}, []
    for loc, g in df.groupby("location"):
        s = g.drop_duplicates("date", keep="last").set_index("date")["value"].astype(float)
        s = s.reindex(index).fillna(0).clip(lower=0)
        if active_only and s.iloc[-C.ACTIVE_WEEKS:].sum() < C.MIN_CASES:
            inactive.append(loc)
            continue
        series[loc] = s
    return series, last_obs, inactive


def weeks_between(start, end) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days // 7


# ---------------------------------------------------------------------- models
def to_levels(levels, q: np.ndarray) -> np.ndarray:
    """Map quantiles at a model's native levels onto the QUANTILES grid.

    Interpolates linearly in normal-score space and extrapolates the tails with
    the slope of the outermost pair, so models that only give 0.1-0.9 still get
    0.01/0.025/0.975/0.99.
    """
    levels = np.asarray(levels, float)
    order = np.argsort(levels)
    z, zt = _z(levels[order]), _z(C.QUANTILES)
    q = np.sort(np.asarray(q, float)[..., order], axis=-1)
    out = np.empty(q.shape[:-1] + (len(zt),))
    lo, hi = zt < z[0], zt > z[-1]
    for idx in np.ndindex(q.shape[:-1]):
        v = q[idx]
        r = np.interp(zt, z, v)
        r[lo] = v[0] + (zt[lo] - z[0]) * (v[1] - v[0]) / (z[1] - z[0])
        r[hi] = v[-1] + (zt[hi] - z[-1]) * (v[-1] - v[-2]) / (z[-1] - z[-2])
        out[idx] = r
    return out


def _predict(model, series, n_steps):
    fwd, inv = _TRANSFORMS[C.TRANSFORM]
    tseries = {k: pd.Series(fwd(s.to_numpy()), index=s.index) for k, s in series.items()}
    levels, preds = model.predict(tseries, n_steps)
    missing = set(series) - set(preds)
    if missing:
        raise RuntimeError(f"no forecast for {len(missing)} series, e.g. {sorted(missing)[:3]}")
    out = {}
    for k in series:
        q = np.asarray(preds[k], float)
        if q.shape != (n_steps, len(levels)):
            raise RuntimeError(f"bad forecast for {k}: shape {q.shape}, expected {(n_steps, len(levels))}")
        if not np.isfinite(q).all():
            raise RuntimeError(f"bad forecast for {k}: {np.size(q) - np.isfinite(q).sum()} non-finite values")
        out[k] = np.maximum(np.sort(inv(to_levels(levels, q)), axis=-1), 0)
    return out


def run_components(pool: ModelPool, names, series, n_steps, warnings: list[str]):
    """{model: {series key: array(n_steps, len(QUANTILES))}} for each model that ran."""
    results = {}
    for name in names:
        try:
            model = pool.get(name)
        except Exception as e:
            log.exception("%s failed to load", name)
            warnings.append(f"{name} failed to load ({type(e).__name__}: {e}); left out")
            continue
        log.info("Running %s on %d series (%d steps)", name, len(series), n_steps)
        try:
            results[name] = _predict(model, series, n_steps)
        except Exception as e:
            log.exception("%s failed", name)
            warnings.append(f"{name} failed ({type(e).__name__}: {e}); left out")
    return results


def ensemble(components: dict) -> dict:
    names = list(components)
    if C.ENSEMBLE_METHOD == "weighted":
        w = np.array([float(C.ENSEMBLE_WEIGHTS.get(n, 0.0)) for n in names])
        if w.sum() <= 0:
            raise ValueError("ENSEMBLE_WEIGHTS gives no weight to the models that ran")
        agg = lambda a: np.tensordot(w / w.sum(), a, axes=1)
    elif C.ENSEMBLE_METHOD == "mean":
        agg = lambda a: a.mean(axis=0)
    else:
        agg = lambda a: np.median(a, axis=0)
    keys = components[names[0]].keys()
    out = {}
    for k in keys:
        q = np.maximum(np.sort(agg(np.stack([components[n][k] for n in names])), axis=-1), 0)
        if C.SPREAD_SCALE != 1.0:
            lq = np.log(q + C.LOG_OFFSET)
            mid = lq[:, [C.MEDIAN_IDX]]
            q = np.maximum(np.exp(mid + C.SPREAD_SCALE * (lq - mid)) - C.LOG_OFFSET, 0)
        out[k] = q
    return out


# ------------------------------------------------------------------ trajectories
def trajectories(quantiles: dict, paths: dict, n_nowcast: int, n_paths: int, rng) -> dict:
    """{location: array (N_SAMPLES, n_nowcast + n_steps)} for one model (or the ensemble).

    quantiles: {"loc|p": array(n_steps, len(QUANTILES))}; paths: {loc: array(n_paths, T)}.
    """
    z_levels, n = _z(C.QUANTILES), C.N_SAMPLES
    out = {}
    for loc, pth in paths.items():
        q0 = quantiles.get(f"{loc}|0")
        if q0 is None:
            continue
        n_steps = q0.shape[0]
        z = np.empty((n, n_steps))
        z[:, 0] = rng.standard_normal(n)
        for t in range(1, n_steps):
            z[:, t] = C.TRAJ_RHO * z[:, t - 1] + np.sqrt(1 - C.TRAJ_RHO**2) * rng.standard_normal(n)
        traj = np.empty((n, n_nowcast + n_steps))
        for i in range(n):
            p = i % n_paths
            q = quantiles[f"{loc}|{p}"]
            traj[i, :n_nowcast] = pth[p, pth.shape[1] - n_nowcast:] if n_nowcast else []
            traj[i, n_nowcast:] = [np.interp(z[i, t], z_levels, q[t]) for t in range(n_steps)]
        out[loc] = np.maximum(traj, 0)
    return out


# ---------------------------------------------------------------------- output
def make_frame(samples: dict, weeks, ref_date) -> pd.DataFrame:
    rows = []
    for loc, traj in samples.items():
        vals = np.rint(traj) if C.ROUND_SAMPLES else np.round(traj, 3)
        for i in range(traj.shape[0]):
            rows += [(ref_date, C.TARGET, w.date(), loc, "sample", i + 1, float(v)) for w, v in zip(weeks, vals[i])]
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["location", "output_type_id", "target_end_date"], kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame, ref_date: dt.date, valid_locations) -> list[str]:
    """Checks mirroring the hub's: the hub's own validation runs on the pull request."""
    problems = []
    if df.empty:
        return ["submission is empty"]
    ted = pd.to_datetime(df["target_end_date"])
    if (ted.dt.dayofweek != 5).any():
        problems.append("target_end_date values that are not Saturdays")
    if ted.max() > pd.Timestamp(ref_date) + pd.Timedelta(weeks=C.MAX_HORIZON) or ted.min() < pd.Timestamp(C.OUTBREAK_START):
        problems.append(f"target_end_date outside {C.OUTBREAK_START} to reference_date + {C.MAX_HORIZON} weeks")
    bad = set(df["location"]) - set(valid_locations)
    if bad:
        problems.append(f"unknown locations: {sorted(bad)[:5]}")
    if not np.isfinite(df["value"]).all() or (df["value"] < 0).any():
        problems.append("negative or non-finite values")
    for loc, g in df.groupby("location"):
        ids = g["output_type_id"].unique()
        if len(ids) != C.N_SAMPLES:
            problems.append(f"{loc}: {len(ids)} samples, expected {C.N_SAMPLES}")
        counts = g.groupby("output_type_id")["target_end_date"].nunique()
        if counts.nunique() != 1 or len(g) != len(ids) * counts.iloc[0]:
            problems.append(f"{loc}: samples do not all cover the same weeks exactly once")
    return problems


# ------------------------------------------------------------------------ main
def run(ref_date: dt.date, frame: pd.DataFrame, as_of, locations: pd.DataFrame, models,
        vintages: dict | None = None, prior_dir: Path | None = None, pool: ModelPool | None = None,
        active_only: bool = True) -> Result:
    """Build the submission for one reference date from one data version (`frame`, released `as_of`).

    vintages: {as_of: frame} past data versions, for the nowcast's revision model.
    prior_dir: folder holding earlier submissions (model-output/<model_id>/), for the nowcast prior.
    """
    pool = pool or ModelPool()
    warnings: list[str] = []
    rng = np.random.default_rng(C.SEED)
    offset = C.LOG_OFFSET
    parent = dict(zip(locations["location"], locations["parent"]))

    series, last_obs, inactive = build_series(frame, locations, active_only)
    log.info("Data released %s through week %s; %d locations with recent cases (%d without, left out)",
             pd.Timestamp(as_of).date(), last_obs.date(), len(series), len(inactive))
    last_target = pd.Timestamp(ref_date) + pd.Timedelta(weeks=C.MAX_HORIZON)
    n_steps = weeks_between(last_obs, last_target)
    if n_steps < 1:
        raise ValueError(f"data run through {last_obs.date()}, past the last target week {last_target.date()}")
    if n_steps > C.MAX_HORIZON + 3:
        warnings.append(f"data end {last_obs.date()}, {n_steps} weeks before the last target week; "
                        "this week's data release may not be posted yet")

    if C.NOWCAST:
        rev = nc.fit_revisions(vintages or {}, as_of, offset, C.NOWCAST_MODEL)
        if 0 not in rev:
            warnings.append("no revision model for the latest week; it is submitted as reported, which is "
                            "far too low (check the delay data and NOWCAST_SOURCE)")
        dates = [last_obs - pd.Timedelta(weeks=k) for k in range(C.NOWCAST_WEEKS)]
        prior = nc.prior_forecasts(prior_dir, ref_date, dates, offset)
        revised, table = nc.nowcast(series, rev, prior, offset, parent)
        n_paths = C.NOWCAST_PATHS
        paths = nc.sample_paths(series, table, n_paths, offset, rng)
        n_nowcast = min(C.NOWCAST_WEEKS, min(len(s) for s in series.values()))
    else:
        revised, table = series, pd.DataFrame(columns=["location", "date", "lag", "reported", "median"])
        n_paths, n_nowcast = 1, 0
        paths = {loc: s.to_numpy(float)[None, :] for loc, s in series.items()}

    index = next(iter(series.values())).index
    inputs = {f"{loc}|{p}": pd.Series(pth[p], index=index) for loc, pth in paths.items() for p in range(n_paths)}
    names = [m for m in models if m != C.BASELINE_MODEL]
    components = run_components(pool, names + [C.BASELINE_MODEL], inputs, n_steps, warnings)

    weeks = pd.date_range(last_obs - pd.Timedelta(weeks=n_nowcast - 1), last_target, freq=FREQ) if n_nowcast else \
        pd.date_range(last_obs + pd.Timedelta(weeks=1), last_target, freq=FREQ)
    samples = {name: trajectories(q, paths, n_nowcast, n_paths, rng) for name, q in components.items()}
    ran = [n for n in names if n in components]
    problems = []
    if len(ran) < C.MIN_MODELS:
        problems.append(f"only {len(ran)} model(s) succeeded (need {C.MIN_MODELS})")
        submission = pd.DataFrame(columns=COLUMNS)
    else:
        samples["ensemble"] = trajectories(ensemble({n: components[n] for n in ran}), paths, n_nowcast, n_paths, rng)
        submission = make_frame(samples["ensemble"], weeks, ref_date)
        problems = validate(submission, ref_date, locations["location"])
    return Result(ref_date, pd.Timestamp(as_of), submission, problems, warnings, last_obs, weeks, series, revised,
                  table, samples, ran)


def save(result: Result, root: Path) -> Path | None:
    """Write the submission (if valid), component samples and the nowcast table under `root`."""
    root = Path(root)
    ref = result.ref_date
    rows = []
    for name, by_loc in result.samples.items():
        for loc, traj in by_loc.items():
            for i in range(traj.shape[0]):
                rows += [(name, loc, w, i + 1, float(v)) for w, v in zip(result.weeks, traj[i])]
    d = root / "components"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["model", "location", "target_end_date", "sample", "value"]).to_parquet(
        d / f"{ref}.parquet", index=False)
    d = root / "nowcasts"
    d.mkdir(parents=True, exist_ok=True)
    result.nowcasts.to_csv(d / f"{ref}.csv", index=False)
    if result.problems:
        log.error("Not writing a submission file: %s", "; ".join(result.problems))
        return None
    path = root / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.submission.to_csv(path, index=False)
    log.info("Wrote %s (%d rows, %d locations)", path, len(result.submission), result.submission["location"].nunique())
    return path
