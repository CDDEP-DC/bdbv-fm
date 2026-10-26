"""Out-of-sample test of the nowcast: `bdbv nowcast-test`.

For each past data date A (weekly), the revision model is fit using only versions up
to A, the data as it stood on A are nowcast, and the nowcast is scored against the
latest data. Two kinds of "data as it stood on A" are used:

- pseudo-vintages (country and provinces) rebuilt from the case-level delays, every week;
- the hub's real releases (all levels, including health zones), where they exist. These
  check that the pseudo-vintages behave like the real thing.

Each revision model kind (none, constant, size, size+growth) is scored by lag, with the
raw report as a reference. Writes <out>/nowcast-test/{rows,summary}.csv.
"""
from __future__ import annotations

import logging
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as C
from . import data
from . import nowcast as nc
from .forecast import build_series

log = logging.getLogger(__name__)
Z = {p: NormalDist().inv_cdf(0.5 + p / 2) for p in (0.5, 0.9)}


def _crps_normal(mu, sd, y):
    z = (y - mu) / sd
    phi = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
    cdf = np.array([NormalDist().cdf(float(v)) for v in np.atleast_1d(z)])
    return sd * (z * (2 * cdf - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def _score(table: pd.DataFrame, final: pd.Series) -> pd.DataFrame:
    t = table.copy()
    t["truth"] = final.reindex(pd.MultiIndex.from_frame(t[["location", "date"]])).to_numpy()
    t = t.dropna(subset=["truth"])
    off = C.LOG_OFFSET
    y = np.log(t["truth"] + off)
    t["abs_err"] = (np.log(t["median"] + off) - y).abs()
    t["raw_abs_err"] = (np.log(t["reported"] + off) - y).abs()
    t["bias"] = np.log(t["median"] + off) - y
    t["crps_log"] = _crps_normal(t["mu"].to_numpy(), t["sd"].to_numpy(), y.to_numpy())
    for p, z in Z.items():
        lo = np.maximum(np.exp(t["mu"] - z * t["sd"]) - off, t["reported"] if C.NOWCAST_FLOOR else 0)
        hi = np.maximum(np.exp(t["mu"] + z * t["sd"]) - off, t["reported"] if C.NOWCAST_FLOOR else 0)
        t[f"cov{int(p * 100)}"] = ((lo <= t["truth"] + 1e-9) & (t["truth"] <= hi + 1e-9)).astype(float)
    return t


def run(out: Path, kinds=nc.MODEL_KINDS) -> Path:
    locations = data.load_locations()
    target = data.load_target()
    latest_as_of, final = data.latest(target)
    delays = data.load_delays(locations)
    final_s = final.drop_duplicates(["location", "date"], keep="last").set_index(["location", "date"])["value"]
    parent = dict(zip(locations["location"], locations["parent"]))

    n = (latest_as_of - pd.Timestamp(C.REVISION_START)).days // 7
    pseudo = data.pseudo_vintages(delays, final, latest_as_of, n_weeks=n)
    real = data.real_vintages(target)
    last_scored = latest_as_of - pd.Timedelta(weeks=C.REVISION_SETTLE_WEEKS)
    tests = [("pseudo", a, v) for a, v in pseudo.items() if a <= last_scored] + \
            [("real", a, v) for a, v in real.items() if a < latest_as_of]

    rows = []
    for source, a, frame in tests:
        # learn from versions before `a`, built on a's weekday so that lags mean the same thing
        same_day = pseudo if source == "pseudo" else data.pseudo_vintages(delays, final, a, n_weeks=n)
        own = {"pseudo": same_day, "real": real, "both": same_day | real}[C.NOWCAST_SOURCE]
        train = {r: v for r, v in own.items() if r < a} | {a: frame}
        series, _, _ = build_series(frame, locations, active_only=source == "real")
        for kind in kinds:
            models = nc.fit_revisions(train, a, C.LOG_OFFSET, kind)
            _, table = nc.nowcast(series, models, {}, C.LOG_OFFSET, parent)
            if table.empty:
                continue
            level = table["location"].map(dict(zip(locations["location"], locations["location_type"])))
            rows.append(_score(table, final_s).assign(source=source, as_of=a, kind=kind, level=level.reindex(table.index)))
        log.info("Nowcast test: %s version %s done", source, a.date())
    if not rows:
        raise RuntimeError("no versions to test; is there enough history after REVISION_START?")
    df = pd.concat(rows, ignore_index=True)
    df = df[(df["reported"] > 0) | (df["truth"] > 0)]
    # Score only versions where a revision model could be fit at every lag (early in the outbreak there
    # was too little settled history); the others just show the raw report.
    fitted = df[df["kind"] == "constant"].groupby(["source", "as_of"])["n_revisions"].min()
    keep = fitted[fitted > 0].index
    skipped = sorted(str(a.date()) for s, a in fitted.index.difference(keep))
    if skipped:
        log.info("Not scored (no revision model yet): %s", ", ".join(skipped))
    df = df.set_index(["source", "as_of"]).loc[lambda x: x.index.isin(keep)].reset_index()
    summary = (df.groupby(["source", "kind", "lag"])
                 .agg(n=("truth", "size"), abs_err=("abs_err", "mean"), raw_abs_err=("raw_abs_err", "mean"),
                      bias=("bias", "mean"), crps_log=("crps_log", "mean"), cov50=("cov50", "mean"),
                      cov90=("cov90", "mean"))
                 .reset_index())
    d = Path(out) / "nowcast-test"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "rows.csv", index=False)
    summary.to_csv(d / "summary.csv", index=False)
    by_level = (df[df["source"] == "real"].groupby(["kind", "level", "lag"])
                  .agg(n=("truth", "size"), abs_err=("abs_err", "mean"), raw_abs_err=("raw_abs_err", "mean"),
                       crps_log=("crps_log", "mean"), cov50=("cov50", "mean"), cov90=("cov90", "mean"))
                  .reset_index())
    by_level.to_csv(d / "real-by-level.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print("\nNowcast scores by lag (log scale; abs_err = |log(nowcast median+1) - log(final+1)|; "
              "raw = the report as it stood):")
        print(summary.round(3).to_string(index=False))
        if not by_level.empty:
            print("\nReal releases, by level:")
            print(by_level.round(3).to_string(index=False))
    return d
