"""Rolling-origin backtests: rerun the whole pipeline on past reference dates.

The hub has published only a few real releases of the target data, so each past
week's data are the pseudo-vintage for that date (see data.py): the country and
province counts as they would have stood AS_OF_LAG_DAYS before the reference date,
rebuilt from the case-level reporting delays. Health zones have no delay data, so
backtests cover the country and provinces only. The nowcast learns only from
pseudo-vintages up to that date, and uses the backtest's own earlier forecasts as
its prior. Outputs go to backtests/<name>/ in the same layout as forecasts/, so
`bdbv evaluate --dir backtests/<name>` scores them.

Caveat: each week's counts are scaled from observed-onset cases to all cases using
the latest data's ratio of the two, a small look-ahead.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import config as C
from . import data, evaluate, forecast
from .models import ModelPool

log = logging.getLogger(__name__)


def run(start: dt.date, end: dt.date, name: str, models, every: int = 1, resume: bool = True) -> dict:
    """Run the backtest and return the evaluation tables. With resume, weeks already done are skipped."""
    root = C.BACKTEST_DIR / name
    locations = data.load_locations()
    target = data.load_target()
    _, final = data.latest(target)
    delays = data.load_delays(locations)
    pool = ModelPool()
    for ref in pd.date_range(data.reference_date_for(start), end, freq=f"{7 * every}D"):
        ref = ref.date()
        done = root / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv"
        if resume and done.exists():
            log.info("Backtest %s: %s already done; skipping", name, ref)
            continue
        as_of = pd.Timestamp(ref) - pd.Timedelta(days=C.AS_OF_LAG_DAYS)
        frame = data.pseudo_vintage(delays, final, as_of)
        vintages = data.pseudo_vintages(delays, final, as_of)
        log.info("=== Backtest %s: reference date %s, data as of %s ===", name, ref, as_of.date())
        result = forecast.run(ref, frame, as_of, locations, models, vintages, prior_dir=root, pool=pool)
        for w in result.warnings:
            log.warning(w)
        forecast.save(result, root)
    return evaluate.evaluate(root)
