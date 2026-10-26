"""Settings for the BDBV foundation-model pipeline.

Any setting here can be overridden in `bdbv.toml` at the repo root (keys are
case-insensitive), for example:

    models = ["chronos2", "timesfm", "moirai2"]
    nowcast_paths = 20
    pipeline_lag_days = 3

See README.md ("Tuning") for which settings matter most.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FORECAST_DIR = REPO / "forecasts"   # weekly outputs (tracked in git)
BACKTEST_DIR = REPO / "backtests"   # backtest outputs
CACHE_DIR = Path(os.environ.get("BDBV_CACHE", REPO / "cache"))  # downloads (not tracked)

# ------------------------------------------------------------------- identity
TEAM_ABBR = "OHTJHU"
MODEL_ABBR = "BM"
HUB_REPO = "InsightNet-US/BDBV-Modeling-Hub"
HUB_BRANCH = "main"
TARGET_PATH = "target-data/time-series.csv"
DELAYS_PATH = "auxiliary-data/weekly-reporting-delays-by-province.csv"
DAILY_PATH = "auxiliary-data/daily-confirmed-case.csv"
LOCATIONS_PATH = "auxiliary-data/locations.csv"

# --------------------------------------------------------------------- targets
TARGET = "wk inc confirmed cases"
N_SAMPLES = 100             # the hub requires exactly 100 trajectories per location
MAX_HORIZON = 3             # last target week is reference_date + 3 weeks
OUTBREAK_START = "2026-02-14"  # earliest target_end_date the hub accepts
ROUND_SAMPLES = True        # submit whole-number case counts

# Which locations to submit: levels, and only those with a case in the last ACTIVE_WEEKS weeks.
LEVELS = ["country", "province", "health zone"]
ACTIVE_WEEKS = 8
MIN_CASES = 1               # ... and at least this many cases in that window

# Internal quantile grid: each model's quantiles are mapped onto it before ensembling,
# and trajectories are sampled from it.
QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
MEDIAN_IDX = int(np.flatnonzero(QUANTILES == 0.5)[0])

# ---------------------------------------------------------------------- models
MODELS = ["chronos2", "timesfm", "moirai2", "timestral"]
BASELINE_MODEL = "baseline" # always run as a reference for evaluation; never in the ensemble
MIN_MODELS = 2              # need this many successful component models
ENSEMBLE_METHOD = "median"  # "median", "mean" or "weighted"
ENSEMBLE_WEIGHTS: dict = {} # model -> weight, used when ENSEMBLE_METHOD = "weighted"
SPREAD_SCALE = 1.0          # widen (>1) or narrow (<1) ensemble intervals on the log scale
TRANSFORM = "none"          # "none", "sqrt" or "log1p", applied before forecasting
CONTEXT_START = "2026-04-01"   # drop the near-empty weeks before the outbreak took off
TRAJ_RHO = 0.9              # week-to-week correlation of forecast errors within a trajectory
DEVICE = os.environ.get("BDBV_DEVICE", "auto")  # "auto" (CUDA if available, else CPU), "cpu", "cuda", "mps"
SEED = 20261031

# --------------------------------------------------------------------- nowcast
# Recent weeks are very incomplete: about 30% of a week's cases are reported by the
# end of that week and 80% a week later, with a tail beyond a month. The revision
# model is learned from "pseudo-vintages": the province and national counts as they
# would have stood on each past date, rebuilt from the case-level onset-to-report
# delays in auxiliary-data. See nowcast.py and `bdbv nowcast-test`.
NOWCAST = True
NOWCAST_WEEKS = 5           # how many of the most recent reported weeks to revise (and submit)
NOWCAST_MODEL = "size"      # "none", "constant", "size" or "size+growth" (see nowcast.py). "size+growth" did
                            # best on country/province pseudo-vintages but badly on health zones, whose
                            # week-over-week changes are mostly noise (`bdbv nowcast-test`, 2026-10-25)
NOWCAST_SD_SCALE = 1.3      # widen (>1) the nowcast intervals: in-sample calibration under-covers out of
                            # sample (90% intervals held 76-79%); 1.3 gives 84-90% (`bdbv nowcast-test`)
NOWCAST_VARIANCE = "power"  # revision s.d. shrinks as a power of the report's size; "constant" = one s.d. per lag
NOWCAST_SOURCE = "pseudo"   # vintages to learn revisions from: "pseudo" (rebuilt from delays), "real" (hub as_of
                            # versions), or "both". Switch to "both" once there are ~8 weekly real versions.
NOWCAST_PATHS = 50          # nowcast trajectories the models are run on (samples reuse them round-robin)
NOWCAST_RHO = 0.5           # correlation of nowcast errors between adjacent weeks within a trajectory
NOWCAST_FLOOR = True        # a nowcast sample is never below the count already reported
PIPELINE_LAG_DAYS = 1.5     # days from a case's report to its appearance in the hub data (fit to the real
                            # releases of 2026-08-18 and 2026-10-09; recheck as releases accumulate)
PSEUDO_WEEKS = 20           # how many weekly pseudo-vintages to rebuild
AS_OF_LAG_DAYS = 14         # backtests: days from the data's as_of to the reference date (2026-10-17 -> 10-31)
REVISION_START = "2026-06-01"  # earliest data version used to learn revision patterns
REVISION_WINDOW_WEEKS = 16  # ... and only the most recent this many weeks of versions (delays drift)
REVISION_SETTLE_WEEKS = 5   # a week counts as "final" once it is this many weeks old (p99 delay is ~32 days)
REVISION_SHRINK = 5.0       # pseudo-count pulling each location's revision stats toward all locations
REVISION_MIN_SD = 0.01      # floor on revision s.d. (log scale)
NOWCAST_FALLBACK_SD = 0.5   # revision s.d. (log scale) for a lag with no revision history
PRIOR_WEIGHT = 1.0          # weight on our own earlier forecast for a week (0 disables)
LOG_OFFSET = 1.0            # offset in log(value + offset)


def model_id() -> str:
    return f"{TEAM_ABBR}-{MODEL_ABBR}"


def hub_raw(path: str, ref: str | None = None) -> str:
    return f"https://raw.githubusercontent.com/{HUB_REPO}/{ref or HUB_BRANCH}/{path}"


def load_overrides(path: Path) -> list[str]:
    """Apply settings from a TOML file; returns the names that were overridden."""
    if not path.exists():
        return []
    changed = []
    for key, value in tomllib.loads(path.read_text()).items():
        name = key.upper()
        if name not in globals() or name.startswith("_") or callable(globals()[name]):
            raise KeyError(f"unknown setting '{key}' in {path}")
        if isinstance(globals()[name], np.ndarray):
            value = np.asarray(value, float)
        globals()[name] = value
        changed.append(name)
    return changed
