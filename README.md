# OHTJHU-BM: BDBV nowcasts and forecasts from a foundation-model ensemble

Weekly submissions to the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub)
(2026 Bundibugyo Ebola outbreak, DRC). The pipeline follows
[flusight-fm](../flusight-fm):
1. nowcast the incomplete recent weeks;
2. forecast with a zero-shot ensemble of Chronos-2, TimesFM 2.5, Moirai 2.0 and Timestral;
3. review the result in an offline report;
4. submit by pull request.

All models run locally from open weights.

## Weekly workflow

```bash
bdbv forecast     # nowcast + models + ensemble -> submission file + review report
                  # open the report link it prints and check it
bdbv submit       # opens a pull request to the hub (asks for confirmation)
```

The hub accepts submissions **Sunday to Tuesday** (Kinshasa time, UTC+1) of the week ending on the
`reference_date` Saturday. The target data are released with an `as_of` date. The 2026-10-17
release was committed on Tuesday 2026-10-20, so check the hub before running. If `bdbv forecast`
warns that the data end more than 6 weeks before the last target week, this week's release
probably isn't posted yet.

## What it submits

| | |
|---|---|
| target | `wk inc confirmed cases`: confirmed cases by **week of symptom onset** |
| output | 100 sample trajectories per location (`output_type = sample`) |
| weeks | the last 5 weeks in the data (nowcasts) through `reference_date` + 3 weeks |
| locations | `CD`, plus provinces and health zones with a case in the last 8 weeks (~60) |

With data through onset week 2026-10-17 and reference date 2026-10-31, a file covers
2026-09-19 to 2026-11-21: five nowcast weeks and five forecast weeks.

## The nowcast (the hard part)

Cases are counted by onset week, and reports arrive late. From the hub's case-level delay data, the
median onset-to-report delay is 5 days, the 90th percentile 14 and the 99th about 32. So when a
week's data are released, its newest week holds only about 15–30% of the cases it will eventually
have. The week before holds about 80%, and a tail keeps arriving for a month. Submitting the
reported numbers as-is would be off by a factor of 4–6 in the newest week.

The hub has published only a handful of real releases of the target data, which is far too few
to learn revision patterns from. Instead, `auxiliary-data/weekly-reporting-delays-by-province.csv`
holds one row per case with its onset week and onset-to-report delay. From it, `data.py` rebuilds
**pseudo-vintages**: the country and province counts as they would have stood on any past date.
- Onset is taken as uniform within the week, and a case appears in the data 1.5 days after it is
  reported.
- Both choices were fit to the real releases of 2026-08-18 and 2026-10-09. A simpler "mid-week
  onset" rule undercounted partial weeks by about 2×.

`nowcast.py` (adapted from flusight-fm) then learns, for each of the 5 most recent weeks:
- the log revision as a robust regression on the report's log size, with province effects shrunk
  toward zero (health zones use their province's);
- a revision s.d. that shrinks as a power of the report's size.

It combines that with last week's forecast for the same week, if there is one, and draws 50 joint
nowcast trajectories that are never below the reported count.

`bdbv nowcast-test` replays all of this out of sample. Results from the 2026-10-25 run, as log-scale
CRPS (lower is better):

| | newest week | 1 week back | 2 weeks back |
|---|---|---|---|
| report as-is: pseudo-vintages (country + provinces) | 0.90 | 0.23 | 0.09 |
| **nowcast: pseudo-vintages** | **0.24** | **0.13** | 0.10 |
| report as-is: real releases (all levels) | 1.24 | 0.39 | 0.19 |
| **nowcast: real releases** | **0.50** | **0.27** | 0.17 |

- For the country, the newest week's error drops from a factor of about 20 (log error 3.0) to about 30% (0.1–0.3).
- `size+growth`, the model flusight-fm uses, scored slightly better on the country and provinces
  but much worse on health zones, whose week-over-week changes are mostly noise. That is why the
  default is `size`.
- In-sample calibration under-covered out of sample, so the nowcast s.d. is scaled by 1.3. That
  brings 90% intervals to 84–90% coverage.
- Two weeks back and older, the shift barely matters either way.

Re-run `bdbv nowcast-test` every few weeks. Once there are about 8 weekly real releases, try
`nowcast_source = "both"`.

## How the forecast is built

1. **Data.** Download the hub's target data (all releases), locations and delay data
   (cached under `cache/`).
2. **Nowcast** (above). Each of the 50 nowcast draws is a full series: settled history plus one
   plausible version of the incomplete weeks.
3. **Models** (`models.py`). Every model forecasts every draw for every location. A model that
   fails is logged and left out. A random-walk `baseline` also runs as a reference for
   evaluation; it is never in the ensemble.
4. **Ensemble** (`forecast.py`). Quantiles are mapped onto a 23-level grid and combined by the
   median across models, separately for each draw.
5. **Trajectories.** Sample *i* takes its nowcast weeks from draw *i* mod 50 and samples its future
   weeks from that draw's ensemble quantiles, joined over weeks by an AR(1) Gaussian copula
   (ρ = 0.9). Nowcast uncertainty therefore carries into the forecast.
6. **Outputs**, under `forecasts/`:
   - `model-output/`: the submission file.
   - `components/`: every model's samples, for evaluation.
   - `nowcasts/`: the nowcast tables.
   - `reports/`: the review page.

The review report (`forecasts/reports/<date>-report.html`, a single offline file) shows:
- checks: validation, failed models, the nowcast multipliers, jumps from last week, model
  disagreement;
- an explorer per location: reported data, nowcast + forecast fan, sample trajectories, each
  model's median, last week's median;
- small multiples by level, a table, and evaluation of past submissions.

## Setup

Same as flusight-fm: Python 3.11, CPU is fine, and a CUDA GPU is used if present.

```bash
cd C:\timeseries\bdbv-fm          # keep it outside OneDrive
uv venv --python 3.11             # or: conda create -n bdbv python=3.11 git gh -y && conda activate bdbv
# activate: .venv\Scripts\Activate.ps1 (Windows) | source .venv/bin/activate (macOS/Linux)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # macOS: uv pip install torch
uv pip install -e ".[all]"
bdbv --help
```

`scripts/weekly.ps1` uses `.venv`, or the conda env `bdbv` if there is no `.venv`. If you already
have a working flusight env, `pip install -e .` into it also works: the model dependencies are
identical.

| variable | needed for |
|---|---|
| `BDBV_HUB_DIR` | `bdbv submit`: path to your clone of your fork of the hub |
| `BDBV_DEVICE` | optional: `cpu`, `cuda`, or `mps` |
| `BDBV_CACHE` | optional: cache folder (default `cache/`) |

**Hub fork** (once):

```bash
gh repo fork InsightNet-US/BDBV-Modeling-Hub --clone=false
git clone https://github.com/<you>/BDBV-Modeling-Hub C:\code\BDBV-Modeling-Hub
git -C C:\code\BDBV-Modeling-Hub remote add upstream https://github.com/InsightNet-US/BDBV-Modeling-Hub
```

**Register the model**: fill in the `TODO`s in `model-metadata/OHTJHU-BM.yml`, then run
`bdbv submit --include-metadata` the first time. The team and model abbreviations are set in
`bdbv_fm/config.py` (`TEAM_ABBR`, `MODEL_ABBR`). They default to flusight-fm's `OHTJHU-BM`; change
both there and in the metadata file name if you want a different id for this hub.

**Running it automatically**: `.github/workflows/weekly.yml` runs Mondays 14:00 UTC. With
Task Scheduler:
`schtasks /Create /TN BDBV /SC WEEKLY /D MON /ST 10:00 /TR "powershell -File C:\timeseries\bdbv-fm\scripts\weekly.ps1"`

## Other commands

```bash
bdbv nowcast-test                  # replay + score the nowcast -> forecasts/nowcast-test/
bdbv backtest --start 2026-08-22 --end 2026-10-03 --name zs-1
bdbv evaluate --dir backtests/zs-1 # CRPS / coverage by horizon, relative to the baseline
bdbv evaluate                      # score past submissions in forecasts/
bdbv report                        # re-render this week's report with fresh scores
```

- **Backtests** use pseudo-vintages as each week's data, so they cover the country and provinces
  only (health zones have no delay data).
- **Evaluation** scores a week only once it is 5 weeks old in the latest data. Before that, the
  "truth" is itself incomplete.
- **Horizons** count from the reference date. With data released two weeks before the reference
  date, horizons −6 to −2 are nowcasts and −1 to 3 are forecasts.

## Tuning

Settings live in `bdbv_fm/config.py`. Override any of them in `bdbv.toml` at the repo root, for
example `nowcast_paths = 20` for a faster run. The ones that matter most:

| setting | default | notes |
|---|---|---|
| `NOWCAST_MODEL` | `size` | revision model; compare with `bdbv nowcast-test` |
| `NOWCAST_SD_SCALE` | 1.3 | widens nowcast intervals; check 90% coverage in `nowcast-test` |
| `PIPELINE_LAG_DAYS` | 1.5 | report-to-data lag for pseudo-vintages; refit as real releases accumulate |
| `NOWCAST_SOURCE` | `pseudo` | `both` once ~8 weekly real releases exist |
| `REVISION_WINDOW_WEEKS` | 16 | delays are shortening over time; a shorter window adapts faster |
| `NOWCAST_PATHS` | 50 | model runs per location; lower it for speed, raise it for smoother samples |
| `ACTIVE_WEEKS`, `LEVELS` | 8, all | which locations are submitted |
| `ENSEMBLE_METHOD`, `MODELS` | median, all four | as in flusight-fm; compare in backtests |
| `TRAJ_RHO` | 0.9 | week-to-week correlation inside a trajectory |

## Known limitations and next steps

- **Levels are not coherent.** Health zones don't sum to their province, nor provinces to the
  country. The hub doesn't require it, but reconciling them (e.g. scaling health zone samples to
  province samples) may score better.
- **Small health zones.** Foundation models see short, mostly-zero series. The 8-week activity
  filter keeps out zones with no recent cases. A zone with new spread is only picked up once it
  has a case.
- **Pseudo-vintages are an approximation.** Each week's observed-onset counts are scaled by the
  latest ratio of all cases to observed-onset cases. Imputed-onset cases are assumed to fill in at
  the same pace. Onset day and processing lag are approximated.
- **Pretraining overlap.** Not a concern for this outbreak's data, but the models have seen past
  Ebola series only if those were in their training corpora.
- **Possible additions:** daily onset data (`auxiliary-data/daily-confirmed-case.csv`) for
  within-week timing, and the INRB dashboard as a covariate source.

## Public domain

Code, forecasts and documentation are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (see `LICENSE`), as in flusight-fm.

## Repository layout

```
bdbv_fm/             the package (cli, data, nowcast, nowcast_test, models, forecast, evaluate, report, backtest, submit)
forecasts/           weekly outputs (tracked)
backtests/<name>/    backtest outputs (only evaluation/ is tracked)
model-metadata/      hub metadata file
scripts/             weekly run scripts for Windows and macOS/Linux
cache/               downloads (not tracked)
```
