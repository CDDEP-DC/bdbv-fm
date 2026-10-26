"""Pre-submission review report: a self-contained HTML page per reference date.

The page is written to <root>/reports/<ref>-report.html. It works offline (no
external scripts) and can be opened on any machine or emailed. The data behind it is
saved next to it as <ref>-data.json, so `bdbv report` can re-render it with
up-to-date evaluation scores without rerunning the models.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate
from .forecast import Result

log = logging.getLogger(__name__)
TEMPLATE = Path(__file__).with_name("report_template.html")
BANDS = {"q05": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}
LEVEL_ORDER = {"country": 0, "province": 1, "health zone": 2}
N_SHOWN_TRAJ = 10
HISTORY_WEEKS = 30


def _iso(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _num(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 1)


def _previous_medians(root: Path, ref: dt.date) -> pd.DataFrame:
    """Median per (location, target_end_date) of last week's submission."""
    f = root / "model-output" / C.model_id() / f"{ref - dt.timedelta(weeks=1)}-{C.model_id()}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["location", "target_end_date", "value"])
    df = pd.read_csv(f, dtype={"location": str})
    return df.groupby(["location", "target_end_date"], as_index=False)["value"].median()


def _checks(result: Result, prev: pd.DataFrame, level: dict) -> list[dict]:
    checks = [{"level": "error", "text": p} for p in result.problems]
    checks += [{"level": "warning", "text": w} for w in result.warnings]
    now = result.nowcasts
    if not now.empty:
        latest = now[(now["lag"] == 0) & now["location"].map(lambda l: level.get(l) in ("country", "province"))]
        for _, r in latest[latest["median"] >= 10].iterrows():
            checks.append({"level": "info", "text": f"Nowcast for {r['location']}, week ending {_iso(r['date'])}: "
                                                    f"reported {r['reported']:.0f} → median {r['median']:.0f} "
                                                    f"(×{(r['median'] + 1) / (r['reported'] + 1):.1f})"})
    ens = result.samples.get("ensemble", {})
    ref_week = pd.Timestamp(result.ref_date)
    k = list(result.weeks).index(ref_week) if ref_week in result.weeks else None
    if not prev.empty and k is not None:
        p = prev[pd.to_datetime(prev["target_end_date"]) == ref_week].set_index("location")["value"]
        for loc, old in p.items():
            if loc in ens:
                new = float(np.median(ens[loc][:, k]))
                if max(new, old) >= 20 and not 0.5 <= (new + 1) / (old + 1) <= 2:
                    checks.append({"level": "warning", "text": f"{loc}: median for the reference week ({new:.0f}) is "
                                                               f"more than 2× off last week's ({old:.0f})"})
    if len(result.models) >= 2 and k is not None:
        for loc in ens:
            meds = [float(np.median(result.samples[m][loc][:, k])) for m in result.models if loc in result.samples[m]]
            if np.median(meds) >= 20 and (max(meds) + 1) / (min(meds) + 1) > 3:
                checks.append({"level": "warning", "text": f"{loc}: models disagree for the reference week "
                                                           f"(medians {min(meds):.0f}–{max(meds):.0f})"})
    if not any(c["level"] in ("error", "warning") for c in checks):
        checks.insert(0, {"level": "good", "text": "No problems found"})
    return checks


def build_data(result: Result, root: Path, locations: pd.DataFrame) -> dict:
    ref = result.ref_date
    prev = _previous_medians(root, ref)
    level = dict(zip(locations["location"], locations["location_type"]))
    names = dict(zip(locations["location"], locations["location_name"]))
    weeks = [_iso(w) for w in result.weeks]
    ens = result.samples.get("ensemble", {})
    now = result.nowcasts
    locs, series, forecast, comps, prior, traj = [], {}, {}, {}, {}, {}
    for loc, s in result.reported.items():
        s = s.iloc[-HISTORY_WEEKS:]
        nl = now[now["location"] == loc].sort_values("date") if not now.empty else now
        series[loc] = {"dates": [_iso(d) for d in s.index], "reported": [_num(v) for v in s],
                       "nowcast": {"dates": [_iso(d) for d in nl["date"]], "median": [_num(v) for v in nl["median"]]}}
        locs.append({"code": loc, "name": names.get(loc, loc), "level": level.get(loc, ""),
                     "total": float(result.reported[loc].iloc[-C.ACTIVE_WEEKS:].sum())})
        if loc not in ens:
            continue
        x = ens[loc]
        forecast[loc] = {"dates": weeks} | {b: [_num(v) for v in np.quantile(x, lv, axis=0)] for b, lv in BANDS.items()}
        comps[loc] = {m: [_num(v) for v in np.median(result.samples[m][loc], axis=0)]
                      for m in result.models + [C.BASELINE_MODEL] if loc in result.samples.get(m, {})}
        traj[loc] = [[_num(v) for v in row] for row in x[:N_SHOWN_TRAJ]]
    for loc, g in prev.groupby("location"):
        g = g.sort_values("target_end_date")
        prior[loc] = {"dates": list(g["target_end_date"].astype(str)), "q50": [_num(v) for v in g["value"]]}
    locs.sort(key=lambda l: (LEVEL_ORDER.get(l["level"], 9), -l["total"], l["name"]))
    return {
        "meta": {
            "model_id": C.model_id(),
            "ref_date": str(ref),
            "as_of": _iso(result.as_of),
            "data_through": _iso(result.last_obs),
            "generated": dt.datetime.now().isoformat(timespec="minutes"),
            "machine": f"{platform.node()} ({platform.system()} {platform.machine()})",
            "models": result.models,
            "baseline": C.BASELINE_MODEL,
            "settings": {"ensemble": C.ENSEMBLE_METHOD, "nowcast": C.NOWCAST, "nowcast_model": C.NOWCAST_MODEL,
                         "nowcast_weeks": C.NOWCAST_WEEKS, "nowcast_paths": C.NOWCAST_PATHS},
            "submission_written": not result.problems,
            "counts": {lv: sum(1 for l in locs if l["level"] == lv and l["code"] in ens) for lv in LEVEL_ORDER},
        },
        "checks": _checks(result, prev, level),
        "locations": locs,
        "series": series,
        "forecast": forecast,
        "comps": comps,
        "prev": prior,
        "traj": traj,
    }


def _evaluation(root: Path) -> dict | None:
    try:
        ev = evaluate.evaluate(root)
    except Exception as e:  # evaluation is a convenience; never block the report on it
        log.warning("Evaluation skipped: %s", e)
        return None
    if not ev:
        return None
    clean = lambda df: json.loads(df.to_json(orient="records", date_format="iso"))
    return {"by_horizon": clean(ev["by_horizon"]), "by_level": clean(ev["by_level"])}


def _dump(payload) -> str:
    return json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def render(payload: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", _dump(payload).replace("</", "<\\/"))
    out.write_text(html, encoding="utf-8")
    return out


def write(result: Result, root: Path, locations: pd.DataFrame) -> Path:
    root = Path(root)
    payload = build_data(result, root, locations)
    d = root / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.ref_date}-data.json").write_text(_dump(payload), encoding="utf-8")
    payload["eval"] = _evaluation(root)
    path = render(payload, d / f"{result.ref_date}-report.html")
    log.info("Review report: %s", path)
    return path


def rerender(ref: dt.date, root: Path) -> Path:
    root = Path(root)
    d = root / "reports"
    payload = json.loads((d / f"{ref}-data.json").read_text(encoding="utf-8"))
    payload["eval"] = _evaluation(root)
    return render(payload, d / f"{ref}-report.html")
