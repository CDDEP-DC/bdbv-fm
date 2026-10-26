"""Downloading and caching hub data, and rebuilding past versions of it.

- target data (`time-series.csv`): weekly confirmed cases by week of symptom onset, for the
  country, provinces and health zones. Every release is kept in the file under its `as_of`
  date, so each release is a "real vintage".
- case-level reporting delays by province: one row per case with an observed onset date,
  giving its onset week and the days from onset to report. From these we rebuild
  "pseudo-vintages": the province and national counts as they would have stood on any
  past date. The hub has only a few real releases, but this gives a weekly history of
  how reports fill in, which is what the nowcast learns from.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)


def reference_date_for(day: dt.date) -> dt.date:
    """The Saturday ending the epiweek containing `day`."""
    return day + dt.timedelta(days=(5 - day.weekday()) % 7)


def _get(url: str, **kwargs) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except requests.HTTPError:
            if r.status_code == 404 or attempt == 2:
                raise
        except requests.RequestException:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _fetch(hub_path: str, refresh: bool = True):
    """Download a hub file to the cache (falling back to the cached copy) and return its path."""
    path = C.CACHE_DIR / "hub" / hub_path
    if refresh or not path.exists():
        try:
            text = _get(C.hub_raw(hub_path)).text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except requests.RequestException:
            if not path.exists():
                raise
            log.warning("Could not download %s; using cached copy", hub_path)
    return path


def load_locations(refresh: bool = True) -> pd.DataFrame:
    """location, location_name, location_type, parent (parent as a location code)."""
    df = pd.read_csv(_fetch(C.LOCATIONS_PATH, refresh), dtype=str, keep_default_na=False)
    # the file names each parent; parents are the country or a province, whose names are unique
    prov = df[df["location_type"].isin(["country", "province"])]
    df["parent"] = df["parent"].map(dict(zip(prov["location_name"], prov["location"]))).fillna("")
    return df


def load_target(refresh: bool = True) -> pd.DataFrame:
    """All releases of the target data: as_of, date, location, value."""
    df = pd.read_csv(_fetch(C.TARGET_PATH, refresh), dtype={"location": str},
                     parse_dates=["as_of", "target_end_date"])
    df = df[df["target"] == C.TARGET]
    return df.rename(columns={"target_end_date": "date", "observation": "value"})[["as_of", "date", "location", "value"]]


def load_delays(locations: pd.DataFrame, refresh: bool = True) -> pd.DataFrame:
    """Case-level delays from the latest release: location (province code), week (onset week end), delay (days)."""
    df = pd.read_csv(_fetch(C.DELAYS_PATH, refresh), parse_dates=["week_ending", "as_of"])
    df = df[df["as_of"] == df["as_of"].max()]
    prov = locations[locations["location_type"] == "province"]
    code = dict(zip(prov["location_name"], prov["location"]))
    unknown = set(df["location_name"]) - set(code)
    if unknown:
        log.warning("Delay data for unknown provinces %s; dropped", sorted(unknown))
    return pd.DataFrame({
        "location": df["location_name"].map(code),
        "week": df["week_ending"],
        "delay": df["days_sym_to_report"].clip(lower=0),  # a few negative delays are data errors
    }).dropna(subset=["location"])


# -------------------------------------------------------------------- vintages
def real_vintages(target: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """{as_of: (date, location, value)} for every release in the target file."""
    return {pd.Timestamp(a): g[["date", "location", "value"]].reset_index(drop=True)
            for a, g in target.groupby("as_of")}


def latest(target: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    a = target["as_of"].max()
    return a, target[target["as_of"] == a][["date", "location", "value"]].reset_index(drop=True)


def vintage_for(vintages: dict, day) -> tuple[pd.Timestamp, pd.DataFrame] | tuple[None, None]:
    eligible = [a for a in vintages if a <= pd.Timestamp(day)]
    if not eligible:
        return None, None
    a = max(eligible)
    return a, vintages[a]


def known_by(delays: pd.DataFrame, as_of) -> pd.DataFrame:
    """Expected observed-onset case counts per (location, week) that had reached the hub data by `as_of`.

    Only the onset week is known, so onset is taken as equally likely on each of its 7 days, and a
    case appears in the data PIPELINE_LAG_DAYS after it is reported. (Taking onset as mid-week instead
    badly undercounts partial weeks: checked against the real releases of 2026-08-18 and 2026-10-09.)
    """
    k = (delays["week"] - pd.Timestamp(as_of)).dt.days + delays["delay"] + C.PIPELINE_LAG_DAYS
    p = np.clip(7 - k, 0, 7) / 7   # share of the week's onset days from which the case was in by as_of
    return p.groupby([delays["location"], delays["week"]]).sum().rename("n").reset_index()


def pseudo_vintage(delays: pd.DataFrame, final: pd.DataFrame, as_of) -> pd.DataFrame:
    """The province and national counts as they would have stood on `as_of`.

    Counts of observed-onset cases known by then are scaled up by each week's eventual ratio of
    all cases (including those with imputed onset) to observed-onset cases, from `final`.
    """
    as_of = pd.Timestamp(as_of)
    last_week = pd.Timestamp(reference_date_for(as_of.date()))
    weeks = pd.date_range(C.OUTBREAK_START, last_week, freq="W-SAT")
    provs = sorted(delays["location"].unique())
    obs_final = delays.groupby(["location", "week"]).size()
    known = known_by(delays, as_of).set_index(["location", "week"])["n"]
    idx = pd.MultiIndex.from_product([provs, weeks], names=["location", "week"])
    tot = final.set_index(["location", "date"])["value"]
    tot = tot[~tot.index.duplicated(keep="last")].reindex(idx).fillna(0).to_numpy()
    obs = obs_final.reindex(idx).fillna(0).to_numpy()
    kn = known.reindex(idx).fillna(0).to_numpy()
    # all-imputed weeks (obs == 0) get the country-wide completeness of that week
    share = pd.Series(kn, idx).groupby(level="week").sum() / pd.Series(obs, idx).groupby(level="week").sum().replace(0, np.nan)
    share = share.fillna(1.0).reindex(idx.get_level_values("week")).to_numpy()
    value = np.where(obs > 0, kn * tot / np.maximum(obs, 1), tot * share)
    prov = pd.DataFrame({"date": idx.get_level_values("week"), "location": idx.get_level_values("location"),
                         "value": np.round(value)})
    country = prov.groupby("date", as_index=False)["value"].sum().assign(location="CD")
    return pd.concat([country, prov], ignore_index=True)[["date", "location", "value"]]


def nowcast_vintages(target: pd.DataFrame, delays: pd.DataFrame, as_of) -> dict:
    """The data versions the nowcast learns from on `as_of`, per NOWCAST_SOURCE."""
    real = {a: v for a, v in real_vintages(target).items() if a <= pd.Timestamp(as_of)}
    if C.NOWCAST_SOURCE == "real":
        return real
    current = real.get(pd.Timestamp(as_of))
    if current is None:  # a backtest date between releases: the pseudo-vintage stands in for the data
        current = latest(target)[1]
    pseudo = pseudo_vintages(delays, current, as_of)
    if C.NOWCAST_SOURCE == "pseudo":
        # the newest version, which sets the "settled" values, is the real data where we have them
        return pseudo | ({pd.Timestamp(as_of): real[pd.Timestamp(as_of)]} if pd.Timestamp(as_of) in real else {})
    return pseudo | real


def pseudo_vintages(delays: pd.DataFrame, final: pd.DataFrame, end, n_weeks: int | None = None) -> dict:
    """{as_of: frame} for `end` and each of the n_weeks weeks before it (same weekday)."""
    n_weeks = C.PSEUDO_WEEKS if n_weeks is None else n_weeks
    end = pd.Timestamp(end)
    return {end - pd.Timedelta(weeks=k): pseudo_vintage(delays, final, end - pd.Timedelta(weeks=k))
            for k in range(n_weeks, -1, -1)}
