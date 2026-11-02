"""Submitting a file to the BDBV hub as a pull request from your fork.

The hub allows several forecasts per reference date, told apart by `forecast_date`. If the hub's
main branch already has a file for this reference date (the original submission), the new rows
are appended to that file as text, so every existing row stays byte-for-byte unchanged, as the
hub's validation requires. Each pull request may add exactly one new forecast_date, and it must be
within one day of today in Kinshasa.
"""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


class SubmissionError(Exception):
    pass


def _read_text_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def new_forecast_date(csv_path: Path) -> dt.date:
    """The single forecast_date in our submission file, checked against the hub's recency rule."""
    df = _read_text_table(csv_path)
    if "forecast_date" not in df.columns:
        raise SubmissionError(f"{csv_path.name} has no forecast_date column; rerun `bdbv forecast`")
    dates = set(df["forecast_date"])
    if len(dates) != 1:
        raise SubmissionError(f"{csv_path.name} has {len(dates)} forecast_date values; expected one")
    fdate = dt.date.fromisoformat(dates.pop())
    today = C.hub_today()
    if abs((fdate - today).days) > 1:
        raise SubmissionError(f"forecast_date {fdate} is more than a day from today in Kinshasa ({today}); "
                              "the hub will reject it. Rerun `bdbv forecast` to make a fresh forecast.")
    return fdate


def existing_forecast_dates(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = _read_text_table(path)
    return set(df["forecast_date"]) if "forecast_date" in df.columns else set()


def merged_text(new_csv: Path, existing: Path) -> str:
    """The existing hub file with our rows appended, in its column order and line endings."""
    old = existing.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in old else "\n"
    header = old.splitlines()[0].split(",")
    new = _read_text_table(new_csv)
    if set(header) != set(new.columns):
        raise SubmissionError(f"columns differ from the hub's file: hub {header}, ours {list(new.columns)}")
    if "forecast_date" not in header:
        raise SubmissionError("the hub's file has no forecast_date column; ask the hub to add it first")
    clash = set(new["forecast_date"]) & existing_forecast_dates(existing)
    if clash:
        raise SubmissionError(f"forecast_date {sorted(clash)} is already in the hub's file; each resubmission "
                              "needs a new forecast_date (rerun `bdbv forecast` on a later day)")
    if not old.endswith("\n"):
        old += nl
    return old + new[header].to_csv(index=False, header=False, lineterminator=nl)


def submit(csv_path: Path, hub_dir: Path, ref_date, include_metadata: bool) -> None:
    """Commit the file to a branch of your fork of the hub and open (or update) a pull request.

    The branch is rebuilt from the hub's main branch each time, so re-running this before the pull
    request is merged replaces its contents rather than adding a second forecast_date to it.
    """
    hub_dir = Path(hub_dir)

    def git(*a):
        subprocess.run(["git", "-C", str(hub_dir), *a], check=True)

    fdate = new_forecast_date(csv_path)
    branch = f"{C.model_id()}-{ref_date}"
    git("fetch", "upstream")
    git("checkout", "-B", branch, f"upstream/{C.HUB_BRANCH}")

    dest = hub_dir / "model-output" / C.model_id() / csv_path.name
    resubmission = dest.exists()
    if resubmission:
        text = merged_text(csv_path, dest)
        log.info("Resubmission: appending forecast_date %s to the hub's file (already has %s)",
                 fdate, ", ".join(sorted(existing_forecast_dates(dest))) or "no forecast_date")
    else:
        text = csv_path.read_bytes().decode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(text.encode("utf-8"))
    git("add", dest.relative_to(hub_dir).as_posix())
    if include_metadata:
        meta = hub_dir / "model-metadata" / f"{C.model_id()}.yml"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_bytes((C.REPO / "model-metadata" / f"{C.model_id()}.yml").read_bytes())
        git("add", meta.relative_to(hub_dir).as_posix())

    kind = "updated nowcast and forecast" if resubmission else "nowcast and forecast"
    git("commit", "-m", f"{C.model_id()} {kind} for {ref_date} (forecast_date {fdate})")
    git("push", "--force", "-u", "origin", branch)

    existing = subprocess.run(
        ["gh", "pr", "list", "--repo", C.HUB_REPO, "--head", branch, "--json", "url", "-q", ".[].url"],
        capture_output=True, text=True, check=True).stdout.strip()
    if existing:
        log.info("Updated existing pull request %s", existing)
        return
    owner = subprocess.run(["gh", "api", "user", "-q", ".login"],
                           capture_output=True, text=True, check=True).stdout.strip()
    title = f"{C.model_id()} {'resubmission' if resubmission else 'submission'} for {ref_date}"
    body = (f"{C.model_id()} {kind} for reference date {ref_date}, forecast_date {fdate}."
            + (" Appends a new forecast to the existing file; earlier rows are unchanged." if resubmission else ""))
    subprocess.run(["gh", "pr", "create", "--repo", C.HUB_REPO, "--base", C.HUB_BRANCH, "--head", f"{owner}:{branch}",
                    "--title", title, "--body", body], check=True)
