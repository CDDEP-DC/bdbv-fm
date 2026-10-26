"""Command line: `bdbv <command>` (or `python -m bdbv_fm <command>`).

Weekly:
    bdbv forecast        nowcast + forecast this week's submission, and its review report
    bdbv submit          open a pull request to the hub (after reviewing the report)

Occasionally:
    bdbv nowcast-test    replay the nowcast on past data versions and score it
    bdbv backtest        rerun the whole pipeline on past weeks, with the data as they stood then
    bdbv evaluate        score past submissions (or a backtest) against the latest data
    bdbv report          re-render a review report with fresh evaluation scores
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger("bdbv")


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def cmd_forecast(args) -> int:
    from . import data, forecast, report

    current_ref = data.reference_date_for(dt.date.today())
    ref = args.reference_date or current_ref
    if ref.weekday() != 5:
        log.error("Reference date %s is not a Saturday", ref)
        return 2
    locations = data.load_locations()
    target = data.load_target()
    real = data.real_vintages(target)
    if ref < current_ref:
        as_of, frame = data.vintage_for(real, pd.Timestamp(ref) - pd.Timedelta(weeks=1))
        if frame is None:
            log.error("No data release before %s", ref)
            return 1
        log.info("Past reference date: using the data released %s", as_of.date())
    else:
        as_of, frame = data.latest(target)
    vintages = data.nowcast_vintages(target, data.load_delays(locations), as_of) if C.NOWCAST else {}
    result = forecast.run(ref, frame, as_of, locations, args.models or C.MODELS, vintages, prior_dir=args.out)
    for w in result.warnings:
        log.warning(w)
    path = forecast.save(result, args.out)
    if not args.no_report:
        html = report.write(result, args.out, locations)
        print(f"\nReview the forecast: {html.resolve().as_uri()}")
    if path:
        print(f"Submission file: {path}\nWhen it looks right: bdbv submit --reference-date {ref}")
    return 0 if path else 1


def cmd_nowcast_test(args) -> int:
    from . import nowcast_test

    out = nowcast_test.run(args.out)
    print(f"\nTables: {out}")
    return 0


def cmd_backtest(args) -> int:
    from . import backtest

    out = backtest.run(args.start, args.end, args.name, args.models or C.MODELS, every=args.every)
    _print_eval(out)
    return 0


def _print_eval(out: dict) -> None:
    if not out:
        print("Nothing to evaluate yet (no submissions with settled data).")
        return
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.max_rows", 200):
        print(out["by_horizon"].round(3).to_string(index=False))


def cmd_evaluate(args) -> int:
    from . import evaluate

    _print_eval(evaluate.evaluate(args.dir))
    return 0


def cmd_report(args) -> int:
    from . import data, report

    ref = args.reference_date or data.reference_date_for(dt.date.today())
    print(report.rerender(ref, args.out).resolve().as_uri())
    return 0


def cmd_submit(args) -> int:
    from . import data, submit

    ref = args.reference_date or data.reference_date_for(dt.date.today())
    csv = args.out / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv"
    if not csv.exists():
        log.error("No submission file %s; run `bdbv forecast` first", csv)
        return 1
    if not args.hub_dir:
        log.error("Set --hub-dir or BDBV_HUB_DIR to your local clone of your fork of the hub")
        return 2
    if not args.yes:
        if not sys.stdin.isatty():
            log.error("Not a terminal; pass --yes to submit without confirmation")
            return 2
        report = args.out / "reports" / f"{ref}-report.html"
        print(f"About to submit {csv.name} to {C.HUB_REPO}.\nReview report: {report.resolve().as_uri()}")
        if input("Submit? [y/N] ").strip().lower() != "y":
            print("Not submitted.")
            return 1
    submit.submit(csv, args.hub_dir, ref, args.include_metadata)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bdbv", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=C.REPO / "bdbv.toml", help="settings file (default: bdbv.toml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forecast", help="build a week's submission and review report")
    f.add_argument("--reference-date", type=_date, help="Saturday reference date (default: this week's)")
    f.add_argument("--models", nargs="+", help="component models (default: settings)")
    f.add_argument("--out", type=Path, default=C.FORECAST_DIR, help="output folder (default: forecasts/)")
    f.add_argument("--no-report", action="store_true")
    f.set_defaults(func=cmd_forecast)

    n = sub.add_parser("nowcast-test", help="replay the nowcast on past data versions and score it")
    n.add_argument("--out", type=Path, default=C.FORECAST_DIR, help="writes <out>/nowcast-test/ (default: forecasts/)")
    n.set_defaults(func=cmd_nowcast_test)

    b = sub.add_parser("backtest", help="rerun past weeks with the data as they stood then")
    b.add_argument("--start", type=_date, required=True, help="first reference date")
    b.add_argument("--end", type=_date, required=True, help="last reference date")
    b.add_argument("--name", required=True, help="output folder under backtests/")
    b.add_argument("--models", nargs="+")
    b.add_argument("--every", type=int, default=1, help="run every Nth week")
    b.set_defaults(func=cmd_backtest)

    e = sub.add_parser("evaluate", help="score past submissions against the latest data")
    e.add_argument("--dir", type=Path, default=C.FORECAST_DIR, help="forecasts/ or backtests/<name>")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="re-render a review report with fresh evaluation scores")
    r.add_argument("--reference-date", type=_date)
    r.add_argument("--out", type=Path, default=C.FORECAST_DIR)
    r.set_defaults(func=cmd_report)

    s = sub.add_parser("submit", help="open a pull request to the BDBV hub")
    s.add_argument("--reference-date", type=_date)
    s.add_argument("--out", type=Path, default=C.FORECAST_DIR)
    s.add_argument("--hub-dir", type=Path, default=os.environ.get("BDBV_HUB_DIR"))
    s.add_argument("--include-metadata", action="store_true", help="also submit model-metadata/<model>.yml")
    s.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    s.set_defaults(func=cmd_submit)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    changed = C.load_overrides(args.config)
    if changed:
        log.info("Settings from %s: %s", args.config, ", ".join(changed))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
