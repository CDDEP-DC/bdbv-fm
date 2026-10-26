"""Submitting a file to the BDBV hub as a pull request from your fork."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from . import config as C

log = logging.getLogger(__name__)


def submit(csv_path: Path, hub_dir: Path, ref_date, include_metadata: bool) -> None:
    """Commit the file to a branch of your fork of the hub and open (or update) a pull request."""
    hub_dir = Path(hub_dir)

    def git(*a):
        subprocess.run(["git", "-C", str(hub_dir), *a], check=True)

    branch = f"{C.model_id()}-{ref_date}"
    git("fetch", "upstream")
    git("checkout", "-B", branch, f"upstream/{C.HUB_BRANCH}")
    files = [(csv_path, hub_dir / "model-output" / C.model_id() / csv_path.name)]
    if include_metadata:
        files.append((C.REPO / "model-metadata" / f"{C.model_id()}.yml",
                      hub_dir / "model-metadata" / f"{C.model_id()}.yml"))
    for src, dest in files:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        git("add", dest.relative_to(hub_dir).as_posix())
    git("commit", "-m", f"{C.model_id()} nowcast and forecast for {ref_date}")
    git("push", "--force", "-u", "origin", branch)

    existing = subprocess.run(
        ["gh", "pr", "list", "--repo", C.HUB_REPO, "--head", branch, "--json", "url", "-q", ".[].url"],
        capture_output=True, text=True, check=True).stdout.strip()
    if existing:
        log.info("Updated existing pull request %s", existing)
        return
    owner = subprocess.run(["gh", "api", "user", "-q", ".login"],
                           capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["gh", "pr", "create", "--repo", C.HUB_REPO, "--base", C.HUB_BRANCH, "--head", f"{owner}:{branch}",
                    "--title", f"{C.model_id()} submission for {ref_date}",
                    "--body", f"Weekly {C.model_id()} nowcast and forecast for reference date {ref_date}."], check=True)
