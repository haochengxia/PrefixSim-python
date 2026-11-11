"""Build the leaderboard site from run.py's JSON lines.

    python leaderboard/build.py results/*.jsonl --out site

Writes site/index.html (self-contained: data, styles and script inline),
site/results.csv and site/results.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess

HIGHLIGHT = ["lru", "s3fifo", "random_compute", "belady_compute"]  # as PrefixSim's README figure
TRACES = {  # from PrefixSim's README
    "freeinference": "Coding and assistant agents on the FreeInference gateway: 327,548 requests over 7.0 days.",
    "chutes": "Chat and roleplay on the Chutes inference proxy: 515,817 requests over 130.9 days.",
}
COLUMNS = ["trace", "algorithm", "cache_size_blocks", "status", "block_hit_ratio", "compute_saving_ratio",
           "engine_hit_ratio", "engine_compute_saving_ratio", "n_requests", "n_requests_skipped", "n_blocks",
           "n_evictions", "pinned", "cost_model", "trace_revision", "prefixsim_version", "elapsed_s", "error"]


def git(root: str, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def source(root: str) -> dict:
    """The PrefixSim and libCacheSim commits a checkout of this repo builds, and their repos."""
    repo = lambda url: url.removesuffix(".git").replace("git@github.com:", "https://github.com/")
    ps = os.path.join(root, "extern", "PrefixSim")
    return dict(prefixsim=git(root, "rev-parse", "HEAD:extern/PrefixSim"),
                prefixsim_repo=repo(git(root, "config", "-f", ".gitmodules", "submodule.extern/PrefixSim.url")),
                libcachesim=git(ps, "rev-parse", "HEAD:libCacheSim"),
                libcachesim_repo=repo(git(ps, "config", "-f", ".gitmodules", "submodule.libCacheSim.url")))


def load(paths: list[str]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            key = (r["trace"], r["algorithm"], r["cache_size_blocks"])
            if key not in rows or r["status"] == "ok":  # a later success replaces a failure
                rows[key] = r
    return sorted(rows.values(), key=lambda r: (r["trace"], r["algorithm"], r["cache_size_blocks"]))


def page_data(rows: list[dict], meta: dict) -> dict:
    order = [t for t in TRACES if any(r["trace"] == t for r in rows)]
    order += sorted({r["trace"] for r in rows} - set(order))
    traces = []
    for t in order:
        mine = [r for r in rows if r["trace"] == t]
        algos: dict[str, dict] = {}
        for r in mine:
            cell = {k: r.get(k) for k in ("status", "block_hit_ratio", "compute_saving_ratio",
                                           "engine_compute_saving_ratio", "elapsed_s", "error")}
            algos.setdefault(r["algorithm"], {})[str(r["cache_size_blocks"])] = cell
        revisions = sorted({r.get("trace_revision", "") for r in mine} - {""})
        traces.append(dict(name=t, desc=TRACES.get(t, ""), revisions=revisions, sizes=sorted({r["cache_size_blocks"] for r in mine}),
                           rows=[dict(algorithm=a, cells=c) for a, c in sorted(algos.items())]))
    return dict(meta=meta, highlight=HIGHLIGHT, traces=traces)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="run.py output files")
    ap.add_argument("--out", default="site")
    ap.add_argument("--source", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="the checkout the runs were built from, for the commits (default: this one)")
    ap.add_argument("--run-url", default="", help="the CI run that produced the results")
    args = ap.parse_args()

    rows = load(args.results)
    versions = sorted({r.get("prefixsim_version", "") for r in rows} - {""})
    meta = dict(version=", ".join(versions), run_url=args.run_url,
                date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                cost_model=rows[0]["cost_model"] if rows else "", **source(args.source))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(meta=meta, results=rows), f, indent=1)
    data = json.dumps(page_data(rows, meta)).replace("</", "<\\/")
    template = open(os.path.join(os.path.dirname(__file__), "template.html")).read()
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(template.replace("__DATA__", data))
    print(f"{len(rows)} runs -> {args.out}/index.html")


if __name__ == "__main__":
    main()
