"""Run the leaderboard sweep: every algorithm at every cache size on one trace.

Each run is its own process, so a run that crashes or passes --timeout costs
only itself. Results go to --out as JSON lines, one per run, as runs finish;
runs already in --out with status "ok" are skipped, so a sweep can be resumed.

    python leaderboard/run.py --trace freeinference --out results/fi.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import prefixsim

COST_MODEL = "qwen3coder30b_blksz_16"
# The slowest, from a full sweep (lfuda passed 6 hours at every size; the rest took
# 1 to 2 hours at some size). They start first, so that none of them is the tail.
SLOW = ["lfuda", "random_compute", "s3fifo", "random_compute_small_queue", "hyperbolic", "cacheus", "wtinylfu"]

# One run, in a fresh interpreter: the last stdout line is its result.
_ONE = r"""
import json, sys, prefixsim
trace, size, algo, cost = sys.argv[1:5]
(r,) = prefixsim.simulate(trace, int(size), algo, cost_model=cost, jobs=1, progress=False, report_interval=None)
r.pop("intervals", None)
print(json.dumps(r))
"""


def run_one(trace: str, name: str, blocks: int, algo: str, timeout: float) -> dict:
    # a downloaded trace sits in .../snapshots/<dataset commit>/
    parts = os.path.normpath(trace).split(os.sep)
    revision = parts[-2] if len(parts) > 2 and parts[-3] == "snapshots" else ""
    row = dict(trace=name, algorithm=algo, cache_size_blocks=blocks, cost_model=COST_MODEL, trace_revision=revision,
               prefixsim_version=importlib.metadata.version("prefixsim"))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", _ONE, trace, str(blocks), algo, COST_MODEL],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return row | dict(status="timeout", elapsed_s=time.time() - t0)
    if p.returncode != 0:
        lines = [x for x in p.stderr.splitlines() if x.strip()]
        return row | dict(status="failed", elapsed_s=time.time() - t0,
                          error=(lines[-1] if lines else f"exit {p.returncode}")[:300])
    result = json.loads(p.stdout.strip().splitlines()[-1])
    keep = ("n_requests", "n_requests_skipped", "n_blocks", "block_hit_ratio", "compute_saving_ratio",
            "engine_hit_ratio", "engine_compute_saving_ratio", "n_evictions", "pinned", "elapsed_s")
    return row | {k: result[k] for k in keep} | dict(status="ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="a trace name (freeinference, chutes) or file")
    ap.add_argument("--sizes", default="16k,32k,64k,128k,256k", help="cache sizes in blocks, comma separated")
    ap.add_argument("--algorithms", default="all", help="comma separated, or all")
    ap.add_argument("--shard", default="0/1", help="i/n: run only every n-th algorithm, from the i-th")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="runs at a time")
    ap.add_argument("--timeout", type=float, default=5400, help="seconds before a run is given up")
    ap.add_argument("--out", required=True, help="JSON lines file to append to")
    args = ap.parse_args()

    name = os.path.basename(args.trace).split(".")[0]
    trace = prefixsim.download_trace(args.trace) if not os.path.exists(args.trace) else args.trace
    sizes = [prefixsim.parse_cache_size(s) for s in args.sizes.split(",")]
    algos = prefixsim.supported_algorithms() if args.algorithms == "all" else args.algorithms.split(",")
    i, n = map(int, args.shard.split("/"))
    algos = algos[i::n]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["algorithm"], r["cache_size_blocks"]))
    # the slowest algorithms and the largest caches first, so short runs fill the tail
    algos = sorted(algos, key=lambda a: SLOW.index(a) if a in SLOW else len(SLOW))
    todo = [(a, b) for b in sorted(sizes, reverse=True) for a in algos if (a, b) not in done]
    print(f"{name}: {len(todo)} runs ({len(done)} already done), {args.jobs} at a time", flush=True)

    lock = threading.Lock()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def one(job):
        a, b = job
        r = run_one(trace, name, b, a, args.timeout)
        with lock, open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
            value = f"saving {r['compute_saving_ratio']:.4f}" if r["status"] == "ok" else r["status"]
            print(f"  {name} {a} @ {b}: {value} ({r['elapsed_s']:.0f}s)", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()
