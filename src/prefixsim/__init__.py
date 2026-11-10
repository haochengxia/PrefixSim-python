"""Python bindings for PrefixSim, the KV-block prefix-cache simulator."""

from __future__ import annotations

import csv
import ctypes
import multiprocessing
import os
import re
import signal
import sys
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Sequence

from . import _core
from .model import ModelSpec

__all__ = ["ModelSpec", "convert", "download_trace", "parse_cache_size", "simulate", "supported_algorithms",
           "trace_info"]

_BLOCKS = {"": 1, "k": 2**10, "m": 2**20}  # as the CLI
_BYTES = {"b": 1, "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
          "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12}
_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}  # as the CLI's --report-interval
_PRINT_EVERY_S = 5  # wall-clock seconds between a run's interval lines


TRACES_REPO = "harvardMadsys/2026_prefixcache"


def download_trace(name: str = "freeinference", *, revision: str | None = None) -> str:
    """Download one of PrefixSim's traces from Hugging Face and return its local path.

    Args:
        name: ``"freeinference"`` (76 MB) or ``"chutes"`` (230 MB), or any file of
            the dataset ``harvardMadsys/2026_prefixcache``.
        revision: a dataset commit, for reproducible results. Files are cached
            (``HF_HOME``), so a second call does not download again.
    """
    from huggingface_hub import hf_hub_download

    filename = name if "." in name else f"{name}.lcp.zst"
    return hf_hub_download(TRACES_REPO, filename, repo_type="dataset", revision=revision)


def supported_algorithms() -> list[str]:
    """The eviction algorithms PrefixSim can drive."""
    return _core.supported_algorithms()


def trace_info(trace: str | os.PathLike, trace_format: str = "auto") -> dict:
    """A trace's header: ``annotated``, ``n_requests``, ``n_blocks``,
    ``block_size_tokens``, ``max_blocks_per_request`` (``None`` if not recorded,
    as for JSONL). ``trace`` is a file or a trace name, as for :func:`simulate`."""
    return _core.trace_info(_trace(trace), trace_format)


def convert(input: str | os.PathLike, output: str | os.PathLike, block_size_tokens: int = 16) -> dict:
    """Convert a JSONL trace (a request per line, with ``hash_ids``) to lcp.

    Args:
        input: the JSONL trace; loaded whole, as ``traceConvLLM`` does.
        output: the lcp trace, zstd-compressed if it ends in ``.zst``. lcp
            traces carry next-access times and stream, so simulate those.
        block_size_tokens: tokens per block the ``hash_ids`` were cut at; goes
            into the header.
    """
    return _core.convert(_existing(input), os.fspath(output), block_size_tokens)


def parse_cache_size(size: int | str, block_bytes: int | None = None) -> int:
    """Return a cache size in blocks.

    Args:
        size: blocks (``16384``, ``"16k"``, ``"1m"``, powers of 1024 as in the
            CLI) or bytes (``"24GiB"``, ``"80GB"``). ``"64g"`` is refused: the
            CLI reads it as 2**30 * 64 blocks.
        block_bytes: KV bytes of one block; needed for sizes in bytes, which are
            rounded down to whole blocks.
    """
    blocks = size
    if isinstance(size, str):
        m = re.fullmatch(r"\s*(\d+\.?\d*)\s*([a-zA-Z]*)\s*", size)
        unit = m[2].lower() if m else None
        if unit in _BLOCKS:
            blocks = float(m[1]) * _BLOCKS[unit]
        elif unit in _BYTES:
            if not block_bytes:
                raise ValueError(f"{size!r} is bytes: pass model= or kv_bytes_per_token=")
            blocks = float(m[1]) * _BYTES[unit] // block_bytes
        else:
            raise ValueError(f"bad cache size {size!r}: blocks (16384, 16k) or bytes (24GiB, 80GB)")
    if blocks < 1 or blocks != int(blocks):
        raise ValueError(f"cache size {size!r} is not a positive whole number of blocks")
    return int(blocks)


def simulate(
    trace: str | os.PathLike,
    cache_size: int | str | Sequence[int | str],
    algorithms: str | Sequence[str] = "lru",
    *,
    model: ModelSpec | str | None = None,
    kv_bytes_per_token: int | None = None,
    cost_model: str | None = None,
    block_size_tokens: int | None = None,
    algo_params: str = "",
    trace_format: str = "auto",
    block_id: str = "prefix",
    replay_order: str = "deepest-first",
    verify: bool = True,
    pin: bool = True,
    engine_metrics: bool = True,
    seed: int | None = None,
    jobs: int | None = None,
    csv_path: str | os.PathLike | None = None,
    progress: bool = True,
    report_interval: float | str | None = "1d",
) -> list[dict]:
    """Simulate every algorithm at every cache size over a trace.

    One run executes in this process. Several run in parallel, each in its own
    worker process, so a worker that dies (PrefixSim aborts on some errors, e.g.
    ``s3fifo`` below 10 blocks) loses only its own run. Workers are spawned: a
    script that starts several runs needs ``if __name__ == "__main__":``.

    Args:
        trace: a trace name, ``"freeinference"`` or ``"chutes"``, downloaded
            with :func:`download_trace`; or a file: lcp (``.lcp``, ``.lcp.zst``;
            streamed) or JSONL (loaded whole; :func:`convert` it first).
        cache_size: one size or a list; blocks or bytes, as for
            :func:`parse_cache_size`. Bytes need ``model`` or ``kv_bytes_per_token``.
        algorithms: a name, ``"lru,s3fifo"`` or a list; see :func:`supported_algorithms`.
        model: a Hugging Face model id (``"Qwen/Qwen3-Coder-30B-A3B-Instruct"``,
            read with :meth:`ModelSpec.from_hf`; ``HF_TOKEN`` for gated ones) or
            a :class:`ModelSpec`. Sets the KV bytes of a block and, unless
            ``cost_model`` is given, the cost model, at the trace's block size.
        kv_bytes_per_token: KV bytes of one token, instead of ``model``.
        cost_model: ``uniform``, ``position``, ``qwen3coder30b_blksz_16``,
            ``linear:A,B`` or ``table:FILE``. Default: the model's, else ``uniform``.
        block_size_tokens: tokens per block, for traces without a header (JSONL).
        algo_params: algorithm parameters, e.g. ``"n-sample=64"``.
        trace_format: ``auto`` (from the file name), ``lcp`` or ``qwen-jsonl``.
        block_id: ``prefix`` (a block's id depends on its whole prefix) or ``raw``.
        replay_order: ``deepest-first`` or ``prefix-order``: the order a
            request's blocks are accessed in after it is served.
        verify: check each replayed block is resident, as PrefixSim does
            unless ``--no-verify``.
        pin: keep a request's own blocks from being evicted while it is served,
            for algorithms that support it (``pinned`` in the result).
        engine_metrics: also report ``engine_hit_ratio`` and
            ``engine_compute_saving_ratio``: only the hits before each request's
            first miss, which is what vLLM and SGLang reuse. Costs one extra
            lookup per such block (about 4% time).
        seed: seed for randomised algorithms. ``None`` starts where a fresh
            ``prefixsim`` process does, so results match the CLI.
        jobs: worker processes for several runs; default one per run, up to the
            number of cores. ``1`` runs them one after another in this process.
        csv_path: append each result to this CSV as it finishes (without
            ``intervals``).
        progress: print a line per finished run to stderr, and while it runs,
            its latest interval report at most every 5 seconds.
        report_interval: trace time between interval reports, as cachesim's
            ``report_interval``: seconds, or ``"30m"``, ``"6h"``, ``"1d"``.
            ``None`` or ``0`` for none.

    Returns:
        One dict per run, by algorithm, then cache size: the settings above,
        PrefixSim's ``block_hit_ratio`` and ``compute_saving_ratio`` (every
        resident block counts, past a miss too), the engine ratios, request and
        eviction counts, ``elapsed_s``, and ``intervals``: one dict per
        ``report_interval`` of trace time, plus one for the rest, with
        ``trace_time_s``, ``elapsed_s``, the ratios so far and the same ratios
        over the interval (``interval_block_hit_ratio``, ...).
    """
    if replay_order not in ("deepest-first", "prefix-order"):
        raise ValueError("replay_order: deepest-first | prefix-order")
    if model is not None and kv_bytes_per_token is not None:
        raise ValueError("pass model or kv_bytes_per_token, not both")
    algos = [a.strip().lower() for a in (algorithms.split(",") if isinstance(algorithms, str) else algorithms)]
    unknown = set(algos) - {a.lower() for a in _core.supported_algorithms()}
    if unknown:
        raise ValueError(f"unsupported algorithms {sorted(unknown)}; see supported_algorithms()")

    trace, model = _trace(trace), _model(model)
    info = _core.trace_info(trace, trace_format)
    tokens = info["block_size_tokens"] or block_size_tokens
    if block_size_tokens and tokens != block_size_tokens:
        raise ValueError(f"the trace's blocks are {tokens} tokens; PrefixSim cannot re-block a trace")
    per_token = model.kv_bytes_per_token if model else kv_bytes_per_token
    if per_token and not tokens:
        raise ValueError("the trace does not record its block size: pass block_size_tokens")
    block_bytes = per_token * tokens if per_token else None
    sizes = [cache_size] if isinstance(cache_size, (int, str)) else list(cache_size)
    blocks = [parse_cache_size(s, block_bytes) for s in sizes]
    if cost_model is None:
        cost_model = model.cost_model(tokens, info["max_blocks_per_request"] or 16384) if model else "uniform"
    if info["annotated"] and (info["max_blocks_per_request"] or 0) > min(blocks):
        warnings.warn("a request is larger than the cache, so the whole trace is loaded into memory")

    interval_s = _seconds(report_interval)
    core = dict(cost_model=cost_model, algo_params=algo_params, trace_format=trace_format, block_id=block_id,
                replay_deepest_first=replay_order == "deepest-first", verify=verify, pin=pin,
                engine_metrics=engine_metrics, seed=-1 if seed is None else seed, report_interval_sec=interval_s)
    live = progress and interval_s > 0
    runs = [(a, b) for a in algos for b in blocks]
    results: list[dict | None] = [None] * len(runs)

    def finish(i: int, row: dict) -> None:
        a, b = runs[i]
        r = results[i] = dict(algorithm=a, trace=trace, cache_size_blocks=b,
                              cache_bytes=block_bytes and b * block_bytes, block_size_tokens=tokens,
                              cost_model=cost_model, model=model and model.name, algo_params=algo_params,
                              block_id=block_id, replay_order=replay_order, seed=seed, **row)
        if csv_path:
            _append_csv(csv_path, {k: v for k, v in r.items() if k != "intervals"})
        if progress:
            engine = r["engine_compute_saving_ratio"]
            print(f"[{sum(x is not None for x in results)}/{len(runs)}] {a} @ {b} blocks: "
                  f"hit {r['block_hit_ratio']:.4f}, saving {r['compute_saving_ratio']:.4f}"
                  + ("" if engine is None else f", engine saving {engine:.4f}")
                  + f" ({r['elapsed_s']:.0f}s)", file=sys.stderr, flush=True)

    label = lambda a, b: (f"{a} @ {b} blocks", info["n_requests"]) if live else None
    n_jobs = jobs or min(len(runs), os.cpu_count() or 1)
    if n_jobs == 1:
        for i, (a, b) in enumerate(runs):
            finish(i, _run(trace, b, a, core, label(a, b)))
    else:
        ctx = multiprocessing.get_context("spawn")
        # workers send their interval lines here; a thread prints them
        reports = ctx.Queue() if live else None
        printer = threading.Thread(target=_print_lines, args=(reports,), daemon=True) if live else None
        if printer:
            printer.start()
        try:
            with ProcessPoolExecutor(n_jobs, mp_context=ctx, initializer=_worker_init, initargs=(reports,)) as pool:
                futures = {pool.submit(_run, trace, b, a, core, label(a, b)): i for i, (a, b) in enumerate(runs)}
                try:
                    for f in as_completed(futures):
                        finish(futures[f], f.result())
                except BrokenProcessPool as e:
                    lost = ", ".join(f"{a}@{b}" for (a, b), r in zip(runs, results) if r is None)
                    raise RuntimeError(f"a worker process died (see its stderr); not finished: {lost}") from e
        finally:
            if printer:
                reports.put(None)
                printer.join()
    for r in results:
        if r["n_unexpected_evictions"]:
            warnings.warn(f"{r['algorithm']}: evictions during replay; its numbers are not trustworthy",
                          RuntimeWarning)
    return results


_reports = None  # in a worker: the queue its interval lines go to


def _run(trace: str, blocks: int, algorithm: str, core: dict, live: tuple[str, int | None] | None = None) -> dict:
    on_report = None
    if live:
        printed = [-_PRINT_EVERY_S]

        def on_report(r: dict) -> None:
            if r["elapsed_s"] - printed[0] < _PRINT_EVERY_S:
                return
            printed[0] = r["elapsed_s"]
            line = _interval_line(*live, r)
            if _reports is None:
                print(line, file=sys.stderr, flush=True)
            else:
                _reports.put(line)
    (row,) = _core.simulate(trace, blocks, [algorithm], on_report=on_report, **core)
    return row


def _interval_line(label: str, n_total: int | None, r: dict) -> str:
    seen = r["n_requests"] + r["n_requests_skipped"]
    share = f" ({100 * seen / n_total:.1f}%)" if n_total else ""
    engine = r["engine_compute_saving_ratio"]
    t = r["trace_time_s"]
    when = f"{t / 3600:.1f} h" if t < 2 * 86400 else f"{t / 86400:.1f} d"
    return (f"  {label}, {when}{share}: "
            f"hit {r['block_hit_ratio']:.4f} (interval {r['interval_block_hit_ratio']:.4f}), "
            f"saving {r['compute_saving_ratio']:.4f} (interval {r['interval_compute_saving_ratio']:.4f})"
            + ("" if engine is None else
               f", engine saving {engine:.4f} (interval {r['interval_engine_compute_saving_ratio']:.4f})")
            + f" ({r['elapsed_s']:.0f}s)")


def _print_lines(queue) -> None:
    while (line := queue.get()) is not None:
        print(line, file=sys.stderr, flush=True)


def _seconds(interval: float | str | None) -> int:
    """``report_interval`` in whole seconds, as PrefixSim takes it; 0 for none."""
    if interval is None:
        return 0
    if isinstance(interval, str):
        m = re.fullmatch(r"\s*(\d+\.?\d*)\s*([smhd]?)\s*", interval)
        if not m:
            raise ValueError(f"bad report_interval {interval!r}: seconds, or 30m, 6h, 1d")
        interval = float(m[1]) * _SECONDS[m[2]]
    if interval < 0 or interval != int(interval):
        raise ValueError(f"report_interval must be whole seconds, not {interval!r}")
    return int(interval)


def _worker_init(reports=None) -> None:
    global _reports
    _reports = reports
    # A worker spends its time in C++, out of reach of KeyboardInterrupt: let
    # Ctrl-C kill it outright, and on Linux have it killed when the parent dies.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _append_csv(path, row: dict) -> None:
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def _existing(path) -> str:
    path = os.fspath(path)
    if not os.path.isfile(path):  # libCacheSim would abort the process
        raise FileNotFoundError(path)
    return path


def _trace(trace) -> str:
    """A local file, else a bare name (no directory) is a trace of TRACES_REPO."""
    path = os.fspath(trace)
    if os.path.isfile(path) or os.path.dirname(path):
        return _existing(path)
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return download_trace(path)
    except EntryNotFoundError:
        raise FileNotFoundError(f"{path}: no such file, nor a trace in {TRACES_REPO}") from None


def _model(model: ModelSpec | str | None) -> ModelSpec | None:
    return ModelSpec.from_hf(model) if isinstance(model, str) else model
