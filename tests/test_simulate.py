import pytest

import prefixsim


def test_lru_by_hand(lru_trace):
    (r,) = prefixsim.simulate(lru_trace, 4, "lru")
    assert (r["n_blocks"], r["n_evictions"], r["block_hit_ratio"]) == (14, 3, 0.5)
    assert r["engine_hit_ratio"] == 0.5  # LRU evicts leaves only: no gaps
    # position: block p costs p + 1; total 27, saved 12
    (r,) = prefixsim.simulate(lru_trace, 4, "lru", cost_model="position")
    assert r["compute_saving_ratio"] == r["engine_compute_saving_ratio"] == pytest.approx(12 / 27)


def test_gap(gap_trace):
    fifo, lru = prefixsim.simulate(gap_trace, 3, "fifo,lru", cost_model="position")
    assert (fifo["block_hit_ratio"], fifo["engine_hit_ratio"]) == (2 / 8, 1 / 8)
    # total 11; saved r1's position 0 (1) and r4's position 1 (2); engine r1's only
    assert (fifo["compute_saving_ratio"], fifo["engine_compute_saving_ratio"]) == pytest.approx((3 / 11, 1 / 11))
    assert lru["engine_hit_ratio"] == lru["block_hit_ratio"] == 2 / 8
    (off,) = prefixsim.simulate(gap_trace, 3, "fifo", engine_metrics=False)
    assert off["engine_hit_ratio"] is None and off["block_hit_ratio"] == 2 / 8


def test_lru_gap_depends_on_replay_order(tmp_path):
    # A, then B = A's first 3 blocks, then 3 new blocks into a 5-block cache.
    # deepest-first: B touches 3, 2, 1, so A's 5 and 4 are oldest: no gap.
    # prefix-order: B touches 1, 2, 3, so 1 is evicted under 2 and 3: a gap.
    import json
    a = [1, 2, 3, 4, 5]
    trace = tmp_path / "ab.jsonl"
    trace.write_text("".join(json.dumps({"timestamp": float(i), "hash_ids": r}) + "\n"
                             for i, r in enumerate([a, a[:3], [6, 7, 8], a])))
    (deep,) = prefixsim.simulate(trace, 5, "lru")
    (prefix,) = prefixsim.simulate(trace, 5, "lru", replay_order="prefix-order")
    assert deep["engine_hit_ratio"] == deep["block_hit_ratio"] == 5 / 16
    assert (prefix["block_hit_ratio"], prefix["engine_hit_ratio"]) == (5 / 16, 3 / 16)


def test_lcp_matches_jsonl(tmp_path, gap_trace):
    lcp = tmp_path / "gap.lcp.zst"
    assert prefixsim.convert(gap_trace, lcp)["n_blocks"] == 8
    assert prefixsim.trace_info(lcp)["block_size_tokens"] == 16
    algos = "fifo,lru,sieve,random_compute"  # not s3fifo: it aborts below 10 blocks
    keep = lambda d: {k: v for k, v in d.items() if k not in ("trace", "elapsed_s", "block_size_tokens")}
    strip = lambda rs: [keep(r) | {"intervals": [keep(iv) for iv in r["intervals"]]} for r in rs]
    assert strip(prefixsim.simulate(gap_trace, 3, algos)) == strip(prefixsim.simulate(lcp, 3, algos))


def test_sizes(tmp_path, lru_trace):
    lcp = tmp_path / "lru.lcp.zst"
    prefixsim.convert(lru_trace, lcp)
    # 64 KiB at 1 KiB per token and 16 tokens per block
    (r,) = prefixsim.simulate(lcp, "64KiB", "lru", kv_bytes_per_token=1024)
    assert (r["cache_size_blocks"], r["cache_bytes"], r["block_hit_ratio"]) == (4, 65536, 0.5)
    with pytest.warns(UserWarning, match="larger than the cache"):
        (r,) = prefixsim.simulate(lcp, 2, "lru")
    assert r["n_requests_skipped"] == 4
    block = 1536 * 1024
    assert prefixsim.parse_cache_size("16k") == prefixsim.parse_cache_size("24GiB", block) == 16384
    assert prefixsim.parse_cache_size("80GB", block) == 80 * 10**9 // block
    for bad, block_bytes in [("64g", None), ("24GiB", None), ("1MiB", block), ("0.3k", None), (0, None)]:
        with pytest.raises(ValueError):
            prefixsim.parse_cache_size(bad, block_bytes)


def test_hf_names(tmp_path, lru_trace, monkeypatch):
    # a bare name is a trace of the dataset, a str model a Hugging Face id; both stubbed, no network
    lcp = tmp_path / "lru.lcp.zst"
    prefixsim.convert(lru_trace, lcp)
    fetched = []
    monkeypatch.setattr(prefixsim, "download_trace", lambda name: fetched.append(name) or str(lcp))
    spec = prefixsim.ModelSpec("org/model", 1024, active_params=10**9, attn_flops_per_token=10**5)
    monkeypatch.setattr(prefixsim.ModelSpec, "from_hf", classmethod(lambda cls, repo: spec))
    (r,) = prefixsim.simulate("lru", "64KiB", "lru", model="org/model")
    assert fetched == ["lru"] and r["trace"] == str(lcp)
    assert (r["model"], r["cache_size_blocks"], r["cost_model"]) == ("org/model", 4, spec.cost_model(16, 3))
    monkeypatch.chdir(tmp_path)  # a file in the working directory comes first
    assert prefixsim.trace_info("lru.lcp.zst")["n_requests"] == 5 and fetched == ["lru"]


def test_bad_inputs(tmp_path, lru_trace):
    with pytest.raises(ValueError, match="unsupported"):
        prefixsim.simulate(lru_trace, 4, "lur")
    with pytest.raises(ValueError, match="cost_model"):
        prefixsim.simulate(lru_trace, 4, "lru", cost_model="linear:x")
    with pytest.raises(FileNotFoundError):
        prefixsim.simulate(tmp_path / "missing.lcp.zst", 4, "lru")


def test_intervals(lru_trace, capsys):
    # requests at t = 0..4 s; every 2 s: r0-r2 (9 blocks, 5 hits), then r3-r4 (5 blocks, 2 hits)
    (r,) = prefixsim.simulate(lru_trace, 4, "lru", report_interval=2)
    ivs = r["intervals"]
    assert [iv["trace_time_s"] for iv in ivs] == [2, 4]
    assert [(iv["interval_n_blocks"], iv["interval_block_hit_ratio"]) for iv in ivs] == [(9, 5 / 9), (5, 2 / 5)]
    assert (ivs[-1]["block_hit_ratio"], ivs[-1]["engine_hit_ratio"]) == (r["block_hit_ratio"], r["engine_hit_ratio"])
    assert "lru @ 4 blocks, 0.0 h: hit 0.5556 (interval 0.5556)" in capsys.readouterr().err
    # the default, a day, gets one report: the rest of the trace
    (r,) = prefixsim.simulate(lru_trace, 4, "lru", progress=False)
    assert len(r["intervals"]) == 1 and capsys.readouterr().err == ""
    (r,) = prefixsim.simulate(lru_trace, 4, "lru", report_interval=None)
    assert r["intervals"] == []
    with pytest.raises(ValueError, match="report_interval"):
        prefixsim.simulate(lru_trace, 4, "lru", report_interval="1w")


def test_grid(tmp_path, gap_trace, capsys):
    lcp, out = tmp_path / "gap.lcp.zst", tmp_path / "out.csv"
    prefixsim.convert(gap_trace, lcp)
    expected = [("fifo", 3), ("fifo", 4), ("lru", 3), ("lru", 4)]
    parallel = prefixsim.simulate(lcp, [3, 4], "fifo,lru", jobs=2, csv_path=out)
    assert [(r["algorithm"], r["cache_size_blocks"]) for r in parallel] == expected
    assert parallel[0]["engine_hit_ratio"] == 1 / 8
    assert len(out.read_text().splitlines()) == 5
    assert "  fifo @ 3 blocks, 0.0 h (100.0%): hit 0.2500" in capsys.readouterr().err  # from a worker
    in_process = prefixsim.simulate(lcp, [3, 4], "fifo,lru", jobs=1)
    no_time = lambda d: {k: v for k, v in d.items() if k != "elapsed_s"}
    drop = lambda r: no_time(r) | {"intervals": [no_time(iv) for iv in r["intervals"]]}
    assert [drop(r) for r in in_process] == [drop(r) for r in parallel]
    # libCacheSim aborts on an S3-FIFO below 10 blocks: only its worker dies
    with pytest.raises(RuntimeError, match="worker process died.*s3fifo@3"):
        prefixsim.simulate(lcp, [3, 4], "s3fifo", jobs=2)
