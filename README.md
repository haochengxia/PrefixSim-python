# PrefixSim-python

Python bindings for [PrefixSim](https://github.com/cacheMon/PrefixSim). The
simulation is PrefixSim's own C++ (the `extern/PrefixSim` submodule), so results
match its CLI. The Python layer adds:

- cache sizes in bytes (`"24GiB"`), from a model's KV size and the trace's block size;
- models from their Hugging Face config: KV bytes per token and a prefill-FLOPs cost model;
- parallel sweeps, one process per (algorithm, cache size);
- engine metrics: hits before each request's first miss, which is what vLLM and SGLang reuse.

## Install

```bash
pip install prefixsim
```

Wheels are prebuilt for Python 3.10–3.14 on Linux (x86_64 and aarch64, glibc
2.34+: Ubuntu 22.04+, Debian 12+, RHEL 9+) and Apple-silicon macOS, with
PrefixSim, libCacheSim, glib and zstd inside: there is nothing to compile or
install first. Elsewhere pip builds from source, which needs a C++17 compiler,
glib, zstd and Python's headers.

## Use

```python
import prefixsim

trace = prefixsim.download_trace("freeinference")  # 76 MB, cached by Hugging Face

# blocks, as the CLI
results = prefixsim.simulate(trace, "16k", "lru,s3fifo", cost_model="qwen3coder30b_blksz_16")

# bytes: the model sets the block's size and the cost model
model = prefixsim.ModelSpec.from_hf("Qwen/Qwen3-Coder-30B-A3B-Instruct")
if __name__ == "__main__":  # sweep spawns worker processes
    results = prefixsim.sweep(trace, ["lru", "s3fifo", "random_compute"],
                              ["24GiB", "48GiB", "96GiB"], model=model, csv_path="results.csv")
```

Each result is a dict. `pandas.DataFrame(results)` gives a table.
`prefixsim.convert("trace.jsonl", "trace.lcp.zst")` turns a JSONL trace into
lcp, which streams instead of being loaded whole.

## Metrics

`block_hit_ratio` and `compute_saving_ratio` are PrefixSim's. They count every
resident block, past a miss too (`doc/design.md`, "A block after a miss still
counts"). `engine_hit_ratio` and `engine_compute_saving_ratio` stop at the
request's first miss. The bindings measure them by probing each request's blocks
just before PrefixSim serves it, which costs about 4% time
(`engine_metrics=False` turns it off). The two can rank policies differently.
Here is freeinference at 16k blocks with `qwen3coder30b_blksz_16`:

| Algorithm | `compute_saving_ratio` | `engine_compute_saving_ratio` |
|---|---:|---:|
| `lru` | 0.5333 | 0.5333 |
| `s3fifo` | 0.5491 | 0.4984 |
| `sieve` | 0.5459 | 0.5036 |
| `random_compute` | 0.6378 | 0.3375 |

They are equal for `lru` only under the defaults, `replay_order="deepest-first"`
and `block_id="prefix"`. In that case a request touches a block's parent after
the block, so the parent is always newer and LRU's victim is a leaf. With
`prefix-order` replay, or with `raw` ids where one id can sit under different
parents, LRU leaves gaps too.

## Notes

- **Models.** Only standard attention (MHA/GQA/MQA) and MLA, dense or MoE, are
  handled. Configs with sliding-window or linear-attention layers are refused
  (`strict=False` counts them as full attention). The bundled traces were cut
  with Qwen3-Coder's tokenizer, so another model changes a block's bytes and
  cost, not where the blocks are.
- **Randomized algorithms.** Their random number generators are reset before
  every run, so a run matches the CLI whatever ran before it; `seed=` picks
  another stream.
- **Aborts.** PrefixSim aborts the process on some errors, for example `s3fifo`
  below 10 blocks. `sweep` runs each simulation in its own process and reports
  which runs were lost.
- **Sizes.** `"64g"` is refused, because the CLI reads it as 2³⁰ × 64 blocks.
  Write `"64GiB"`.
- **tcmalloc** is left out: it cannot be loaded into a running Python process.
- **License.** Apache-2.0, as PrefixSim and libCacheSim. The Linux wheels link
  GLib (LGPL-2.1-or-later) statically; see `third_party/NOTICE.md`.

## Releasing

`.github/workflows/wheels.yml` builds and tests every wheel on each push.
Running it by hand publishes to TestPyPI, and publishing a GitHub release
publishes to PyPI, both through trusted publishing. To build locally:
`uvx cibuildwheel --platform linux` (needs Docker), or `uv build --sdist`.
