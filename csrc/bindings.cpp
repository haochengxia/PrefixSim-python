// pybind11 bindings for PrefixSim's simulator (extern/PrefixSim/src).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
#include <optional>

#include "cost_model.hpp"
#include "lcp_writer.hpp"
#include "simulator.hpp"
#include "trace.hpp"
#include "utils/include/mymath.h"

namespace py = pybind11;
using namespace prefixsim;

namespace {

// Put the RNGs where a fresh `prefixsim` process has them (libCacheSim's
// utils/mymath.c, libc's srand(1)), so results match the CLI whatever ran
// before in this process. seed >= 0 picks another stream.
void reset_rng(int64_t seed) {
  if (seed < 0) {
    rand_seed = 0;
    g_lehmer64_state = 0xdeadbeef;
    seed = 1;
  } else {
    set_rand_seed(seed);
  }
  srand(seed);
  srandom(seed);
}

TraceFormat parse_format(const std::string &path, const std::string &name) {
  TraceFormat format;
  if (name == "auto") return guess_trace_format(path);
  if (!parse_trace_format(name, format)) throw py::value_error("trace_format: auto | lcp | qwen-jsonl");
  return format;
}

BlockIdMode parse_block_id(const std::string &name) {
  if (name == "prefix") return BlockIdMode::kPrefix;
  if (name == "raw") return BlockIdMode::kRaw;
  throw py::value_error("block_id: prefix | raw");
}

std::unique_ptr<TraceReader> open(const std::string &path, TraceFormat format,
                                  BlockIdMode block_id = BlockIdMode::kPrefix) {
  std::string error;
  auto reader = open_trace(path, format, block_id, error);
  if (!reader) throw py::value_error(error);
  return reader;
}

py::object positive_or_none(int64_t v) { return v > 0 ? py::object(py::int_(v)) : py::none(); }

// Counts the blocks before each request's first miss, which is what vLLM and
// SGLang reuse; PrefixSim counts hits past a miss too (doc/design.md). The
// simulator calls next() right before serving a request, so probing with the
// cache's read-only find() sees the cache as the simulator's own match does.
class EngineMatch : public TraceReader {
 public:
  EngineMatch(TraceReader *in, const std::vector<Request> &loaded, cache_t *cache,
              const CostModel &cost, int64_t cache_blocks)
      : in_(in), loaded_(loaded), cache_(cache), cost_(cost), cache_blocks_(cache_blocks) {
    memset(&probe_, 0, sizeof(probe_));
    probe_.obj_size = 1;
    probe_.valid = true;
  }

  bool next(Request &r) override {
    if (in_ ? !in_->next(r) : pos_ >= loaded_.size()) return false;
    if (!in_) r = loaded_[pos_++];
    if (exceeds_cache(r, cache_blocks_)) return true;  // skipped by the simulator
    for (size_t p = 0; p < r.blocks.size(); ++p) {
      probe_.obj_id = r.blocks[p];
      if (!cache_->find(cache_, &probe_, false)) break;
      hits++;
      saved += cost_.cost(static_cast<int64_t>(p));
    }
    return true;
  }
  const char *format_name() const override { return in_ ? in_->format_name() : "loaded"; }
  bool annotated() const override { return true; }  // load_trace() annotates
  int64_t n_requests_total() const override {
    return in_ ? in_->n_requests_total() : static_cast<int64_t>(loaded_.size());
  }
  int64_t n_blocks_total() const override { return in_ ? in_->n_blocks_total() : -1; }
  int32_t block_size_tokens() const override { return in_ ? in_->block_size_tokens() : 0; }
  // a loaded trace keeps requests larger than the cache, for the simulator to skip
  int64_t max_blocks_per_request() const override {
    return in_ ? in_->max_blocks_per_request() : -1;
  }

  int64_t hits = 0;
  double saved = 0;

 private:
  TraceReader *in_;
  const std::vector<Request> &loaded_;
  size_t pos_ = 0;
  cache_t *cache_;
  CostModel cost_;
  int64_t cache_blocks_;
  request_t probe_;
};

// One interval report (SimulatorConfig::on_report), with the engine counts at
// the time and over the interval.
struct Interval {
  IntervalReport r;
  int64_t engine_hits = 0, interval_engine_hits = 0;
  double engine_saved = 0, interval_engine_saved = 0;
};

py::object ratio_or_none(bool on, double num, double den) {
  return on ? py::object(py::float_(den > 0 ? num / den : 0)) : py::none();
}

py::dict interval_dict(const Interval &iv, bool engine) {
  const Stats &t = iv.r.total, &i = iv.r.interval;
  py::dict d;
  d["trace_time_s"] = iv.r.trace_time_s;
  d["elapsed_s"] = iv.r.elapsed_s;
  d["n_requests"] = t.n_requests;
  d["n_requests_skipped"] = t.n_requests_skipped;
  d["n_blocks"] = t.n_blocks;
  d["block_hit_ratio"] = t.block_hit_ratio();
  d["compute_saving_ratio"] = t.compute_saving_ratio();
  d["engine_hit_ratio"] = ratio_or_none(engine, double(iv.engine_hits), double(t.n_blocks));
  d["engine_compute_saving_ratio"] = ratio_or_none(engine, iv.engine_saved, t.total_cost);
  d["interval_n_requests"] = i.n_requests;
  d["interval_n_blocks"] = i.n_blocks;
  d["interval_block_hit_ratio"] = i.block_hit_ratio();
  d["interval_compute_saving_ratio"] = i.compute_saving_ratio();
  d["interval_engine_hit_ratio"] =
      ratio_or_none(engine, double(iv.interval_engine_hits), double(i.n_blocks));
  d["interval_engine_compute_saving_ratio"] =
      ratio_or_none(engine, iv.interval_engine_saved, i.total_cost);
  return d;
}

py::dict trace_info(const std::string &path, const std::string &trace_format) {
  auto r = open(path, parse_format(path, trace_format));
  py::dict d;
  d["annotated"] = r->annotated();
  d["n_requests"] = positive_or_none(r->n_requests_total());
  d["n_blocks"] = positive_or_none(r->n_blocks_total());
  d["block_size_tokens"] = positive_or_none(r->block_size_tokens());
  d["max_blocks_per_request"] = positive_or_none(r->max_blocks_per_request());
  return d;
}

py::list simulate(const std::string &path, int64_t cache_blocks,
                  const std::vector<std::string> &algorithms, const std::string &cost_spec,
                  const std::string &params, const std::string &trace_format,
                  const std::string &block_id_name, bool deepest_first, bool verify, bool pin,
                  bool engine, int64_t seed, double report_interval_s, py::object on_report) {
  std::string error;
  CostModel cost;
  if (!CostModel::parse(cost_spec, cost, error)) throw py::value_error("cost_model: " + error);
  const TraceFormat format = parse_format(path, trace_format);
  const BlockIdMode block_id = parse_block_id(block_id_name);
  if (block_id == BlockIdMode::kRaw && cost.position_dependent() &&
      PyErr_WarnEx(PyExc_UserWarning,
                   "block_id='raw' with a position-dependent cost model: a block reached "
                   "through two prefixes gets two costs", 1) < 0) {
    throw py::error_already_set();
  }

  // As the CLI: stream an annotated trace, unless a request is larger than the
  // cache; skipping it would make the stored next-access times wrong, so the
  // trace is loaded and annotated again.
  auto reader = open(path, format, block_id);
  const bool stream = reader->annotated() && reader->max_blocks_per_request() <= cache_blocks;
  std::vector<Request> loaded;
  int64_t n_blocks = reader->n_blocks_total();
  if (!stream) {
    py::gil_scoped_release nogil;
    if (!load_trace(*reader, loaded, error, cache_blocks)) throw std::runtime_error(error);
    reader.reset();
    n_blocks = 0;
    for (const Request &r : loaded) n_blocks += static_cast<int64_t>(r.blocks.size());
  }

  py::list out;
  for (const std::string &algo : algorithms) {
    Stats s;
    SimulatorConfig cfg;
    int64_t hits = 0;
    double saved = 0, secs = 0;
    std::vector<Interval> intervals;
    std::optional<py::error_already_set> report_error;  // from on_report, raised after the run
    {
      py::gil_scoped_release nogil;
      if (stream && !reader) reader = open(path, format, block_id);  // a stream is read once per algorithm
      reset_rng(seed);  // before the cache: some algorithms draw at init
      cache_t *cache = create_cache_by_name(algo, cache_blocks,
                                            params.empty() ? nullptr : params.c_str(), n_blocks);
      if (!cache) throw std::runtime_error("cannot create a '" + algo + "' cache");
      cfg.cache_size_blocks = cache_blocks;
      cfg.cost_model = cost;
      cfg.verify = verify;
      cfg.pin_request_blocks = pin && algorithm_supports_pinning(algo);
      cfg.replay_deepest_first = deepest_first;
      const auto t0 = std::chrono::steady_clock::now();
      bool ok;
      {
        EngineMatch em(stream ? reader.get() : nullptr, loaded, cache, cost, cache_blocks);
        cfg.report_interval_s = report_interval_s;
        cfg.on_report = [&](const IntervalReport &r) {
          Interval iv;
          iv.r = r;
          iv.engine_hits = em.hits;
          iv.engine_saved = em.saved;
          iv.interval_engine_hits = em.hits - (intervals.empty() ? 0 : intervals.back().engine_hits);
          iv.interval_engine_saved = em.saved - (intervals.empty() ? 0 : intervals.back().engine_saved);
          intervals.push_back(iv);
          if (on_report.is_none() || report_error) return;
          py::gil_scoped_acquire gil;
          try {
            on_report(interval_dict(iv, engine));
          } catch (py::error_already_set &e) {
            report_error.emplace(std::move(e));  // stop calling it
          }
        };
        Simulator sim(cache, cfg);
        ok = engine ? sim.run(em, error) : stream ? sim.run(*reader, error) : sim.run(loaded, error);
        s = sim.stats();
        hits = em.hits;
        saved = em.saved;
      }  // the simulator detaches from the cache before it is freed
      secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      cache->cache_free(cache);
      reader.reset();
      if (!ok) throw std::runtime_error("[" + algo + "] " + error);
    }
    if (report_error) throw std::move(*report_error);
    py::dict d;
    d["n_requests"] = s.n_requests;
    d["n_requests_skipped"] = s.n_requests_skipped;
    d["n_blocks"] = s.n_blocks;
    d["block_hit_ratio"] = s.block_hit_ratio();
    d["compute_saving_ratio"] = s.compute_saving_ratio();
    d["engine_hit_ratio"] = engine ? py::object(py::float_(s.n_blocks ? double(hits) / s.n_blocks : 0)) : py::none();
    d["engine_compute_saving_ratio"] = engine ? py::object(py::float_(s.total_cost > 0 ? saved / s.total_cost : 0)) : py::none();
    d["n_evictions"] = s.n_evictions;
    d["n_unexpected_evictions"] = s.n_unexpected_evictions;
    d["pinned"] = cfg.pin_request_blocks;
    d["elapsed_s"] = secs;
    py::list rows;
    for (const Interval &iv : intervals) rows.append(interval_dict(iv, engine));
    d["intervals"] = rows;
    out.append(d);
  }
  return out;
}

py::dict convert(const std::string &input, const std::string &output, int32_t block_size_tokens) {
  auto reader = open(input, TraceFormat::kQwenJsonl);
  std::string error;
  lcp_trace_stat_t st;
  memset(&st, 0, sizeof(st));
  {
    py::gil_scoped_release nogil;
    std::vector<Request> requests;
    const bool zstd = output.size() > 4 && output.compare(output.size() - 4, 4, ".zst") == 0;
    if (!load_trace(*reader, requests, error) ||
        !write_lcp(requests, block_size_tokens, output, zstd, st, error)) {
      throw std::runtime_error(error);
    }
  }
  py::dict d;  // lcp_trace_stat_t is packed: copy the fields out
  d["n_requests"] = int64_t(st.n_req);
  d["n_blocks"] = int64_t(st.n_block);
  d["n_unique_blocks"] = int64_t(st.n_unique_block);
  d["max_blocks_per_request"] = int64_t(st.max_blocks_per_req);
  return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  using py::arg;
  m.def("supported_algorithms", &supported_algorithms);
  m.def("trace_info", &trace_info, arg("path"), arg("trace_format") = "auto");
  m.def("simulate", &simulate, arg("path"), arg("cache_size_blocks"), arg("algorithms"),
        arg("cost_model") = "uniform", arg("algo_params") = "", arg("trace_format") = "auto",
        arg("block_id") = "prefix", arg("replay_deepest_first") = true, arg("verify") = true,
        arg("pin") = true, arg("engine_metrics") = true, arg("seed") = -1,
        arg("report_interval_s") = 0.0, arg("on_report") = py::none());
  m.def("convert", &convert, arg("input"), arg("output"), arg("block_size_tokens") = 16);
}
