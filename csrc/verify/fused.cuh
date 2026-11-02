// Fused decode path. A call launches a fixed sequence of kernels:
// row_stats (validation and each row's maximum), row_scan (temperature-only
// rows drawn in one sliced streaming pass), row_hist and row_gather (top-k /
// top-p candidates by distance from the maximum) and fused_rows, one
// 512-thread block per remaining row, which applies mask, temperature, min-p,
// top-k and top-p, makes the ratio test for its proposal, and draws every
// sample its request may need. The kept set is represented in one of two ways:
//
//  * List: when at most list_max tokens survive top-k, they are gathered,
//    sorted and processed in FP64 with the oracle's own formulas.
//  * Full row: otherwise the kept set is a predicate (an exact key cut plus
//    exact min-p). Sums and Gumbel scores are computed in FP32 together with
//    an error bound. A decision whose interval does not exclude the other
//    outcome is recomputed in FP64, so FP32 rounding never changes a result.
//
// Every row is processed the same way whatever else is in the batch, and
// every reduction has a fixed order, so results are batch invariant.
#pragma once
#include "device.cuh"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <type_traits>
#include <cub/block/block_radix_sort.cuh>

namespace exactsample {
namespace fused {
namespace {
constexpr int THREADS = 512;
constexpr int WARPS = THREADS / 32;
constexpr int UNROLL = 2;         // row groups loaded per thread before any is used
constexpr int LIST_MAX = 256;     // largest kept set processed as an FP64 list
constexpr int TOP_P_LIST = 128;   // tokens a top-p row first tries as a list (FP64 cost grows with it)
constexpr int GATHER = 2048;      // candidate buffer for top-k selection
constexpr int CHUNK32 = 1024;     // top-p scans a full row in sorted chunks of this many tokens
constexpr int CHUNK64 = 256;      // (64-bit keys are rank-sorted, so their chunks are smaller)
// A 32-bit uniform u = (w + 0.5) / 2^32 gives Gumbel noise -log(-log(u)) in
// [-3.13004, 22.87390]. GUMBEL_MAX bounds it from above.
constexpr float GUMBEL_MAX = 22.875f;
// Tokens are skipped only this far below the bar, so a skipped bound never
// reaches the lower bound of a winner that is certain otherwise.
constexpr float PRUNE_MARGIN = 1e-3f;
// Weights below e^-40 are bounded rather than computed.
constexpr float SUM_PRUNE = -40.f;
constexpr double SUM_PRUNE_W = 4.25e-18;
// Error bounds (docs/design.md, Certified FP32).
//   |x32 - x| and |w32 / w - 1| are below X_ERR * (1 + |x|).
//   |s32 - s| for a Gumbel score s = x + G is below SCORE_ERR * (1 + |x| + |G|).
constexpr float X_ERR = 4.8e-7f;
constexpr float SCORE_ERR = 1.9073486e-06f;  // 2^-19
constexpr float LOG_ERR = 4e-7f;              // __logf: |error| <= LOG_ERR * (1 + |result|)

template <DType D> struct In;
template <> struct In<F32> { using T = float; using W = float; using Key = uint32_t; };
template <> struct In<F16> { using T = __half; using W = float; using Key = uint32_t; };
template <> struct In<BF16> { using T = __nv_bfloat16; using W = float; using Key = uint32_t; };
template <> struct In<F64> { using T = double; using W = double; using Key = uint64_t; };

__device__ inline float widen(float v) { return v; }
__device__ inline float widen(__half v) { return __half2float(v); }
__device__ inline float widen(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline double widen(double v) { return v; }

// Unsigned keys in value order. -0 and +0 are equal, as in the oracle.
__device__ inline uint32_t order_key(float v) {
  uint32_t b = __float_as_uint(v == 0.f ? 0.f : v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ inline uint64_t order_key(double v) {
  uint64_t b = uint64_t(__double_as_longlong(v == 0. ? 0. : v));
  return (b >> 63) ? ~b : (b | (1ull << 63));
}

__device__ inline void load4(const float* p, float v[4]) {
  float4 x = *reinterpret_cast<const float4*>(p);
  v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
}
__device__ inline void load4(const double* p, double v[4]) {
  double2 x = reinterpret_cast<const double2*>(p)[0], y = reinterpret_cast<const double2*>(p)[1];
  v[0] = x.x; v[1] = x.y; v[2] = y.x; v[3] = y.y;
}
__device__ inline void load4(const __nv_bfloat16* p, float v[4]) {
  uint2 x = *reinterpret_cast<const uint2*>(p);
  v[0] = __uint_as_float(x.x << 16); v[1] = __uint_as_float(x.x & 0xffff0000u);
  v[2] = __uint_as_float(x.y << 16); v[3] = __uint_as_float(x.y & 0xffff0000u);
}
__device__ inline void load4(const __half* p, float v[4]) {
  uint2 x = *reinterpret_cast<const uint2*>(p);
  v[0] = __half2float(__ushort_as_half(uint16_t(x.x))); v[1] = __half2float(__ushort_as_half(uint16_t(x.x >> 16)));
  v[2] = __half2float(__ushort_as_half(uint16_t(x.y))); v[3] = __half2float(__ushort_as_half(uint16_t(x.y >> 16)));
}

// One logits row. Tokens are read in groups of four, which is also the
// Philox block: one counter yields the noise for all four. VECTOR rows are
// contiguous and aligned (checked on the host), so a group is one load.
template <DType D, bool VECTOR> struct RowView {
  using T = typename In<D>::T;
  using W = typename In<D>::W;
  const T* base;
  int64_t stride;
  int vocab, groups;
  __device__ RowView(const Matrix& m, int64_t row, int vocab_)
      : base(static_cast<const T*>(m.data) + row * m.s0), stride(m.s1), vocab(vocab_), groups((vocab_ + 3) / 4) {}
  __device__ void load(int g, W v[4]) const {
    int t = 4 * g;
    if constexpr (VECTOR) load4(base + t, v);
    else for (int j = 0; j < 4; ++j) v[j] = t + j < vocab ? widen(base[int64_t(t + j) * stride]) : W(-INFINITY);
  }
  __device__ W value(int t) const { return widen(base[int64_t(t) * stride]); }
  // Calls body(g, v) for this thread's groups in [first, last). UNROLL loads
  // are issued before any is used. The trip count is uniform across a warp,
  // so bodies may use full-warp primitives; groups past the range hold -inf.
  template <int NT = THREADS, class Body> __device__ void each(Body&& body, int first = 0, int last = -1) const {
    if (last < 0) last = groups;
    for (int base = first + (threadIdx.x & ~31); base < last; base += NT * UNROLL) {
      const int g0 = base + (threadIdx.x & 31);
      W v[UNROLL][4];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        if (g0 + u * NT < last) load(g0 + u * NT, v[u]);
        else for (int j = 0; j < 4; ++j) v[u][j] = W(-INFINITY);
      }
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) body(g0 + u * NT, v[u]);
    }
  }
};

__device__ inline unsigned group_mask(const Arguments& a, int64_t row, int g, int vocab) {
  int t = 4 * g;
  if (!a.mask || t >= vocab) return a.mask ? 0u : 0xfu;
  return (uint32_t(a.mask[row * a.mask_s0 + (t / 32) * a.mask_s1]) >> (t % 32)) & 0xfu;
}

__device__ inline int lane() { return threadIdx.x & 31; }
__device__ inline int warp() { return threadIdx.x >> 5; }

template <class W> struct Best { W v; int id; };
// Certified argmax. mid is the computed score and [lo, hi] bounds the exact
// one. others bounds every exact score except the winner's, so the winner is
// certain when lo > others.
template <class F> struct Pick { F mid, lo, hi, others; int id; };
struct DD { double hi, lo; };

template <class T> __device__ inline T shfl(T v, int o) { return __shfl_xor_sync(0xffffffffu, v, o); }
template <class W> __device__ inline Best<W> shfl(Best<W> b, int o) { return {shfl(b.v, o), shfl(b.id, o)}; }
template <class F> __device__ inline Pick<F> shfl(Pick<F> p, int o) {
  return {shfl(p.mid, o), shfl(p.lo, o), shfl(p.hi, o), shfl(p.others, o), shfl(p.id, o)};
}
__device__ inline DD shfl(DD d, int o) { return {shfl(d.hi, o), shfl(d.lo, o)}; }

template <class W> __device__ inline Best<W> best_of(Best<W> a, Best<W> b) {
  return b.v > a.v || (b.v == a.v && b.id < a.id) ? b : a;
}
// The two largest (value, ID) pairs in best_of order.
template <class W> struct Top2 { Best<W> first, second; };
template <class W> __device__ inline Top2<W> shfl(Top2<W> t, int o) { return {shfl(t.first, o), shfl(t.second, o)}; }
template <class W> __device__ inline Top2<W> top2_of(Top2<W> a, Top2<W> b) {
  const bool b_wins = b.first.v > a.first.v || (b.first.v == a.first.v && b.first.id < a.first.id);
  const Top2<W> w = b_wins ? b : a, l = b_wins ? a : b;
  return {w.first, best_of(w.second, l.first)};
}
template <class F> __device__ inline Pick<F> no_pick() { return {-INFINITY, -INFINITY, -INFINITY, -INFINITY, INT_MAX}; }
template <class F> __device__ inline Pick<F> pick_of(Pick<F> a, Pick<F> b) {
  bool b_wins = b.mid > a.mid || (b.mid == a.mid && b.id < a.id);
  Pick<F> w = b_wins ? b : a, l = b_wins ? a : b;
  w.others = fmax(fmax(a.others, b.others), l.hi);
  return w;
}
// Double-double addition. The rounding error of the leading sum is exact,
// so the result does not depend on operand order.
__device__ inline DD dd_add(DD a, DD b) {
  double s = a.hi + b.hi, v = s - a.hi, e = (a.hi - (s - v)) + (b.hi - v);
  double lo = (a.lo + b.lo) + e, hi = s + lo;
  return {hi, lo - (hi - s)};
}

// Every thread receives op over all threads' v. The butterfly gives every
// lane bitwise the same value; identity pads the unused warp slots.
template <class T, class Op> __device__ T block_all(T v, Op op, T identity, T* scratch) {
  for (int o = 16; o; o >>= 1) v = op(v, shfl(v, o));
  __syncthreads();
  if (lane() == 0) scratch[warp()] = v;
  __syncthreads();
  v = lane() < WARPS ? scratch[lane()] : identity;
  for (int o = 16; o; o >>= 1) v = op(v, shfl(v, o));
  return v;
}
// op over the 32 lanes of a warp; every lane receives the same value.
template <class T, class Op> __device__ T warp_all(T v, Op op) {
  for (int o = 16; o; o >>= 1) v = op(v, shfl(v, o));
  return v;
}
// Inclusive prefix sum in thread order.
__device__ int block_scan(int v, int* scratch, int* total) {
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, v, o);
    if (lane() >= o) v += y;
  }
  __syncthreads();
  if (lane() == 31) scratch[warp()] = v;
  __syncthreads();
  int w = lane() < WARPS ? scratch[lane()] : 0;
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, w, o);
    if (lane() >= o) w += y;
  }
  *total = __shfl_sync(0xffffffffu, w, 31);
  return v + __shfl_sync(0xffffffffu, w, warp()) - scratch[warp()];
}

// Inclusive prefix sum of doubles in thread order (fixed order, so deterministic).
__device__ double block_scan(double v, double* scratch, double* total) {
  for (int o = 1; o < 32; o <<= 1) {
    double y = __shfl_up_sync(0xffffffffu, v, o);
    if (lane() >= o) v += y;
  }
  __syncthreads();
  if (lane() == 31) scratch[warp()] = v;
  __syncthreads();
  double w = lane() < WARPS ? scratch[lane()] : 0.0;
  for (int o = 1; o < 32; o <<= 1) {
    double y = __shfl_up_sync(0xffffffffu, w, o);
    if (lane() >= o) w += y;
  }
  *total = __shfl_sync(0xffffffffu, w, 31);
  return v + (__shfl_sync(0xffffffffu, w, warp()) - scratch[warp()]);
}

// Gumbel noise from one Philox word, in FP32 (error bound in docs/design.md).
// -log(u) is taken from u when u < 1/2 and from log1p(-(1 - u)) otherwise, so
// the large-noise tail keeps the word's full 32-bit resolution.
__device__ inline float gumbel32(uint32_t w) {
  float e = w < 0x80000000u ? -__logf((float(w) + 0.5f) * 2.3283064365386963e-10f)
                            : -log1pf(-(float(~w) + 0.5f) * 2.3283064365386963e-10f);
  return -__logf(e);
}
// A cheap upper bound on the same value: -log(u) >= 1 - u, and two fast logs.
__device__ inline float gumbel_bound(uint32_t w) {
  float e = w < 0x80000000u ? -__logf((float(w) + 0.5f) * 2.3283064365386963e-10f)
                            : (float(~w) + 0.5f) * 2.3283064365386963e-10f;
  float g = -__logf(e * (1.f - 1e-6f));
  return g + 1e-6f * (1.f + fabsf(g));
}
__device__ inline double gumbel64(uint32_t w) { return -log(-log((double(w) + 0.5) * (1.0 / 4294967296.0))); }

template <class Key> struct Cut {
  bool on;
  Key key;
  int id;
  // Kept iff the token comes at or before (key, id) in (logit desc, id asc) order.
  __device__ bool keeps(Key k, int t) const { return !on || k > key || (k == key && t <= id); }
};

// Which kernel finishes a row: LIST also takes greedy rows and failed rows;
// SLICED rows (temperature-only, one-hot) are scanned by row_scan, several
// blocks per row; its last block finishes the row (DONE) unless the result
// needs FP64, which the FULL kernel provides.
enum Part : int { LIST_PART = 0, FULL_PART = 1, SLICED_PART = 2, DONE_PART = 3 };
// What row_hist and row_gather prepare for a list row.
enum Gather : int { NO_GATHER = 0, TOP_K_GATHER = 1, TOP_P_GATHER = 2 };
// Small batches split each row over several blocks where that is cheap;
// results do not depend on the split (docs/design.md).
__host__ __device__ inline int slice_count(int64_t rows) {
  return int(rows >= 512 ? 1 : rows >= 16 ? 512 / rows : MAX_SLICES);
}
struct StatsPart { double maximum, qmax, runner_up; int greedy, runner_up_id, valid, bad, qvalid, qbad; };
// pad0 and pad1 are the alignment padding, named so that the parts, written
// whole by row_scan and row_hist, have every byte set.
struct DrawPart { Pick<float> pick; int pad0; double sum, err; int pruned, pad1; };
static_assert(sizeof(Pick<float>) == 20 && offsetof(DrawPart, sum) == 24 && offsetof(DrawPart, pruned) == 40 &&
              sizeof(DrawPart) == 48, "DrawPart layout: pad0 and pad1 are padding");
static_assert(sizeof(RowState) + MAX_SLICES * (sizeof(StatsPart) + sizeof(DrawPart)) + CANDIDATES * sizeof(int) +
              (HIST_BINS + 4) * sizeof(unsigned) + 2 * sizeof(unsigned) <= FUSED_ROW_BYTES, "fused workspace layout");
__device__ inline StatsPart* stats_parts(const Arguments& a) {
  return reinterpret_cast<StatsPart*>(row_states(a) + a.rows);
}
__device__ inline DrawPart* draw_parts(const Arguments& a) {
  return reinterpret_cast<DrawPart*>(stats_parts(a) + a.rows * MAX_SLICES);
}
// Run by a whole warp: a row's draw parts merged in a fixed order (one per
// lane, then a butterfly), so a given slice count always gives the same sums.
// Loads go through L2, since other blocks may have written the parts during
// this kernel.
__device__ inline DrawPart merge_draws(const Arguments& a, int row) {
  DrawPart m{no_pick<float>(), 0, 0.0, 0.0, 0, 0};
  if (lane() < slice_count(a.rows)) {
    const DrawPart* q = draw_parts(a) + int64_t(row) * MAX_SLICES + lane();
    m = {{__ldcg(&q->pick.mid), __ldcg(&q->pick.lo), __ldcg(&q->pick.hi), __ldcg(&q->pick.others), __ldcg(&q->pick.id)},
         0, __ldcg(&q->sum), __ldcg(&q->err), __ldcg(&q->pruned), 0};
  }
  auto plus = [](auto x, auto y) { return x + y; };
  m.pick = warp_all(m.pick, [](Pick<float> x, Pick<float> y) { return pick_of(x, y); });
  m.sum = warp_all(m.sum, plus);
  m.err = warp_all(m.err, plus);
  m.pruned = warp_all(m.pruned, plus);
  return m;
}
__device__ inline int* candidates(const Arguments& a, int row) {
  return reinterpret_cast<int*>(draw_parts(a) + a.rows * MAX_SLICES) + int64_t(row) * CANDIDATES;
}
// HIST_BINS + 1 counters per row (the last counts tokens 64 or more logits down).
__device__ inline unsigned* row_hist(const Arguments& a, int row) {
  return reinterpret_cast<unsigned*>(candidates(a, 0) + a.rows * CANDIDATES) + int64_t(row) * (HIST_BINS + 4);
}
template <class W> __device__ inline int hist_bin(W distance) {
  return distance < W(HIST_BINS / 2) ? int(distance * W(2)) : HIST_BINS;
}
// Run by a whole warp: the smallest bin c whose cumulative count reaches n,
// with *count the tokens in bins 0..c; -1 if that needs the last bin or more
// than CANDIDATES tokens.
__device__ inline int window_of(const unsigned* hist, int n, int* count) {
  const int lane_id = threadIdx.x & 31;
  int bins[HIST_BINS / 32], local = 0;
  for (int k = 0; k < HIST_BINS / 32; ++k) local += bins[k] = int(hist[4 * lane_id + k]);
  int incl = local;
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane_id >= o) incl += y;
  }
  const unsigned reach = __ballot_sync(0xffffffffu, incl >= n);
  if (!reach) return -1;
  const int owner = __ffs(reach) - 1;
  int cum = __shfl_sync(0xffffffffu, incl - local, owner), c = -1;
  for (int k = 0; k < HIST_BINS / 32 && c < 0; ++k) {
    cum += __shfl_sync(0xffffffffu, bins[k], owner);
    if (cum >= n) c = 4 * owner + k;
  }
  *count = cum;
  return cum <= CANDIDATES ? c : -1;
}
// Run by a whole warp: the largest non-empty bin c such that bins 0..c hold
// at most cap tokens, with *count their number; -1 if bin 0 alone holds more.
__device__ inline int window_prefix(const unsigned* hist, int cap, int* count) {
  const int lane_id = threadIdx.x & 31;
  int bins[HIST_BINS / 32], local = 0;
  for (int k = 0; k < HIST_BINS / 32; ++k) local += bins[k] = int(hist[4 * lane_id + k]);
  int incl = local;
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane_id >= o) incl += y;
  }
  int best = -1, best_cum = 0, cum = incl - local;
  for (int k = 0; k < HIST_BINS / 32; ++k) {
    cum += bins[k];
    if (bins[k] > 0 && cum <= cap) { best = 4 * lane_id + k; best_cum = cum; }
  }
  for (int o = 16; o; o >>= 1) {  // the largest qualifying bin over the warp
    const int other = __shfl_xor_sync(0xffffffffu, best, o), other_cum = __shfl_xor_sync(0xffffffffu, best_cum, o);
    if (other > best) { best = other; best_cum = other_cum; }
  }
  *count = best_cum;
  return best;
}
// Arrival counters: the last block of a row to finish merges the row's parts in
// slice order, then resets the counter.
__device__ inline bool last_arrival(const Arguments& a, int row, int which, int slices) {
  __shared__ bool last;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned* counter = arrival_counters(a) + 2 * row + which;
    last = atomicAdd(counter, 1u) == unsigned(slices - 1);
    if (last) *counter = 0;
    __threadfence();
  }
  __syncthreads();
  return last;
}
__device__ inline int request_of(const Arguments& a, int row) {
  int lo = 0, hi = int(a.requests) - 1;  // the last request starting at or before this row
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    if (at(a.cu_rows, mid) <= row) lo = mid; else hi = mid - 1;
  }
  return lo;
}

constexpr int SCAN_THREADS = 256;
// Warp then block reduction for the 256-thread blocks of the streaming kernels.
template <class T, class Op> __device__ T scan_reduce(T v, Op op, T* scratch) {
  for (int o = 16; o; o >>= 1) v = op(v, shfl(v, o));
  __syncthreads();
  if (lane() == 0) scratch[warp()] = v;
  __syncthreads();
  v = scratch[0];
  for (int w = 1; w < SCAN_THREADS / 32; ++w) v = op(v, scratch[w]);  // fixed order
  return v;
}

// Validation and maximum of the target row (and of the proposal row for
// full-q draft rows), several blocks per row for small batches.
template <DType D, bool VECTOR>
__global__ void __launch_bounds__(SCAN_THREADS, 6) row_stats(Arguments a) {
  using W = typename In<D>::W;
  __shared__ int s_int[SCAN_THREADS / 32];
  __shared__ W s_val[SCAN_THREADS / 32];
  __shared__ Top2<W> s_top[SCAN_THREADS / 32];
  const int row = blockIdx.y, slice = blockIdx.x, slices = gridDim.x, tid = threadIdx.x;
  if (*reinterpret_cast<const int*>(a.workspace)) return;
  const int req = request_of(a, row);
  if (a.status[req * a.status_s] != OK || a.path[req * a.path_s] != FUSED) return;
  const int end = int(at(a.cu_rows, req + 1)), vocab = int(a.vocab);
  const bool full_q = row + 1 < end && at(a.draft_kind, req) == 1;
  const RowView<D, VECTOR> view(a.target, row, vocab), qview(a.draft, row, vocab);
  const int first = int(int64_t(view.groups) * slice / slices), last = int(int64_t(view.groups) * (slice + 1) / slices);
  auto plus = [](int x, int y) { return x + y; };
  auto maxi = [](int x, int y) { return x > y ? x : y; };
  W best_v = W(-INFINITY);
  int best_id = INT_MAX, bad = 0, valid = 0;
  view.template each<SCAN_THREADS>([&](int g, const W* v) {
    const unsigned bits = group_mask(a, row, g, vocab);
#pragma unroll
    for (int j = 0; j < 4; ++j) {  // branch-free; padding is -inf and fails every test
      bad |= isnan(v[j]) | (v[j] == W(INFINITY));
      const bool ok = ((bits >> j) & 1u) & (v[j] > W(-INFINITY));
      valid += ok;
      const bool take = ok & (v[j] > best_v);  // IDs rise within a thread: ties keep the lowest
      best_v = take ? v[j] : best_v;
      best_id = take ? 4 * g + j : best_id;
    }
  }, first, last);
  // The argmax, and for a row that makes a ratio test, as runner-up, the best
  // of the other threads' maxima: a valid token whose score bounds the
  // residual draw when the draft is the argmax (row_scan).
  const Best<W> none{W(-INFINITY), INT_MAX};
  Top2<W> top2{none, none};
  if (row + 1 < end && a.mode == 0)  // uniform across the block
    top2 = scan_reduce(Top2<W>{{best_v, best_id}, none}, [](Top2<W> x, Top2<W> y) { return top2_of(x, y); }, s_top);
  else
    top2.first = scan_reduce(Best<W>{best_v, best_id}, [](Best<W> x, Best<W> y) { return best_of(x, y); },
                             reinterpret_cast<Best<W>*>(s_top));
  const Best<W> best = top2.first;
  bad = scan_reduce(bad, maxi, s_int);
  valid = scan_reduce(valid, plus, s_int);
  W qmax = W(-INFINITY);
  int qbad = 0, qvalid = 0;
  if (full_q) {
    qview.template each<SCAN_THREADS>([&](int g, const W* y) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        qbad |= isnan(y[j]) | (y[j] == W(INFINITY));
        qvalid += y[j] > W(-INFINITY);
        qmax = y[j] > qmax ? y[j] : qmax;
      }
    }, first, last);
    qmax = scan_reduce(qmax, [](W x, W y) { return x > y ? x : y; }, s_val);
    qbad = scan_reduce(qbad, maxi, s_int);
    qvalid = scan_reduce(qvalid, plus, s_int);
  }
  StatsPart* parts = stats_parts(a) + int64_t(row) * MAX_SLICES;
  if (tid == 0)
    parts[slice] = {double(best.v), double(qmax), double(top2.second.v), best.id, top2.second.id, valid, bad, qvalid, qbad};
  if (!last_arrival(a, row, 0, slices) || warp() != 0) return;
  // Warp 0 of the last block merges the slices, one per lane (slices <= 32),
  // reading through L2 since other blocks wrote them.
  StatsPart all{-INFINITY, -INFINITY, -INFINITY, INT_MAX, INT_MAX, 0, 0, 0, 0};
  if (lane() < slices) {
    const StatsPart* p = &parts[lane()];
    all = {__ldcg(&p->maximum), __ldcg(&p->qmax), __ldcg(&p->runner_up), __ldcg(&p->greedy), __ldcg(&p->runner_up_id),
           __ldcg(&p->valid), __ldcg(&p->bad), __ldcg(&p->qvalid), __ldcg(&p->qbad)};
  }
  const Top2<double> top = warp_all(Top2<double>{{all.maximum, all.greedy}, {all.runner_up, all.runner_up_id}},
                                    [](Top2<double> x, Top2<double> y) { return top2_of(x, y); });
  all.maximum = top.first.v;
  all.greedy = top.first.id;
  all.qmax = warp_all(all.qmax, [](double x, double y) { return fmax(x, y); });
  all.valid = warp_all(all.valid, plus);
  all.bad = warp_all(all.bad, maxi);
  all.qvalid = warp_all(all.qvalid, plus);
  all.qbad = warp_all(all.qbad, maxi);
  for (int i = lane(); i < HIST_BINS + 1; i += 32) row_hist(a, row)[i] = 0;  // row_hist fills it
  if (lane() != 0) return;
  RowState state{all.maximum, all.qmax, all.greedy, all.valid,
                 all.bad ? INVALID_INPUT : all.valid == 0 ? NO_VALID_TOKEN : OK, OK, LIST_PART, 0,
                 top.second.v, top.second.id, 0};
  if (full_q) {
    const int draft = int(a.draft_ids[req * a.draft_s0 + (row - int(at(a.cu_rows, req))) * a.draft_s1]);
    const W y = qview.value(draft);
    state.qstatus = all.qbad ? INVALID_INPUT : all.qvalid == 0 ? NO_VALID_TOKEN
                  : !(exp(double(y) - all.qmax) > 0) ? DRAFT_ZERO_PROB : OK;
  }
  // Which kernel finishes the row, and which candidates row_hist and row_gather prepare.
  const double temperature = at(a.temperature, req);
  const int64_t top_k = at(a.top_k, req);
  const bool failed = state.status != OK || (full_q && state.qstatus != OK);
  const int n_keep = failed ? 0 : top_k > 0 && top_k < state.valid ? int(top_k) : state.valid;
  const int list_max = min(a.list_max, LIST_MAX);
  const double top_p = at(a.top_p, req);
  // Without lists (list_max 0) the list kernels are not launched, and
  // fused_rows (FULL) also finishes greedy and failed rows.
  state.part = failed || temperature == 0 ? (list_max > 0 ? LIST_PART : FULL_PART)
             : n_keep <= list_max ? LIST_PART
             : n_keep == state.valid && top_p == 1 && !full_q ? SLICED_PART : FULL_PART;
  state.gather = state.part == LIST_PART && !failed && temperature != 0 && n_keep < state.valid ? TOP_K_GATHER : NO_GATHER;
  if (state.part == FULL_PART && n_keep == state.valid && top_p < 1 && list_max > 0) {
    // Top-p without top-k: try a list of the largest tokens first (on real
    // logits the nucleus is a handful of them); FULL takes over if it is not enough.
    state.part = LIST_PART;
    state.gather = TOP_P_GATHER;
  }
  row_states(a)[row] = state;
}

// exp(x) > 0 and exp(x) >= min_p in FP64, for the rare token whose FP32 value
// is too close to a threshold. Out of line, so it does not weigh on the scan.
__device__ __noinline__ bool weight_kept_exact(double x, double min_p) {
  const double w = exp(x);
  return w > 0 && w >= min_p;
}

// The streaming pass of sliced rows, kept lean (few registers, many warps per
// SM): the draw, with the sum of weights when the row makes a ratio test. The
// last block to finish merges the slices and commits whatever the certificate
// settles; FP64 fallbacks are left to fused_rows (FULL).
template <DType D, bool VECTOR>
__global__ void __launch_bounds__(SCAN_THREADS, 4) row_scan(Arguments a) {
  using W = typename In<D>::W;
  __shared__ Pick<float> s_pick[SCAN_THREADS / 32];
  __shared__ double s_sum[SCAN_THREADS / 32];
  __shared__ int s_pruned[SCAN_THREADS / 32];
  __shared__ float s_floor;
  const int row = blockIdx.y, slice = blockIdx.x, slices = gridDim.x, tid = threadIdx.x;
  // The row's state and the batch flag first, in one round trip: most rows
  // of most batches are not sliced. A stale state (left by an earlier call
  // for a request that now fails validation) falls through to the checks below.
  const int malformed = *reinterpret_cast<const int*>(a.workspace);
  const RowState state = row_states(a)[row];
  if (malformed || state.part != SLICED_PART) return;
  const int req = request_of(a, row);
  if (a.status[req * a.status_s] != OK || a.path[req * a.path_s] != FUSED) return;
  const int vocab = int(a.vocab);
  const RowView<D, VECTOR> view(a.target, row, vocab);
  const int first = int(int64_t(view.groups) * slice / slices), last = int(int64_t(view.groups) * (slice + 1) / slices);
  const W maximum = W(state.maximum);

  // Sliced draw: substream 0, or the one-hot residual (substream 2, without
  // the proposal) for a row that makes a ratio test.
  const int begin = int(at(a.cu_rows, req)), end = int(at(a.cu_rows, req + 1));
  const bool verify = row + 1 < end && a.mode == 0;
  const int excluded = verify ? int(a.draft_ids[req * a.draft_s0 + (row - begin) * a.draft_s1]) : -1;
  const uint32_t sub = verify ? 2 : 0;
  const double temperature = at(a.temperature, req), min_p = at(a.min_p, req);
  const uint64_t seed = uint64_t(at(a.seed, req));
  const uint32_t stream = uint32_t(at(a.stream, req)), position = uint32_t(at(a.position, row));
  const float inv_t = float(1.0 / temperature);
  const bool slow_x = !(inv_t < 1e30f);
  const float ln_min_p = min_p > 0 ? float(log(min_p)) : -INFINITY;
  const float min_p_tol = min_p > 0 ? 1e-6f * (1.f + fabsf(ln_min_p)) : 0.f;
  auto x64 = [&](W v) { return (double(v) - double(maximum)) / temperature; };
  auto x32 = [&](W v) -> float {
    if constexpr (sizeof(W) == 8) return float(x64(v));
    else {
      float d = v - maximum;
      return slow_x ? float(double(d) / temperature) : d * inv_t;
    }
  };
  if (tid == 0) {
    // A kept candidate's score bounds the winner's from below: the greedy
    // token's (always kept, x = 0), or the runner-up's when the greedy token is
    // the excluded draft, as it usually is. The runner-up must be certainly
    // kept (min-p, and a weight above FP64 underflow); otherwise no floor.
    const bool runner = excluded == state.greedy;
    const int t = runner ? state.runner_up_id : state.greedy;
    const float x = runner ? x32(W(state.runner_up)) : 0.f;
    float floor = -INFINITY;
    if (t != INT_MAX && (!runner || (x > -744.f && x - ln_min_p > X_ERR * fabsf(x) + min_p_tol))) {
      const Uint4 r4 = philox(seed, stream, position, sub, uint32_t(t / 4));
      const uint32_t words[4] = {r4.x, r4.y, r4.z, r4.w};
      const float g = gumbel32(words[t % 4]);
      floor = x + g - SCORE_ERR * (1.f + fabsf(x) + fabsf(g));
    }
    s_floor = floor;
  }
  __syncthreads();
  float bar = s_floor - PRUNE_MARGIN, skip = -INFINITY, s = 0, c = 0, e = 0;
  int pruned = 0;
  Pick<float> best = no_pick<float>();
  view.template each<SCAN_THREADS>([&](int g, const W* v) {
    const unsigned bits = group_mask(a, row, g, vocab);
    float xs[4];
    unsigned live = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = xs[j] = x32(v[j]);
      // exp(x) is certainly 0 below -746.5 and certainly positive above -744;
      // only a token near that or near min-p needs the FP64 test.
      const bool in = ((bits >> j) & 1u) & (v[j] > W(-INFINITY)) & (x >= -746.5f);
      const bool clear = (x > -744.f) & (fabsf(x - ln_min_p) > X_ERR * fabsf(x) + min_p_tol);
      const bool keep = in && (clear ? x > ln_min_p : weight_kept_exact(x64(v[j]), min_p));
      if (verify) {
        const bool counted = keep & (x >= SUM_PRUNE);
        pruned += keep & !counted;
        const float w = counted ? expf(x) : 0.f, d = w - c, t = s + d;
        c = (t - s) - d; s = t;
        e += counted ? w * (X_ERR * (1.f + fabsf(x))) : 0.f;
      }
      const bool candidate = keep & (4 * g + j != excluded);
      const float ub = x + GUMBEL_MAX + SCORE_ERR * (1.f + fabsf(x) + GUMBEL_MAX);
      const bool alive = candidate & (ub >= bar);
      skip = fmaxf(skip, candidate & !alive ? ub : -INFINITY);
      live |= unsigned(alive) << j;
    }
    if (!live) return;
    const Uint4 r4 = philox(seed, stream, position, sub, uint32_t(g));
    const uint32_t words[4] = {r4.x, r4.y, r4.z, r4.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (!((live >> j) & 1u)) continue;
      const float b = gumbel_bound(words[j]);
      const float ub = xs[j] + b + SCORE_ERR * (1.f + fabsf(xs[j]) + fabsf(b));
      if (ub < bar) { skip = fmaxf(skip, ub); continue; }
      const float G = gumbel32(words[j]), score = xs[j] + G;
      const float err = SCORE_ERR * (1.f + fabsf(xs[j]) + fabsf(G));
      best = pick_of(best, Pick<float>{score, score - err, score + err, -INFINITY, 4 * g + j});
      bar = fmaxf(bar, best.lo - PRUNE_MARGIN);
    }
  }, first, last);
  best.others = fmaxf(best.others, skip);
  best = scan_reduce(best, [](Pick<float> x, Pick<float> y) { return pick_of(x, y); }, s_pick);
  double sum = 0, err = 0;  // only a ratio test accumulates them
  if (verify) {
    sum = scan_reduce(double(s) - double(c), [](double x, double y) { return x + y; }, s_sum);
    err = scan_reduce(double(e), [](double x, double y) { return x + y; }, s_sum);
    pruned = scan_reduce(pruned, [](int x, int y) { return x + y; }, s_pruned);
  }
  if (tid == 0) draw_parts(a)[int64_t(row) * MAX_SLICES + slice] = {best, 0, sum, err, pruned, 0};
  // The last slice to finish merges them (warp 0) and decides whatever is
  // certain; anything else is left for fused_rows (FULL), which has the FP64
  // fallbacks.
  if (!last_arrival(a, row, 1, slices) || warp() != 0) return;
  const DrawPart merged = merge_draws(a, row);
  const Pick<float> pick = merged.pick;
  const double sp = merged.sum, err_sum = merged.err;
  const int pruned_sum = merged.pruned;
  if (lane() != 0) return;
  const bool certain = pick.id != INT_MAX && pick.lo > pick.others;
  RowResult result{0, 1, -1, -1, OK, OK};
  if (verify) {
    // Ratio test against a one-hot proposal: p(x) = w(x) / sum, u < p(x).
    const double esp = 1.01 * err_sum + 4.8e-7 * sp + pruned_sum * SUM_PRUNE_W;
    const W vx = view.value(excluded);
    const float xx = x32(vx);
    const bool kx = allowed(a, row, excluded) && vx > W(-INFINITY) && xx >= -746.5f &&
                    ((xx > -744.f) & (fabsf(xx - ln_min_p) > X_ERR * fabsf(xx) + min_p_tol) ? xx > ln_min_p
                                                                                          : weight_kept_exact(x64(vx), min_p));
    const double wx = kx ? exp(x64(vx)) : 0.0, u = uniform(seed, stream, position, 1, 0);
    bool accept = false;
    if (wx > 0) {
      const double lo = fmin(1.0, wx / (sp + esp)) * (1 - 1e-12), hi = fmin(1.0, wx / (sp - esp)) * (1 + 1e-12);
      if (!(esp >= 0 && sp - esp > 0) || !(u < lo || u >= hi)) return;  // not certain: FULL decides
      accept = u < lo;
    }
    result.p = accept ? 1.0 : 0.0;
    if (!accept) {
      if (!certain) return;
      result.residual = pick.id;
    }
  } else {
    if (!certain) return;
    result.sample = pick.id;
  }
  reinterpret_cast<RowResult*>(a.workspace + 8)[row] = result;
  row_states(a)[row].part = DONE_PART;
}

// For list rows: the histogram of distances below the row maximum (into
// row_hist), from which row_gather takes the bins it needs; for top-p rows
// also the FP32 sum of kept weights, with its bound, into the draw parts.
template <DType D, bool VECTOR>
__global__ void __launch_bounds__(SCAN_THREADS, 4) row_hist(Arguments a) {
  using W = typename In<D>::W;
  __shared__ unsigned s_bins[HIST_BINS];
  __shared__ double s_sum[SCAN_THREADS / 32];
  __shared__ int s_count[SCAN_THREADS / 32];
  const int row = blockIdx.y, slice = blockIdx.x, slices = gridDim.x, tid = threadIdx.x;
  const int malformed = *reinterpret_cast<const int*>(a.workspace);  // as in row_scan
  const RowState state = row_states(a)[row];
  if (malformed || state.gather == NO_GATHER) return;
  const int req = request_of(a, row);
  if (a.status[req * a.status_s] != OK || a.path[req * a.path_s] != FUSED) return;
  const int vocab = int(a.vocab);
  const RowView<D, VECTOR> view(a.target, row, vocab);
  const int first = int(int64_t(view.groups) * slice / slices), last = int(int64_t(view.groups) * (slice + 1) / slices);
  const W maximum = W(state.maximum);
  const bool top_p = state.gather == TOP_P_GATHER;
  const double temperature = at(a.temperature, req), min_p = at(a.min_p, req);
  const float inv_t = float(1.0 / temperature);
  const bool slow_x = !(inv_t < 1e30f);
  const float ln_min_p = min_p > 0 ? float(log(min_p)) : -INFINITY;
  const float min_p_tol = min_p > 0 ? 1e-6f * (1.f + fabsf(ln_min_p)) : 0.f;
  auto x64 = [&](W v) { return (double(v) - double(maximum)) / temperature; };
  auto x32 = [&](W v) -> float {
    if constexpr (sizeof(W) == 8) return float(x64(v));
    else {
      float d = v - maximum;
      return slow_x ? float(double(d) / temperature) : d * inv_t;
    }
  };
  for (int i = tid; i < HIST_BINS; i += SCAN_THREADS) s_bins[i] = 0;
  __syncthreads();
  int far = 0, pruned = 0;
  float s = 0, c = 0, e = 0;
  view.template each<SCAN_THREADS>([&](int g, const W* v) {
    const unsigned bits = group_mask(a, row, g, vocab);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (!(((bits >> j) & 1u) && v[j] > W(-INFINITY))) continue;
      const int b = hist_bin(maximum - v[j]);
      if (b < HIST_BINS) atomicAdd(&s_bins[b], 1u); else ++far;
      if (top_p) {
        const float x = x32(v[j]);
        if (!(x >= -746.5f)) continue;
        const bool clear = (x > -744.f) & (fabsf(x - ln_min_p) > X_ERR * fabsf(x) + min_p_tol);
        if (!(clear ? x > ln_min_p : weight_kept_exact(x64(v[j]), min_p))) continue;
        if (x < SUM_PRUNE) { ++pruned; continue; }
        const float w = expf(x), d = w - c, t = s + d;
        c = (t - s) - d; s = t;
        e += w * (X_ERR * (1.f + fabsf(x)));
      }
    }
  }, first, last);
  __syncthreads();
  unsigned* hist = row_hist(a, row);
  for (int i = tid; i < HIST_BINS; i += SCAN_THREADS)
    if (s_bins[i]) atomicAdd(&hist[i], s_bins[i]);
  far = scan_reduce(far, [](int x, int y) { return x + y; }, s_count);
  if (tid == 0 && far) atomicAdd(&hist[HIST_BINS], unsigned(far));
  if (!top_p) return;
  const double sum = scan_reduce(double(s) - double(c), [](double x, double y) { return x + y; }, s_sum);
  const double err = scan_reduce(double(e), [](double x, double y) { return x + y; }, s_sum);
  pruned = scan_reduce(pruned, [](int x, int y) { return x + y; }, s_count);
  if (tid == 0) draw_parts(a)[int64_t(row) * MAX_SLICES + slice] = {no_pick<float>(), 0, sum, err, pruned, 0};
}

// Gathers the top-k candidates of list rows: every valid token in the
// histogram bins that hold the k largest, if they fit the candidate buffer.
template <DType D, bool VECTOR>
__global__ void __launch_bounds__(SCAN_THREADS, 4) row_gather(Arguments a) {
  using W = typename In<D>::W;
  __shared__ int s_window;
  const int row = blockIdx.y, slice = blockIdx.x, slices = gridDim.x, tid = threadIdx.x;
  const int malformed = *reinterpret_cast<const int*>(a.workspace);  // as in row_scan
  const RowState state = row_states(a)[row];
  if (malformed || state.gather == NO_GATHER) return;
  const int req = request_of(a, row);
  if (a.status[req * a.status_s] != OK || a.path[req * a.path_s] != FUSED) return;
  if (tid < 32) {  // top-k: the bins holding the k largest; top-p: the largest prefix a list can hold
    int count;
    const int c = state.gather == TOP_K_GATHER ? window_of(row_hist(a, row), int(at(a.top_k, req)), &count)
                                               : window_prefix(row_hist(a, row), TOP_P_LIST, &count);
    if (tid == 0) s_window = c;
  }
  __syncthreads();
  const int window = s_window;
  if (window < 0) return;  // fused_rows selects over the row instead
  const int vocab = int(a.vocab);
  const RowView<D, VECTOR> view(a.target, row, vocab);
  const int first = int(int64_t(view.groups) * slice / slices), last = int(int64_t(view.groups) * (slice + 1) / slices);
  const W maximum = W(state.maximum);
  int* out = candidates(a, row);
  unsigned* count = arrival_counters(a) + 2 * row + 1;
  view.template each<SCAN_THREADS>([&](int g, const W* v) {
    const unsigned bits = group_mask(a, row, g, vocab);
    unsigned take = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      take |= unsigned(((bits >> j) & 1u) & (v[j] > W(-INFINITY)) && hist_bin(maximum - v[j]) <= window) << j;
    const int c = __popc(take);
    if (!__any_sync(0xffffffffu, c)) return;
    int incl = c;
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane() >= o) incl += y;
    }
    unsigned base = 0;
    if (lane() == 31) base = atomicAdd(count, unsigned(incl));
    base = __shfl_sync(0xffffffffu, base, 31) + unsigned(incl - c);
    for (int j = 0; j < 4; ++j)
      if ((take >> j) & 1u) { if (base < CANDIDATES) out[base] = 4 * g + j; ++base; }
  }, first, last);
}

// Tokens that come after `cut` in (logit desc, id asc) order; every token when !on.
template <class Key> struct After {
  bool on;
  Cut<Key> cut;
  __device__ bool admits(Key k, int t) const { return !on || !cut.keeps(k, t); }
};
// Reduction operators of fused_rows.
struct PlusOp { template <class T> __device__ T operator()(T x, T y) const { return x + y; } };
struct MinOp { __device__ int operator()(int x, int y) const { return x < y ? x : y; } };
struct DDOp { __device__ DD operator()(DD x, DD y) const { return dd_add(x, y); } };
struct Best64Op { __device__ Best<double> operator()(Best<double> x, Best<double> y) const { return best_of(x, y); } };
__device__ inline Best<double> no_best64() { return {-INFINITY, INT_MAX}; }

// Shared memory of fused_rows: what both kernels use, and what only the
// list (LIST_PART) or only the full and sliced paths use.
struct ListShared {
  int s_sorted[LIST_MAX];
  double s_p[LIST_MAX];
  int s_n2, s_pick_id, s_defer, s_settled;  // warp 0's results, for the whole block
};
struct FullShared {
  Best<double> s_best64[WARPS];
  Pick<float> s_pick32[WARPS];
  DD s_scan[THREADS];
  float s_floor;
  DrawPart s_merged;
};
template <DType D, int PART> struct RowShared : std::conditional_t<PART == LIST_PART, ListShared, FullShared> {
  using Key = typename In<D>::Key;
  using Sort = cub::BlockRadixSort<unsigned long long, THREADS, 4>;
  struct Buffer { Key key[GATHER]; int id[GATHER]; };
  int s_int[WARPS];
  double s_double[WARPS];
  DD s_dd[WARPS];
  unsigned s_hist[256];
  union Space { Buffer buf; typename Sort::TempStorage sort; } s_space;  // candidates, or the sort's storage
  Key s_keys[WARPS];
  int s_a, s_b, s_n;
};

// One row of fused_rows, run by the whole block: the request's parameters,
// the row's state from row_stats and its result, and the steps that finish
// the row. Every member function is inlined into the kernel.
template <DType D, bool VECTOR, int PART> struct RowWork {
  using W = typename In<D>::W;
  using Key = typename In<D>::Key;
  using Sort = typename RowShared<D, PART>::Sort;
  static constexpr int KEY_BITS = int(sizeof(Key) * 8);
  static constexpr int PER_LANE = LIST_MAX / 32;  // list entries per lane of warp 0
  RowShared<D, PART>& sh;
  Key* const s_key;  // candidate buffer, GATHER entries
  int* const s_id;
  const Arguments& a;
  const int row, tid, req, begin, end;
  const bool draft_row, coupled;
  const int draft_id;
  const bool full_q, verify;  // verify: this row makes a ratio test
  const double temperature, top_p, min_p;
  const int64_t top_k;
  const uint64_t seed;
  const uint32_t stream, position;
  const int vocab;
  const RowView<D, VECTOR> view, qview;
  const int groups;
  const RowState state;
  const W maximum, qmax;
  const int greedy, valid;
  const bool failed;
  const int n_keep;
  RowResult result;
  // Actual proposal distribution q for full-q draft rows: q at the draft
  // token is wq_draft / sq, and esq bounds the error of sq.
  double wq_draft = 0, sq = 0, esq = 0;
  bool sq_exact = false;
  // Positive temperature (begin_positive).
  float inv_t = 0, ln_min_p = 0, min_p_tol = 0;
  bool slow_x = false;
  double u_accept = 0;
  // The top-k (and, once found, top-p) cut of the kept set, and the sum of
  // the kept weights with its error bound.
  Cut<Key> cut{false, 0, 0};
  double sp = 0, esp = 0;
  bool sp_known = false, sp_exact = false;

  __device__ __forceinline__ RowWork(const Arguments& a_, RowShared<D, PART>& sh_, int row_, int req_)
      : sh(sh_), s_key(sh_.s_space.buf.key), s_id(sh_.s_space.buf.id), a(a_), row(row_), tid(threadIdx.x),
        req(req_), begin(int(at(a_.cu_rows, req_))), end(int(at(a_.cu_rows, req_ + 1))), draft_row(row_ + 1 < end),
        coupled(a_.mode == 1),
        draft_id(draft_row ? int(a_.draft_ids[req_ * a_.draft_s0 + (row_ - begin) * a_.draft_s1]) : -1),
        full_q(draft_row && at(a_.draft_kind, req_) == 1), verify(draft_row && !coupled),
        temperature(at(a_.temperature, req_)), top_p(at(a_.top_p, req_)), min_p(at(a_.min_p, req_)),
        top_k(at(a_.top_k, req_)), seed(uint64_t(at(a_.seed, req_))), stream(uint32_t(at(a_.stream, req_))),
        position(uint32_t(at(a_.position, row_))), vocab(int(a_.vocab)), view(a_.target, row_, vocab),
        qview(a_.draft, row_, vocab), groups(view.groups), state(row_states(a_)[row_]), maximum(W(state.maximum)),
        qmax(W(state.qmax)), greedy(state.greedy), valid(state.valid),
        failed(state.status != OK || (full_q && state.qstatus != OK)),
        n_keep(failed ? 0 : top_k > 0 && top_k < valid ? int(top_k) : valid),
        result{0, 1, -1, -1, state.status, state.status} {}

  // Counts FP64 recomputations (workspace word 1) so tests and benchmarks can report them.
  __device__ __forceinline__ void fallback() const {
    if (tid == 0) atomicAdd(reinterpret_cast<unsigned*>(a.workspace) + 1, 1u);
  }
  __device__ __forceinline__ void finish() const {
    if (tid == 0) reinterpret_cast<RowResult*>(a.workspace + 8)[row] = result;
  }
  __device__ __forceinline__ void finish_failed() {
    if (state.status == OK) result.status = state.qstatus;
    finish();
  }

  // ---- Actual proposal distribution q for full-q draft rows. ----
  __device__ __forceinline__ double y64(W y) const { return double(y) - double(qmax); }
  __device__ __forceinline__ float yq32(W y) const { return float(y - qmax); }
  __device__ __forceinline__ void q_sum64() {
    fallback();
    DD acc{0, 0};
    qview.each([&](int g, const W* y) {
      for (int j = 0; j < 4 && 4 * g + j < vocab; ++j)
        if (y[j] != W(-INFINITY)) acc = dd_add(acc, DD{exp(y64(y[j])), 0});
    });
    sq = block_all(acc, DDOp{}, DD{0, 0}, sh.s_dd).hi;
    esq = 0;
    sq_exact = true;
  }
  // q at the draft token, and an FP32 sum of q weights with an error bound.
  __device__ __forceinline__ void q_sum32() {
    wq_draft = exp(y64(qview.value(draft_id)));
    float s = 0, c = 0, e = 0;
    int pruned = 0;
    qview.each([&](int g, const W* y) {
      for (int j = 0; j < 4 && 4 * g + j < vocab; ++j) {
        if (y[j] == W(-INFINITY)) continue;
        float x = yq32(y[j]);
        if (x < SUM_PRUNE) { ++pruned; continue; }
        float w = expf(x), d = w - c, t = s + d;
        c = (t - s) - d; s = t;
        e += w * (X_ERR * (1.f + fabsf(x)));
      }
    });
    sq = block_all(double(s) - double(c), PlusOp{}, 0.0, sh.s_double);
    double et = block_all(double(e), PlusOp{}, 0.0, sh.s_double);
    pruned = block_all(pruned, PlusOp{}, 0, sh.s_int);
    esq = 1.01 * et + 4.8e-7 * sq + pruned * SUM_PRUNE_W;
  }

  // ---- Greedy rows: the target is a point mass at the lowest-ID maximum. ----
  __device__ __forceinline__ void finish_greedy() {
    result.sample = greedy;
    if (verify) {
      // p is 1 at the greedy token and 0 elsewhere, so q never changes the test.
      result.p = draft_id == greedy ? 1.0 : 0.0;
      if (draft_id != greedy) {
        if (!full_q) result.residual = greedy;
        else {
          // max(p - q, 0) is positive only at the greedy token, unless q puts all its mass there.
          W y = qview.value(greedy);
          double wq = y == W(-INFINITY) ? 0.0 : exp(y64(y));
          if (!(wq * (1 + 1e-12) < sq - esq)) q_sum64();
          result.residual = 1.0 - wq / sq > 0 ? greedy : -1;
        }
      }
    }
    finish();
  }

  // ---- Positive temperature. ----
  __device__ __forceinline__ void begin_positive() {
    inv_t = float(1.0 / temperature);
    slow_x = !(inv_t < 1e30f);
    ln_min_p = min_p > 0 ? float(log(min_p)) : -INFINITY;
    min_p_tol = min_p > 0 ? 1e-6f * (1.f + fabsf(ln_min_p)) : 0.f;
    u_accept = uniform(seed, stream, position, 1, 0);
  }
  __device__ __forceinline__ double x64(W v) const { return (double(v) - double(maximum)) / temperature; }
  __device__ __forceinline__ float x32(W v) const {
    if constexpr (sizeof(W) == 8) return float(x64(v));
    else {
      float d = v - maximum;
      return slow_x ? float(double(d) / temperature) : d * inv_t;
    }
  }
  // exp(x64) > 0 and exp(x64) >= min_p, decided exactly: FP32 settles every
  // token whose value is clear of the threshold, FP64 the rest.
  __device__ __forceinline__ bool weight_kept(W v, float x) const {
    // Common case, one comparison chain: clear of both thresholds (ln_min_p is -inf without min-p).
    if ((x > -744.f) & (fabsf(x - ln_min_p) > X_ERR * fabsf(x) + min_p_tol)) return x > ln_min_p;
    if (!(x > -744.f) && (x < -746.5f || !(exp(x64(v)) > 0))) return false;
    if (!(min_p > 0)) return true;
    float tol = X_ERR * fabsf(x) + min_p_tol;
    if (x > ln_min_p + tol) return true;
    if (x < ln_min_p - tol) return false;
    return exp(x64(v)) >= min_p;
  }

  // ---- Selection: exact cuts in (logit desc, id asc) order. ----
  // The n-th entry, in (key desc, id asc) order, of s_key/s_id[0, m): radix
  // select on the key, then on the ID among entries with that key.
  __device__ __forceinline__ Cut<Key> buffer_select(int m, int n) {
    int need = n;
    Key prefix = 0, known = 0;
    for (int shift = KEY_BITS - 8; shift >= 0; shift -= 8) {
      if (tid < 256) sh.s_hist[tid] = 0;
      __syncthreads();
      for (int i = tid; i < m; i += THREADS)
        if ((s_key[i] & known) == prefix) atomicAdd(&sh.s_hist[(s_key[i] >> shift) & 255], 1u);
      __syncthreads();
      int count = tid < 256 ? int(sh.s_hist[255 - tid]) : 0, total;  // descending digits
      int cum = block_scan(count, sh.s_int, &total);
      if (tid < 256 && cum >= need && cum - count < need) { sh.s_a = 255 - tid; sh.s_b = cum - count; sh.s_n = count; }
      __syncthreads();
      need -= sh.s_b;
      prefix |= Key(sh.s_a) << shift;
      known |= Key(255) << shift;
      const bool whole = need == sh.s_n;
      __syncthreads();
      if (whole) {
        // Every entry with this prefix is kept, so the n-th is the last of
        // them: the smallest key, and the largest ID among its ties. (The cut
        // must name it exactly: tokens outside the buffer can share the prefix.)
        Key smallest = ~Key(0);
        for (int i = tid; i < m; i += THREADS)
          if ((s_key[i] & known) == prefix && s_key[i] < smallest) smallest = s_key[i];
        smallest = block_all(smallest, [](Key x, Key y) { return x < y ? x : y; }, ~Key(0), sh.s_keys);
        int largest = -1;
        for (int i = tid; i < m; i += THREADS)
          if (s_key[i] == smallest && s_id[i] > largest) largest = s_id[i];
        largest = block_all(largest, [](int x, int y) { return x > y ? x : y; }, -1, sh.s_int);
        return {true, smallest, largest};
      }
    }
    unsigned id = 0, id_known = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) sh.s_hist[tid] = 0;
      __syncthreads();
      for (int i = tid; i < m; i += THREADS)
        if (s_key[i] == prefix && (unsigned(s_id[i]) & id_known) == id) atomicAdd(&sh.s_hist[(s_id[i] >> shift) & 255], 1u);
      __syncthreads();
      int count = tid < 256 ? int(sh.s_hist[tid]) : 0, total;  // ascending digits
      int cum = block_scan(count, sh.s_int, &total);
      if (tid < 256 && cum >= need && cum - count < need) { sh.s_a = tid; sh.s_b = cum - count; }
      __syncthreads();
      need -= sh.s_b;
      id |= unsigned(sh.s_a) << shift;
      id_known |= 255u << shift;
      __syncthreads();
    }
    return {true, prefix, int(id)};
  }
  // The ID of the need-th lowest-ID valid token with this key, among those `after` admits.
  __device__ __forceinline__ int tie_id(Key key, int need, After<Key> after) {
    // Count the ties per tile of 1,024 IDs in one pass (a warp's 32 groups lie
    // in one tile), then find the need-th inside its tile. Uses s_id as scratch.
    constexpr int TILE_GROUPS = 256;
    const int tiles = (view.groups + TILE_GROUPS - 1) / TILE_GROUPS;
    for (int i = tid; i < tiles; i += THREADS) s_id[i] = 0;
    __syncthreads();
    view.each([&](int g, const W* v) {
      const unsigned bits = group_mask(a, row, g, vocab);
      int c = 0;
      for (int j = 0; j < 4; ++j)
        c += ((bits >> j) & 1u) && v[j] > W(-INFINITY) && order_key(v[j]) == key && after.admits(key, 4 * g + j);
      c = __reduce_add_sync(0xffffffffu, c);
      if (lane() == 0 && c) atomicAdd(&s_id[g / TILE_GROUPS], c);
    });
    __syncthreads();
    if (tid == 0) {
      int before = 0, tile = 0;
      for (; tile < tiles - 1 && before + s_id[tile] < need; ++tile) before += s_id[tile];
      sh.s_a = tile; sh.s_b = before;
    }
    __syncthreads();
    const int t0 = sh.s_a * TILE_GROUPS * 4 + 2 * tid, before = sh.s_b;
    int is[2];
    for (int u = 0; u < 2; ++u) {
      const int t = t0 + u;
      is[u] = 0;
      if (t < vocab && allowed(a, row, t)) {
        const W v = view.value(t);
        is[u] = v != W(-INFINITY) && order_key(v) == key && after.admits(key, t);
      }
    }
    int total, incl = block_scan(is[0] + is[1], sh.s_int, &total);
    const int rank = before + incl - is[0] - is[1];  // ties before this thread's tokens
    if (is[0] && rank + 1 == need) sh.s_n = t0;
    if (is[1] && rank + is[0] + 1 == need) sh.s_n = t0 + 1;
    __syncthreads();
    const int id = sh.s_n;
    __syncthreads();
    return id;
  }
  // Appends the tokens flagged in `take` (bit j is token 4g + j) to s_key/s_id.
  // One atomic per warp; entries past GATHER are only counted.
  __device__ __forceinline__ void append(unsigned take, int g, const W* v) {
    const int c = __popc(take);
    if (!__any_sync(0xffffffffu, c)) return;
    int incl = c;
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane() >= o) incl += y;
    }
    int base = 0;
    if (lane() == 31) base = atomicAdd(&sh.s_n, incl);
    base = __shfl_sync(0xffffffffu, base, 31) + incl - c;
    for (int j = 0; j < 4; ++j) {
      if (!((take >> j) & 1u)) continue;
      if (base < GATHER) { s_key[base] = order_key(v[j]); s_id[base] = 4 * g + j; }
      ++base;
    }
  }
  // Histogram of the distance below ref of the valid tokens that `after`
  // admits and `keep` keeps (per-warp copies in s_id, then summed): thread b
  // < HIST_BINS receives bin b's count, every thread the count beyond them.
  __device__ __forceinline__ int bucket_counts(After<Key> after, Cut<Key> keep, W ref, int* far) {
    static_assert(WARPS * HIST_BINS <= GATHER, "histograms live in s_id");
    unsigned* hist = reinterpret_cast<unsigned*>(s_id);
    for (int i = tid; i < WARPS * HIST_BINS; i += THREADS) hist[i] = 0;
    __syncthreads();
    int beyond = 0;
    view.each([&](int g, const W* v) {
      const unsigned bits = group_mask(a, row, g, vocab);
      for (int j = 0; j < 4; ++j) {
        const Key k = order_key(v[j]);
        if (!(((bits >> j) & 1u) && v[j] > W(-INFINITY) && after.admits(k, 4 * g + j) && keep.keeps(k, 4 * g + j)))
          continue;
        const int b = hist_bin(ref - v[j]);
        if (b < HIST_BINS) atomicAdd(&hist[warp() * HIST_BINS + b], 1u); else ++beyond;
      }
    });
    __syncthreads();
    int here = 0;
    if (tid < HIST_BINS) for (int w = 0; w < WARPS; ++w) here += int(hist[w * HIST_BINS + tid]);
    *far = block_all(beyond, PlusOp{}, 0, sh.s_int);
    return here;
  }
  // Gathers those tokens in bins 0..c into s_key/s_id; returns their count.
  __device__ __forceinline__ int gather_bins(After<Key> after, Cut<Key> keep, W ref, int c) {
    if (tid == 0) sh.s_n = 0;
    __syncthreads();
    view.each([&](int g, const W* v) {
      const unsigned bits = group_mask(a, row, g, vocab);
      unsigned take = 0;
      for (int j = 0; j < 4; ++j) {
        const Key k = order_key(v[j]);
        take |= unsigned(((bits >> j) & 1u) && v[j] > W(-INFINITY) && hist_bin(ref - v[j]) <= c &&
                         after.admits(k, 4 * g + j) && keep.keeps(k, 4 * g + j)) << j;
      }
      append(take, g, v);
    });
    __syncthreads();
    return sh.s_n;
  }
  // The largest bin-aligned prefix of the tokens `after` admits and `keep`
  // keeps with at most `cap` tokens (or just the first bin, if that alone is
  // larger but fits the buffer), gathered into s_key/s_id: its count, or -1
  // if the first bin does not fit (or every token is 64 logits down).
  __device__ __forceinline__ int gather_prefix(After<Key> after, Cut<Key> keep, W ref, int cap) {
    int far;
    const int here = bucket_counts(after, keep, ref, &far);
    int near_total, cum = block_scan(here, sh.s_int, &near_total);
    int fits = tid < HIST_BINS && here > 0 && (cum <= cap || (cum == here && cum <= GATHER)) ? tid : -1;
    fits = block_all(fits, [](int x, int y) { return x > y ? x : y; }, -1, sh.s_int);
    if (fits < 0) return -1;
    return gather_bins(after, keep, ref, fits);
  }
  // Exact cut keeping the first n, in (logit desc, id asc) order, of the valid
  // tokens `after` admits; fewer than all of them. Their values are at most ref.
  // One pass gathers the candidates within a window below ref; if they fit
  // GATHER they are radix-selected in shared memory. *count is the number
  // gathered, or -1 if the cut was radix-selected over the whole row instead.
  __device__ __forceinline__ Cut<Key> select_next(After<Key> after, W ref, int n, int* count) {
    // Pass 1: the histogram; the bin holding the n-th token gives the window.
    int far;
    const Cut<Key> all{false, 0, 0};
    const int here = bucket_counts(after, all, ref, &far);
    int near_total, cum = block_scan(here, sh.s_int, &near_total);
    if (tid < HIST_BINS && cum >= n && cum - here < n) { sh.s_a = tid; sh.s_b = cum; }
    if (tid == 0 && near_total < n) { sh.s_a = HIST_BINS; sh.s_b = near_total + far; }
    __syncthreads();
    const int c = sh.s_a, m = sh.s_b;
    __syncthreads();
    if (c < HIST_BINS && m <= GATHER) {
      // Pass 2: gather the m tokens of bins 0..c; the n-th of them is the cut.
      *count = gather_bins(after, all, ref, c);
      return buffer_select(m, n);
    }
    // Ties or a degenerate row: radix-select the exact key over the whole row.
    *count = -1;
    int need = n, ties = 0;
    Key prefix = 0, known = 0;
    for (int shift = KEY_BITS - 8; shift >= 0; shift -= 8) {
      if (tid < 256) sh.s_hist[tid] = 0;
      __syncthreads();
      view.each([&](int g, const W* v) {
        const unsigned bits = group_mask(a, row, g, vocab);
        for (int j = 0; j < 4; ++j) {
          if (!((bits >> j) & 1u) || !(v[j] > W(-INFINITY))) continue;
          Key k = order_key(v[j]);
          if ((k & known) == prefix && after.admits(k, 4 * g + j)) atomicAdd(&sh.s_hist[(k >> shift) & 255], 1u);
        }
      });
      __syncthreads();
      int here = tid < 256 ? int(sh.s_hist[255 - tid]) : 0, sum;
      int run = block_scan(here, sh.s_int, &sum);
      if (tid < 256 && run >= need && run - here < need) { sh.s_a = 255 - tid; sh.s_b = run - here; sh.s_n = here; }
      __syncthreads();
      need -= sh.s_b;
      ties = sh.s_n;
      prefix |= Key(sh.s_a) << shift;
      known |= Key(255) << shift;
      __syncthreads();
    }
    return {true, prefix, need < ties ? tie_id(prefix, need, after) : INT_MAX};
  }
  // Moves the entries of s_key/s_id[0, m) that `keep` admits to the front; returns their count.
  __device__ __forceinline__ int compact(int m, Cut<Key> keep) {
    if (tid == 0) sh.s_n = 0;
    __syncthreads();
    for (int base = 0; base < m; base += THREADS) {  // read a stripe before any of it is overwritten
      const bool mine = base + tid < m && keep.keeps(s_key[base + tid], s_id[base + tid]);
      const Key k = mine ? s_key[base + tid] : 0;
      const int t = mine ? s_id[base + tid] : 0;
      __syncthreads();
      if (mine) { int i = atomicAdd(&sh.s_n, 1); s_key[i] = k; s_id[i] = t; }
      __syncthreads();
    }
    return sh.s_n;
  }
  // Gathers, from the row, the valid tokens that `after` admits and `keep` keeps.
  __device__ __forceinline__ int gather(After<Key> after, Cut<Key> keep) {
    if (tid == 0) sh.s_n = 0;
    __syncthreads();
    view.each([&](int g, const W* v) {
      const unsigned bits = group_mask(a, row, g, vocab);
      unsigned take = 0;
      for (int j = 0; j < 4; ++j) {
        Key k = order_key(v[j]);
        take |= unsigned(((bits >> j) & 1u) && v[j] > W(-INFINITY) && after.admits(k, 4 * g + j) &&
                         keep.keeps(k, 4 * g + j)) << j;
      }
      append(take, g, v);
    });
    __syncthreads();
    return sh.s_n;
  }
  // Sorts s_key/s_id[0, m) into (key desc, id asc) order.
  __device__ __forceinline__ void sort_buffer(int m) {
    if (sizeof(Key) == 4 && m > THREADS) {
      unsigned long long items[4];
      for (int u = 0; u < 4; ++u) {
        int i = 4 * tid + u;
        items[u] = i < m ? (static_cast<unsigned long long>(s_key[i]) << 32) | (0xffffffffu - unsigned(s_id[i])) : 0ull;
      }
      __syncthreads();
      Sort(sh.s_space.sort).SortDescending(items);
      __syncthreads();
      for (int u = 0; u < 4; ++u) {
        int i = 4 * tid + u;
        if (i < m) { s_key[i] = Key(items[u] >> 32); s_id[i] = int(0xffffffffu - unsigned(items[u])); }
      }
    } else {
      Key k[GATHER / THREADS];
      int t[GATHER / THREADS], rank[GATHER / THREADS];
      for (int u = 0; u < GATHER / THREADS; ++u) {
        int i = tid + u * THREADS;
        rank[u] = -1;
        if (i >= m) continue;
        k[u] = s_key[i]; t[u] = s_id[i]; rank[u] = 0;
        for (int j = 0; j < m; ++j) rank[u] += s_key[j] > k[u] || (s_key[j] == k[u] && s_id[j] < t[u]);
      }
      __syncthreads();
      for (int u = 0; u < GATHER / THREADS; ++u)
        if (rank[u] >= 0) { s_key[rank[u]] = k[u]; s_id[rank[u]] = t[u]; }
    }
    __syncthreads();
  }

  // ---- List: at most LIST_MAX tokens, processed in FP64. ----
  // Moves row_gather's candidates into s_key/s_id.
  __device__ __forceinline__ void load_candidates(int gathered) {
    const int* ids = candidates(a, row);
    for (int i = tid; i < gathered; i += THREADS) {
      s_id[i] = ids[i];
      s_key[i] = order_key(view.value(ids[i]));
    }
    __syncthreads();
  }
  // The kept tokens in s_key/s_id[0, n): returns n, or -1 when the row
  // belongs to the FULL kernel after all (it is then marked so).
  __device__ __forceinline__ int list_candidates() {
    const After<Key> none{false, {false, 0, 0}};
    int m = -1, n_list = n_keep;
    if (state.gather == TOP_P_GATHER) {
      // The largest bin prefix that fits a list (row_gather); FULL takes the row if it is missing.
      if (tid < 32) {
        int count;
        const int c = window_prefix(row_hist(a, row), TOP_P_LIST, &count);
        if (tid == 0) { sh.s_a = c; sh.s_b = count; }
      }
      __syncthreads();
      const int window = sh.s_a, expected = sh.s_b, gathered = int(arrival_counters(a)[2 * row + 1]);
      __syncthreads();
      if (window < 0 || gathered != expected) {
        if (tid == 0) row_states(a)[row].part = FULL_PART;
        return -1;
      }
      load_candidates(gathered);
      return gathered;
    }
    if (n_keep < valid) {
      if (tid < 32) {  // the window row_gather used, if any
        int count;
        const int c = window_of(row_hist(a, row), n_keep, &count);
        if (tid == 0) { sh.s_a = c; sh.s_b = count; }
      }
      __syncthreads();
      const int window = sh.s_a, expected = sh.s_b, gathered = int(arrival_counters(a)[2 * row + 1]);
      __syncthreads();
      if (state.gather && window >= 0 && gathered == expected) {
        // row_gather gathered every token of the bins holding the top k.
        load_candidates(gathered);
        m = gathered;
        cut = buffer_select(m, n_keep);
      } else {
        cut = select_next(none, maximum, n_keep, &m);
      }
    }
    if (m >= 0) compact(m, cut);
    else gather(none, cut);
    return n_list;
  }
  // Ranks s_key/s_id[0, n_list) into s_sorted.
  __device__ __forceinline__ void rank_list(int n_list) {
    for (int i = tid; i < n_list; i += THREADS) {
      int rank = 0;
      for (int j = 0; j < n_list; ++j) rank += s_key[j] > s_key[i] || (s_key[j] == s_key[i] && s_id[j] < s_id[i]);
      sh.s_sorted[rank] = s_id[i];
    }
    __syncthreads();
  }
  // Warp 0: the list's probabilities after min-p and top-p into s_p[0, s_n2),
  // or s_defer when the top-p boundary is not certain or may run past the
  // list. On this GPU FP64 is 1/64 of the FP32 rate, and a block-wide FP64
  // reduction would cost microseconds. Lane l owns entries l, l + 32, ...
  __device__ __forceinline__ void list_probabilities(int n_list) {
    double w[PER_LANE];
    int first_off = INT_MAX;
    for (int k = 0; k < PER_LANE; ++k) {
      const int i = lane() + 32 * k;
      w[k] = 0;
      if (i >= n_list) continue;
      w[k] = exp(x64(view.value(sh.s_sorted[i])));
      if (!(w[k] > 0 && w[k] >= min_p)) first_off = min(first_off, i);  // min-p and positivity cut a suffix
    }
    const int n1 = min(n_list, warp_all(first_off, MinOp{}));
    auto sum_to = [&](int n) {  // double-double sum of entries [0, n)
      DD acc{0, 0};
      for (int k = 0; k < PER_LANE; ++k)
        if (lane() + 32 * k < n) acc = dd_add(acc, DD{w[k], 0});
      return warp_all(acc, DDOp{});
    };
    int n2 = n1, defer = 0;
    if (top_p < 1) {
      // The target top_p * total. The total is the list's own sum, unless the
      // kept set runs past a top-p list: then it is row_hist's FP32 sum, an
      // interval, and the boundary must be certain under the whole interval.
      double lo_t, hi_t;
      if (state.gather == TOP_P_GATHER && n1 == n_list) {
        const DrawPart merged = merge_draws(a, row);
        const double total = merged.sum;
        const double bound = 1.01 * merged.err + 4.8e-7 * total + merged.pruned * SUM_PRUNE_W;
        lo_t = top_p * (total - bound) * (1 - 1e-15);
        hi_t = top_p * (total + bound) * (1 + 1e-15);
      } else {
        lo_t = hi_t = top_p * sum_to(n1).hi;
      }
      // Shortest prefix whose sum reaches the target: an inclusive scan per
      // round of 32 entries, carried from round to round.
      DD carry{0, 0};
      int reached_lo = INT_MAX, reached_hi = INT_MAX;
      for (int k = 0; k < PER_LANE && 32 * k < n1 && reached_hi == INT_MAX; ++k) {
        const int i = lane() + 32 * k;
        DD x{i < n1 ? w[k] : 0.0, 0};
        for (int o = 1; o < 32; o <<= 1) {
          const DD y{__shfl_up_sync(0xffffffffu, x.hi, o), __shfl_up_sync(0xffffffffu, x.lo, o)};
          if (lane() >= o) x = dd_add(y, x);
        }
        const DD prefix = dd_add(carry, x);
        reached_lo = min(reached_lo, warp_all(i < n1 && prefix.hi >= lo_t ? i : INT_MAX, MinOp{}));
        reached_hi = warp_all(i < n1 && prefix.hi >= hi_t ? i : INT_MAX, MinOp{});
        carry = {__shfl_sync(0xffffffffu, prefix.hi, 31), __shfl_sync(0xffffffffu, prefix.lo, 31)};
      }
      // Not certain, or the nucleus may run past the list: FULL decides.
      defer = state.gather == TOP_P_GATHER && (reached_hi == INT_MAX ? n1 == n_list : reached_lo != reached_hi);
      n2 = reached_hi == INT_MAX ? n1 : reached_hi + 1;
    }
    const double denominator = sum_to(n2).hi;
    for (int k = 0; k < PER_LANE; ++k) {
      const int i = lane() + 32 * k;
      if (i < n2) sh.s_p[i] = w[k] / denominator;
    }
    if (lane() == 0) { sh.s_n2 = n2; sh.s_defer = defer; }
  }
  // One draw over the list s_sorted/s_p[0, n2) (warp 0); excluded drops a token.
  __device__ __forceinline__ int list_draw(int n2, int sub, int excluded) {
    if (warp() == 0) {
      Best<double> b = no_best64();
      for (int i = lane(); i < n2; i += 32)
        if (sh.s_sorted[i] != excluded)
          b = best_of(b, Best<double>{log(sh.s_p[i]) + gumbel(a, req, row, sub, sh.s_sorted[i]), sh.s_sorted[i]});
      b = warp_all(b, Best64Op{});
      if (lane() == 0) sh.s_pick_id = b.id == INT_MAX ? -1 : b.id;
    }
    __syncthreads();
    const int id = sh.s_pick_id;
    __syncthreads();
    return id;
  }
  // Residual max(p - q, 0) over the list. Only q's normalizer is inexact;
  // it is recomputed if the winner is not certain.
  __device__ __forceinline__ int list_residual_q(int n2) {
    for (;;) {
      if (warp() == 0) {
        Pick<double> pick = no_pick<double>();
        for (int i = lane(); i < n2; i += 32) {
          const int t = sh.s_sorted[i];
          const W y = qview.value(t);
          const double wq = y == W(-INFINITY) ? 0.0 : exp(y64(y)), p = sh.s_p[i];
          const double rmid = p - wq / sq, rlo = p - wq / (sq - esq), rhi = p - wq / (sq + esq);
          if (rhi > 0) {
            const double g = gumbel(a, req, row, 2, t);
            pick = pick_of(pick, Pick<double>{rmid > 0 ? log(rmid) + g : -INFINITY, rlo > 0 ? log(rlo) + g : -INFINITY,
                                              log(rhi) + g, -INFINITY, t});
          }
        }
        pick = warp_all(pick, [](Pick<double> x, Pick<double> y) { return pick_of(x, y); });
        if (lane() == 0) {
          sh.s_settled = sq_exact || pick.id == INT_MAX || pick.lo > pick.others;
          sh.s_pick_id = pick.id == INT_MAX || !(pick.mid > -INFINITY) ? -1 : pick.id;
        }
      }
      __syncthreads();
      const bool settled = sh.s_settled;
      const int id = sh.s_pick_id;
      __syncthreads();
      if (settled) return id;
      q_sum64();
    }
  }
  __device__ __forceinline__ void list_row() {
    const int n_list = list_candidates();
    if (n_list < 0) return;
    rank_list(n_list);
    if (warp() == 0) list_probabilities(n_list);
    __syncthreads();
    if (sh.s_defer) {
      if (tid == 0) row_states(a)[row].part = FULL_PART;
      return;
    }
    const int n2 = sh.s_n2;
    if (!verify) { result.sample = list_draw(n2, 0, -1); finish(); return; }
    int at_draft = INT_MAX;
    for (int i = tid; i < n2; i += THREADS)
      if (sh.s_sorted[i] == draft_id) at_draft = i;
    at_draft = block_all(at_draft, MinOp{}, INT_MAX, sh.s_int);
    const double px = at_draft == INT_MAX ? 0.0 : sh.s_p[at_draft];
    bool accept;
    if (!full_q) accept = u_accept < fmin(1.0, px);
    else {
      double rlo = fmin(1.0, px / (wq_draft / (sq + esq))) * (1 - 1e-12);
      double rhi = fmin(1.0, px / (wq_draft / (sq - esq))) * (1 + 1e-12);
      const bool bounded = esq >= 0 && sq - esq > 0;
      if (px > 0 && !(bounded && (u_accept < rlo || u_accept >= rhi))) q_sum64();
      accept = u_accept < fmin(1.0, px / (wq_draft / sq));
    }
    result.p = accept ? 1.0 : 0.0;
    if (!accept) result.residual = full_q ? list_residual_q(n2) : list_draw(n2, 2, draft_id);
    finish();
  }

  // ---- Full row. ----
  // The top-k cut, for every row the FULL kernel finishes.
  __device__ __forceinline__ void keep_top_k() {
    if (n_keep < valid) {
      int m;
      cut = select_next(After<Key>{false, {false, 0, 0}}, maximum, n_keep, &m);
    }
  }
  // Whether token t (value v, bit j of bits) is in the kept set; *x is always set.
  __device__ __forceinline__ bool kept(W v, int t, unsigned bits, int j, float* x) const {
    *x = x32(v);
    const Key k = order_key(v);
    const bool in = ((bits >> j) & 1u) & (v > W(-INFINITY)) &
                    (!cut.on | (k > cut.key) | ((k == cut.key) & (t <= cut.id)));
    return in && weight_kept(v, *x);
  }
  // FP32 sum of kept weights; *err bounds its error.
  __device__ __forceinline__ double sum32(double* err) {
    float s = 0, c = 0, e = 0;
    int pruned = 0;
    view.each([&](int g, const W* v) {
      const unsigned bits = group_mask(a, row, g, vocab);
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // branch-free: a token not counted adds zero
        float x;
        const bool in = kept(v[j], 4 * g + j, bits, j, &x), counted = in & (x >= SUM_PRUNE);
        pruned += in & !counted;
        const float w = counted ? expf(x) : 0.f, d = w - c, t = s + d;
        c = (t - s) - d; s = t;
        e += counted ? w * (X_ERR * (1.f + fabsf(x))) : 0.f;  // x is -inf for tokens not kept
      }
    });
    double total = block_all(double(s) - double(c), PlusOp{}, 0.0, sh.s_double);
    double et = block_all(double(e), PlusOp{}, 0.0, sh.s_double);
    pruned = block_all(pruned, PlusOp{}, 0, sh.s_int);
    *err = 1.01 * et + 4.8e-7 * total + pruned * SUM_PRUNE_W;
    return total;
  }
  __device__ __forceinline__ double sum64() {
    fallback();
    DD acc{0, 0};
    view.each([&](int g, const W* v) {
      unsigned bits = group_mask(a, row, g, vocab);
      for (int j = 0; j < 4 && 4 * g + j < vocab; ++j) {
        float x;
        if (kept(v[j], 4 * g + j, bits, j, &x)) acc = dd_add(acc, DD{exp(x64(v[j])), 0});
      }
    });
    return block_all(acc, DDOp{}, DD{0, 0}, sh.s_dd).hi;
  }
  // Calls emit(t, v, y, word, mid, lo, hi) for every kept token of groups
  // [first, last) that can beat `floor` in the draw (sub, excluded,
  // residual_q); others are only bounded, in *skipped. With running set, a
  // thread also skips tokens that cannot beat its own best so far.
  template <class Emit>
  __device__ __forceinline__ void scan(int sub, int excluded, bool residual_q, int first, int last, float floor,
                                       bool running, Emit&& emit, Pick<float>* pick, float* skipped) {
    const float inv_sp = float(1.0 / sp), inv_sq = float(1.0 / sq);
    const float rel_sp = float(esp / sp) + 2.4e-7f, rel_sq = float(esq / sq) + 2.4e-7f;
    view.each([&](int g, const W* v) {
      W y[4];
      if (residual_q) qview.load(g, y);
      const unsigned bits = group_mask(a, row, g, vocab);
      unsigned live = 0;
      float xs[4], skip = *skipped;
      const float bar = (running ? fmaxf(floor, pick->lo) : floor) - PRUNE_MARGIN;
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // branch-free classification of the four tokens
        const bool in = kept(v[j], 4 * g + j, bits, j, &xs[j]);
        const bool candidate = in & (4 * g + j != excluded);
        const float ub = residual_q ? INFINITY : xs[j] + GUMBEL_MAX + SCORE_ERR * (1.f + fabsf(xs[j]) + GUMBEL_MAX);
        const bool alive = candidate & (ub >= bar);
        skip = fmaxf(skip, candidate & !alive ? ub : -INFINITY);
        live |= unsigned(alive) << j;
      }
      *skipped = skip;
      if (!live) return;
      Uint4 r4 = philox(seed, stream, position, uint32_t(sub), uint32_t(g));
      const uint32_t words[4] = {r4.x, r4.y, r4.z, r4.w};
      for (int j = 0; j < 4; ++j) {
        if (!((live >> j) & 1)) continue;
        if (!residual_q) {
          float b = gumbel_bound(words[j]);
          float ub = xs[j] + b + SCORE_ERR * (1.f + fabsf(xs[j]) + fabsf(b));
          if (ub < (running ? fmaxf(floor, pick->lo) : floor) - PRUNE_MARGIN) { *skipped = fmaxf(*skipped, ub); continue; }
          const float G = gumbel32(words[j]);
          float s = xs[j] + G, e = SCORE_ERR * (1.f + fabsf(xs[j]) + fabsf(G));
          emit(4 * g + j, v[j], W(0), words[j], s, s - e, s + e);
          continue;
        }
        const float G = gumbel32(words[j]);
        float p = expf(xs[j]) * inv_sp, ep = p * (X_ERR * (1.f + fabsf(xs[j])) + rel_sp) + 1e-37f;
        float q = 0, eq = 0;
        if (y[j] != W(-INFINITY)) {
          float yq = yq32(y[j]);
          q = expf(yq) * inv_sq;
          eq = q * (X_ERR * (1.f + fabsf(yq)) + rel_sq) + 1e-37f;
        }
        float r = p - q, er = ep + eq + 2.4e-7f * (p + q);
        if (!(r + er > 0)) continue;
        float eg = SCORE_ERR * (1.f + fabsf(G)), hi = __logf(r + er), lo = -INFINITY;
        hi += LOG_ERR * (1.f + fabsf(hi));
        if (r - er > 0) { lo = __logf(r - er); lo -= LOG_ERR * (1.f + fabsf(lo)); }
        emit(4 * g + j, v[j], y[j], words[j], r > 0 ? __logf(r) + G : -INFINITY, lo + G - eg, hi + G + eg);
      }
    }, first, last);
  }
  // The greedy token is always kept; its score bounds the winner's from below.
  __device__ __forceinline__ float floor_of(int sub, int excluded, bool residual_q) {
    if (tid == 0) {
      float floor = -INFINITY;
      if (!residual_q && excluded != greedy) {
        Uint4 r4 = philox(seed, stream, position, uint32_t(sub), uint32_t(greedy / 4));
        const uint32_t words[4] = {r4.x, r4.y, r4.z, r4.w};
        float g = gumbel32(words[greedy % 4]);
        floor = g - SCORE_ERR * (1.f + fabsf(g));
      }
      sh.s_floor = floor;
    }
    __syncthreads();
    return sh.s_floor;
  }
  // The exact winner when the FP32 one is not certain: rescore in FP64 every
  // token whose interval reaches the FP32 winner's lower bound.
  __device__ __forceinline__ int rescore(int sub, int excluded, bool residual_q, float threshold) {
    fallback();
    if (residual_q) {
      if (!sp_exact) { sp = sum64(); sp_exact = true; }
      if (!sq_exact) q_sum64();
    }
    Best<double> exact = no_best64();
    Pick<float> no_running_pick = no_pick<float>();
    float skipped = -INFINITY;
    scan(sub, excluded, residual_q, 0, groups, threshold, false, [&](int t, W v, W y, uint32_t word, float, float, float hi) {
      if (hi < threshold) return;
      double score = gumbel64(word);
      if (!residual_q) score += x64(v);
      else {
        double r = exp(x64(v)) / sp - (y == W(-INFINITY) ? 0.0 : exp(y64(y)) / sq);
        if (!(r > 0)) return;
        score += log(r);
      }
      exact = best_of(exact, Best<double>{score, t});
    }, &no_running_pick, &skipped);
    exact = block_all(exact, Best64Op{}, no_best64(), sh.s_best64);
    return exact.id == INT_MAX ? -1 : exact.id;
  }
  // The ratio test given the sum of kept weights sp +- esp; recomputes sp in FP64 if it decides nothing.
  __device__ __forceinline__ bool ratio_test() {
    W vx = view.value(draft_id);
    float xx;
    const double wx = allowed(a, row, draft_id) && kept(vx, draft_id, 1u, 0, &xx) ? exp(x64(vx)) : 0.0;
    if (!(wx > 0)) return false;
    double qlo = full_q ? wq_draft / (sq + esq) : 1.0, qhi = full_q ? wq_draft / (sq - esq) : 1.0;
    double rlo = fmin(1.0, wx / (sp + esp) / qhi) * (1 - 1e-12);
    double rhi = fmin(1.0, wx / (sp - esp) / qlo) * (1 + 1e-12);
    // Bounds that are not finite and positive never certify (fmin would hide a NaN).
    const bool bounded = esp >= 0 && sp - esp > 0 && (!full_q || (esq >= 0 && sq - esq > 0));
    if (bounded && u_accept < rlo) return true;
    if (bounded && u_accept >= rhi) return false;
    sp = sum64(); esp = 0; sp_exact = true;
    if (full_q) q_sum64();
    return u_accept < fmin(1.0, (wx / sp) / (full_q ? wq_draft / sq : 1.0));
  }
  // Top-p: scan the kept set in sorted chunks until the prefix reaches
  // top_p * total, and narrow the cut to that prefix (sp, esp: its sum).
  // Prefix sums are FP32 with an error bound; if the bound does not settle
  // the boundary, the total, the mass before the chunk and the chunk are
  // recomputed in FP64 and the scan continues exactly.
  __device__ __forceinline__ void top_p_cut() {
    constexpr int CHUNK = sizeof(Key) == 4 ? CHUNK32 : CHUNK64;
    double e0, s0 = sum32(&e0), target = 0;
    double above = 0, e_above = 0;
    DD above64{0, 0};
    bool exact = false;
    After<Key> after{false, {false, 0, 0}};
    W ref = maximum;
    for (int left = n_keep; left > 0;) {
      // A bin-aligned prefix of what is left, as large as the buffer allows;
      // if the first bin alone overflows, an exact cut of CHUNK tokens.
      // The first chunk is small: on real logits the nucleus is a handful of tokens.
      int n = gather_prefix(after, cut, ref, after.on ? GATHER : LIST_MAX);
      if (n < 0) {
        n = min(CHUNK, left);
        int m = -1;
        const Cut<Key> chunk = n < left ? select_next(after, ref, n, &m) : cut;
        if (m >= 0) compact(m, chunk);
        else gather(after, chunk);
      }
      sort_buffer(n);
      for (bool redo = true; redo;) {
        redo = false;
        // This thread's entries 4 tid + u, their weights, and the first entry
        // that min-p or positivity removes (the kept set ends there).
        double w[4], e[4], local = 0, local_e = 0;
        DD local64{0, 0};
        int first_out = INT_MAX;
        for (int u = 0; u < 4; ++u) {
          const int i = 4 * tid + u;
          w[u] = e[u] = 0;
          if (i >= n) continue;
          const W v = view.value(s_id[i]);
          const float x = x32(v);
          if (!weight_kept(v, x)) { first_out = min(first_out, i); continue; }
          if (exact) w[u] = exp(x64(v));
          else if (x < SUM_PRUNE) e[u] = SUM_PRUNE_W;
          else { w[u] = expf(x); e[u] = w[u] * (X_ERR * (1.f + fabsf(x))) * 1.01; }
        }
        first_out = block_all(first_out, MinOp{}, INT_MAX, sh.s_int);
        const int kept_n = min(n, first_out);
        for (int u = 0; u < 4; ++u) {
          if (4 * tid + u >= kept_n) w[u] = e[u] = 0;
          local += w[u]; local_e += e[u];
          local64 = dd_add(local64, DD{w[u], 0});
        }
        // Exclusive prefix over threads, then inclusive within the thread.
        double total, total_e, prefix = block_scan(local, sh.s_double, &total) - local;
        double prefix_e = block_scan(local_e, sh.s_double, &total_e) - local_e;
        if (exact) {  // double-double scan over thread totals
          sh.s_scan[tid] = local64;
          __syncthreads();
          for (int offset = 1; offset < THREADS; offset <<= 1) {
            DD other = tid >= offset ? sh.s_scan[tid - offset] : DD{0, 0};
            __syncthreads();
            if (tid >= offset) sh.s_scan[tid] = dd_add(sh.s_scan[tid], other);
            __syncthreads();
          }
          local64 = tid ? dd_add(above64, sh.s_scan[tid - 1]) : above64;
        }
        // First entry whose prefix certainly reaches the target, and first that may.
        int surely = INT_MAX, maybe = INT_MAX;
        double run = above + prefix, run_e = e_above + prefix_e;
        for (int u = 0; u < 4; ++u) {
          const int i = 4 * tid + u;
          if (i >= kept_n) break;
          if (exact) {
            local64 = dd_add(local64, DD{w[u], 0});
            if (local64.hi >= target) { surely = maybe = min(surely, i); }
            continue;
          }
          run += w[u]; run_e += e[u];
          const double slack = run_e + 1e-15 * run;
          const double lo_t = top_p * (s0 - e0) * (1 - 1e-15), hi_t = top_p * (s0 + e0) * (1 + 1e-15);
          if (run + slack >= lo_t) maybe = min(maybe, i);
          if (run - slack >= hi_t) surely = min(surely, i);
        }
        surely = block_all(surely, MinOp{}, INT_MAX, sh.s_int);
        maybe = block_all(maybe, MinOp{}, INT_MAX, sh.s_int);
        if (!exact && maybe != INT_MAX && maybe != surely) {
          // The boundary is not certain: switch to FP64 and redo this chunk.
          fallback();
          DD all{0, 0}, before{0, 0};
          view.each([&](int g, const W* v) {
            const unsigned bits = group_mask(a, row, g, vocab);
            for (int j = 0; j < 4; ++j) {
              float x;
              if (!kept(v[j], 4 * g + j, bits, j, &x)) continue;
              const DD wd{exp(x64(v[j])), 0};
              all = dd_add(all, wd);
              if (after.on && !after.admits(order_key(v[j]), 4 * g + j)) before = dd_add(before, wd);
            }
          });
          s0 = block_all(all, DDOp{}, DD{0, 0}, sh.s_dd).hi;
          above64 = block_all(before, DDOp{}, DD{0, 0}, sh.s_dd);
          target = top_p * s0;
          exact = redo = true;
          continue;
        }
        const int last = surely != INT_MAX ? surely : kept_n - 1;
        // The prefix through `last`: its sum (for the denominator) and its token.
        if (tid == (last >> 2)) {
          double r = above + prefix, re = e_above + prefix_e;
          DD r64 = exact ? (tid ? dd_add(above64, sh.s_scan[tid - 1]) : above64) : DD{0, 0};
          for (int u = 0; u <= (last & 3); ++u) { r += w[u]; re += e[u]; r64 = dd_add(r64, DD{w[u], 0}); }
          sh.s_double[0] = exact ? r64.hi : r;
          sh.s_double[1] = exact ? 0.0 : re;
          sh.s_dd[0] = r64;
          s_key[GATHER - 1] = s_key[last];
          s_id[GATHER - 1] = s_id[last];
        }
        __syncthreads();
        const double reached = sh.s_double[0], reached_e = sh.s_double[1];
        const DD reached64 = sh.s_dd[0];
        const Key last_key = s_key[GATHER - 1];
        const int last_id = s_id[GATHER - 1];
        __syncthreads();
        if (surely != INT_MAX || kept_n < n || left == n) {
          // Done: the boundary is here, or the kept set ends without reaching it.
          cut = kept_n > 0 ? Cut<Key>{true, last_key, last_id} : after.cut;
          sp = kept_n > 0 ? reached : (exact ? above64.hi : above);
          esp = exact ? 0.0 : (kept_n > 0 ? reached_e : e_above);
          sp_known = true;
          sp_exact = exact;
          left = 0;
        } else {
          above = reached; e_above = reached_e; above64 = reached64;
          after = {true, {true, last_key, last_id}};
          ref = view.value(last_id);
          left -= n;
        }
      }
    }
  }
  // Draws one token from the kept set (excluded drops a token) or, for
  // residual_q, from max(p - q, 0).
  __device__ __forceinline__ int draw(int sub, int excluded, bool residual_q) {
    const float lower = floor_of(sub, excluded, residual_q);
    Pick<float> pick = no_pick<float>();
    float skipped = -INFINITY;
    scan(sub, excluded, residual_q, 0, groups, lower, true, [&](int t, W, W, uint32_t, float mid, float lo, float hi) {
      pick = pick_of(pick, Pick<float>{mid, lo, hi, -INFINITY, t});
    }, &pick, &skipped);
    pick.others = fmaxf(pick.others, skipped);
    pick = block_all(pick, [](Pick<float> x, Pick<float> y) { return pick_of(x, y); }, no_pick<float>(), sh.s_pick32);
    if (pick.id == INT_MAX) return -1;
    if (pick.lo > pick.others) return pick.id;
    return rescore(sub, excluded, residual_q, pick.lo);
  }
  __device__ __forceinline__ void full_row() {
    if (top_p < 1) top_p_cut();
    if (verify) {
      if (!sp_known) sp = sum32(&esp);
      const bool accept = ratio_test();
      result.p = accept ? 1.0 : 0.0;
      if (accept) { finish(); return; }
    }
    if (!verify) result.sample = draw(0, -1, false);
    else result.residual = full_q ? draw(2, -1, true) : draw(2, draft_id, false);
    finish();
  }

  // ---- Sliced: temperature-only one-hot rows. row_scan has scanned each
  // slice for the draw (and, for a ratio test, the sum of weights); merge
  // the slices in order and decide. ----
  __device__ __forceinline__ void sliced_row() {
    const int sub = verify ? 2 : 0, excluded = verify ? draft_id : -1;
    // Warp 0 merges the parts in a fixed order.
    if (warp() == 0) {
      const DrawPart merged = merge_draws(a, row);
      if (lane() == 0) sh.s_merged = merged;
    }
    __syncthreads();
    const Pick<float> pick = sh.s_merged.pick;
    sp = sh.s_merged.sum;
    const double err = sh.s_merged.err;
    const int pruned = sh.s_merged.pruned;
    if (verify) {
      esp = 1.01 * err + 4.8e-7 * sp + pruned * SUM_PRUNE_W;
      const bool accept = ratio_test();
      result.p = accept ? 1.0 : 0.0;
      if (accept) { finish(); return; }
    }
    const int token = pick.id == INT_MAX ? -1 : pick.lo > pick.others ? pick.id : rescore(sub, excluded, false, pick.lo);
    if (verify) result.residual = token; else result.sample = token;
    finish();
  }
};

// One block per row. LIST_PART finishes list, greedy and failed rows;
// FULL_PART finishes full rows, the sliced rows row_scan could not settle,
// and greedy and failed rows when there are no lists. The arguments are
// __grid_constant__: the kernel passes them by reference to code that is not
// inlined, which would otherwise copy all 736 bytes to local memory in every
// thread before the early exit (2.4 us per launch).
template <DType D, bool VECTOR, int PART>
__global__ void __launch_bounds__(THREADS, PART == LIST_PART ? 2 : 1) fused_rows(const __grid_constant__ Arguments a) {
  __shared__ RowShared<D, PART> shared;
  const int row = blockIdx.y;
  if (*reinterpret_cast<const int*>(a.workspace)) return;  // malformed offsets
  {
    // Leave rows of the other kernel at once (before the request lookup and
    // parameter loads): most launches of this large kernel have nothing to do.
    // A stale state (for a request that failed validation) only falls through
    // to the checks below.
    const int part = row_states(a)[row].part;
    if (PART == LIST_PART ? part != LIST_PART : part != FULL_PART && part != SLICED_PART) return;
  }
  const int req = request_of(a, row);
  if (a.status[req * a.status_s] != OK || a.path[req * a.path_s] != FUSED) return;
  RowWork<D, VECTOR, PART> work(a, shared, row, req);
  const int part = work.state.part;
  if (PART == LIST_PART ? part != LIST_PART : part != FULL_PART && part != SLICED_PART) return;
  if (work.failed) { work.finish_failed(); return; }
  if (work.full_q) work.q_sum32();
  if (work.temperature == 0) { work.finish_greedy(); return; }
  work.begin_positive();
  if constexpr (PART == LIST_PART) {
    work.list_row();
  } else {
    work.keep_top_k();
    if (part == FULL_PART) work.full_row();
    else work.sliced_row();
  }
}

template <DType D, bool VECTOR> void launch_parts(const Arguments& a, cudaStream_t stream) {
  const dim3 sliced(slice_count(a.rows), unsigned(a.rows)), single(1, unsigned(a.rows));
  row_stats<D, VECTOR><<<sliced, SCAN_THREADS, 0, stream>>>(a);
  row_scan<D, VECTOR><<<sliced, SCAN_THREADS, 0, stream>>>(a);
  if (a.list_max > 0) {  // otherwise no row is a list row (row_stats)
    row_hist<D, VECTOR><<<sliced, SCAN_THREADS, 0, stream>>>(a);
    row_gather<D, VECTOR><<<sliced, SCAN_THREADS, 0, stream>>>(a);
    fused_rows<D, VECTOR, LIST_PART><<<single, THREADS, 0, stream>>>(a);
  }
  fused_rows<D, VECTOR, FULL_PART><<<single, THREADS, 0, stream>>>(a);
}
}  // namespace

// Launches every fused kernel for rows of dtype D. Instantiated once per dtype
// (fused_<dtype>.cu), so the four instantiations compile in parallel.
template <DType D> void launch_rows(const Arguments& a, cudaStream_t stream) {
  using T = typename In<D>::T;
  // Every row (target and, if present, draft) contiguous and aligned to a group of four.
  auto vectorizable = [&](const Matrix& m) {
    return !m.data || (m.s1 == 1 && a.vocab % 4 == 0 && (m.s0 * sizeof(T)) % (4 * sizeof(T)) == 0 &&
                       reinterpret_cast<uintptr_t>(m.data) % (4 * sizeof(T)) == 0);
  };
  if (vectorizable(a.target) && vectorizable(a.draft)) launch_parts<D, true>(a, stream);
  else launch_parts<D, false>(a, stream);
}
}  // namespace fused
}  // namespace exactsample
