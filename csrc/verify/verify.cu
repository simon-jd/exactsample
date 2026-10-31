#include "device.cuh"

namespace exactsample {
namespace {
__device__ int chosen_path(const Arguments& a, int) { return a.dispatch == 1 ? REFERENCE : FUSED; }
__device__ int metadata_status(const Arguments& a, int req, bool malformed) {
  if (malformed) return INVALID_INPUT;
  int64_t begin = at(a.cu_rows, req), end = at(a.cu_rows, req + 1);
  if (begin < 0 || end > a.rows || end <= begin || end - begin > a.max_draft + 1) return INVALID_INPUT;
  double t = at(a.temperature, req), p = at(a.top_p, req), m = at(a.min_p, req);
  int64_t k = at(a.top_k, req), kind = at(a.draft_kind, req);
  if (!isfinite(t) || t < 0 || !isfinite(p) || p <= 0 || p > 1 ||
      !isfinite(m) || m < 0 || m > 1 || k < 0 || k > a.vocab || kind < 0 || kind > 1)
    return INVALID_INPUT;
  if ((a.mode == 1 && kind == 1) || (kind == 1 && end - begin > 1 && !a.draft.data) ||
      at(a.stream, req) < 0 || uint64_t(at(a.stream, req)) > UINT32_MAX)
    return INVALID_INPUT;
  if (a.dispatch == 2 && chosen_path(a, req) != FUSED) return INVALID_INPUT;
  for (int64_t row = begin; row < end; ++row) {
    if (at(a.position, row) < 0 || uint64_t(at(a.position, row)) > UINT32_MAX) return INVALID_INPUT;
    if (row + 1 < end) {
      int64_t id = a.draft_ids[req * a.draft_s0 + (row - begin) * a.draft_s1];
      if (id < 0 || id >= a.vocab) return INVALID_INPUT;
    }
  }
  return OK;
}
// The global cu_rows check. Every block of initialize makes it (a few loads
// per request), so no separate launch has to publish it first.
__device__ bool offsets_malformed(const Arguments& a) {
  __shared__ int bad;
  if (threadIdx.x == 0) bad = at(a.cu_rows, 0) != 0 || at(a.cu_rows, a.requests) != a.rows;
  __syncthreads();
  for (int64_t i = threadIdx.x; i < a.requests; i += blockDim.x) {
    int64_t x = at(a.cu_rows, i), y = at(a.cu_rows, i + 1);
    if (x < 0 || y > a.rows || x > y) atomicExch(&bad, 1);
  }
  __syncthreads();
  return bad;
}
__global__ void initialize(Arguments a) {
  const bool malformed = offsets_malformed(a);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    reinterpret_cast<int*>(a.workspace)[0] = malformed;
    reinterpret_cast<unsigned*>(a.workspace)[1] = 0;  // fused FP64 fallback count
  }
  int req = blockIdx.x * blockDim.x + threadIdx.x;
  if (req >= a.requests) return;
  for (int i = 0; i <= a.max_draft; ++i) a.tokens[req * a.token_s0 + i * a.token_s1] = -1;
  a.accepted[req * a.accepted_s] = 0;
  a.path[req * a.path_s] = chosen_path(a, req);
  a.status[req * a.status_s] = metadata_status(a, req, malformed);
  if (a.path[req * a.path_s] == FUSED && a.status[req * a.status_s] == OK)
    for (int64_t row = at(a.cu_rows, req); row < at(a.cu_rows, req + 1); ++row)
      arrival_counters(a)[2 * row] = arrival_counters(a)[2 * row + 1] = 0;
}

// Ascending heap produces descending (score, inverse token-id) order.
__device__ bool worse(int x, int y, const double* values) {
  return values[x] < values[y] || (values[x] == values[y] && x > y);
}
__device__ void sift(int* ids, int root, int n, const double* values) {
  while (root * 2 + 1 < n) {
    int child = root * 2 + 1;
    if (child + 1 < n && worse(ids[child + 1], ids[child], values)) ++child;
    if (!worse(ids[child], ids[root], values)) break;
    int tmp = ids[root]; ids[root] = ids[child]; ids[child] = tmp;
    root = child;
  }
}
__device__ void sort_ids(int* ids, int n, const double* values) {
  for (int i = n / 2; i > 0; --i) sift(ids, i - 1, n, values);
  for (int i = n - 1; i > 0; --i) {
    int tmp = ids[0]; ids[0] = ids[i]; ids[i] = tmp;
    sift(ids, 0, i, values);
  }
}
__device__ int distribution(const Arguments& a, int req, int row, double* p, int* order, bool proposal) {
  Matrix source = proposal ? a.draft : a.target;
  double maxv = -INFINITY;
  int best = -1;
  for (int token = 0; token < a.vocab; ++token) {
    double v = get(source, row, token);
    if (isnan(v) || v == INFINITY) return INVALID_INPUT;
    if (!proposal && !allowed(a, row, token)) v = -INFINITY;
    p[token] = v;
    order[token] = token;
    if (v > maxv) { maxv = v; best = token; }
  }
  if (best < 0) return NO_VALID_TOKEN;
  double temperature = proposal ? 1.0 : at(a.temperature, req);
  if (temperature == 0) {
    for (int token = 0; token < a.vocab; ++token) p[token] = token == best ? 1.0 : 0.0;
    return OK;
  }
  // Sort raw scores, retaining exact token-id tie breaking even if exp underflows.
  if (!proposal) sort_ids(order, int(a.vocab), p);
  double min_p = proposal ? 0.0 : at(a.min_p, req);
  for (int token = 0; token < a.vocab; ++token) {
    double w = exp((p[token] - maxv) / temperature);
    p[token] = w >= min_p ? w : 0.0;
  }
  if (!proposal) {
    int64_t k = at(a.top_k, req);
    if (k > 0) for (int i = int(k); i < a.vocab; ++i) p[order[i]] = 0;
  }
  double sum = 0;
  for (int token = 0; token < a.vocab; ++token) sum += p[token];
  if (!(sum > 0)) return NO_VALID_TOKEN;
  if (!proposal && at(a.top_p, req) < 1) {
    double cumulative = 0, threshold = at(a.top_p, req) * sum;
    bool reached = false;
    for (int i = 0; i < a.vocab; ++i) {
      int token = order[i];
      if (reached) p[token] = 0;
      else { cumulative += p[token]; if (cumulative >= threshold) reached = true; }
    }
    sum = 0;
    for (int token = 0; token < a.vocab; ++token) sum += p[token];
  }
  for (int token = 0; token < a.vocab; ++token) p[token] /= sum;
  return OK;
}
__device__ int sample(const Arguments& a, int req, int row, const double* p,
                      const double* q, int excluded, int sub) {
  int best = -1;
  double score = -INFINITY;
  for (int token = 0; token < a.vocab; ++token) {
    double weight = q ? fmax(p[token] - q[token], 0.0) : (token == excluded ? 0.0 : p[token]);
    if (weight <= 0) continue;
    double s = log(weight) + gumbel(a, req, row, sub, token);
    if (best < 0 || better(s, token, score, best)) { score = s; best = token; }
  }
  return best;
}
__global__ void reference_rows(Arguments a) {
  int req = blockIdx.x;
  if (threadIdx.x || a.status[req * a.status_s] != OK || chosen_path(a, req) != REFERENCE) return;
  auto* results = reinterpret_cast<RowResult*>(a.workspace + 8);
  double* probs = reinterpret_cast<double*>(results + a.rows);
  double* qs = probs + a.rows * a.vocab;
  int* orders = reinterpret_cast<int*>(qs + a.rows * a.vocab);
  int begin = int(at(a.cu_rows, req)), end = int(at(a.cu_rows, req + 1));
  for (int row = begin; row < end; ++row) {
    RowResult r{0, 1, -1, -1, OK, OK};
    double* p = probs + int64_t(row) * a.vocab;
    double* q = qs + int64_t(row) * a.vocab;
    int* order = orders + int64_t(row) * a.vocab;
    r.status = distribution(a, req, row, p, order, false);
    r.target_status = r.status;
    bool full = at(a.draft_kind, req) == 1 && row + 1 < end;
    int qstatus = full ? distribution(a, req, row, q, order, true) : OK;
    if (r.status == OK && qstatus != OK) r.status = qstatus;
    if (r.status == OK) {
      int id = row + 1 < end ? int(a.draft_ids[req * a.draft_s0 + (row - begin) * a.draft_s1]) : -1;
      r.p = id < 0 ? 0.0 : p[id];
      r.q = full ? q[id] : 1.0;
      if (full && r.q == 0) r.status = DRAFT_ZERO_PROB;
      r.sample = sample(a, req, row, p, nullptr, -1, 0);
      if (id >= 0) r.residual = sample(a, req, row, p, full ? q : nullptr, full ? -1 : id, 2);
    }
    results[row] = r;
  }
}

// One warp per request. Lane l takes the request's rows begin + l, begin + l +
// 32, ..., so their results are read and their acceptance tests made in
// parallel; ballots find the first failed row and the first rejection.
__global__ void finalize(Arguments a) {
  const int64_t req = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (req >= a.requests || a.status[req * a.status_s] != OK) return;  // the whole warp
  const auto* results = reinterpret_cast<const RowResult*>(a.workspace + 8);
  const int begin = int(at(a.cu_rows, req)), end = int(at(a.cu_rows, req + 1));
  int32_t& status = a.status[req * a.status_s];
  int64_t* tokens = a.tokens + req * a.token_s0;
  // The first row whose target failed; failing that, the first row that failed.
  for (int pass = 0; pass < 2; ++pass)
    for (int base = begin; base < end; base += 32) {
      const int row = base + lane;
      const int s = row < end ? (pass == 0 ? results[row].target_status : results[row].status) : OK;
      const unsigned bad = __ballot_sync(0xffffffffu, s != OK);
      if (bad) {
        const int first = __shfl_sync(0xffffffffu, s, __ffs(bad) - 1);
        if (lane == 0) status = first;
        return;
      }
    }
  // The first rejected draft, or the bonus row when every draft is accepted.
  const uint64_t seed = uint64_t(at(a.seed, req));
  const uint32_t stream = uint32_t(at(a.stream, req));
  int stop = end - 1;
  for (int base = begin; base < end - 1; base += 32) {
    const int row = base + lane;
    bool reject = false;
    if (row < end - 1) {
      const RowResult r = results[row];
      const int draft = int(a.draft_ids[req * a.draft_s0 + (row - begin) * a.draft_s1]);
      reject = !(a.mode == 1 ? r.sample == draft
                 : uniform(seed, stream, uint32_t(at(a.position, row)), 1, 0) < fmin(1.0, r.p / r.q));
    }
    const unsigned rejected = __ballot_sync(0xffffffffu, reject);
    if (rejected) { stop = base + __ffs(rejected) - 1; break; }
  }
  const RowResult r = results[stop];
  const int token = stop == end - 1 || a.mode == 1 ? r.sample : r.residual, accepted = stop - begin;
  if (stop < end - 1 && token < 0) {
    if (lane == 0) status = INVALID_INPUT;
    for (int i = lane; i <= a.max_draft; i += 32) tokens[i * a.token_s1] = -1;
    return;
  }
  for (int i = lane; i < accepted; i += 32) tokens[i * a.token_s1] = a.draft_ids[req * a.draft_s0 + i * a.draft_s1];
  if (lane == 0) {
    tokens[accepted * a.token_s1] = token;
    a.accepted[req * a.accepted_s] = accepted;
  }
}
}  // namespace

cudaError_t launch(const Arguments& a, cudaStream_t stream) {
  if (a.requests == 0) return cudaSuccess;
  initialize<<<(a.requests + 127) / 128, 128, 0, stream>>>(a);
  if (a.dispatch == 1) reference_rows<<<a.requests, 1, 0, stream>>>(a);
  cudaError_t fused = a.dispatch == 1 ? cudaSuccess : launch_fused(a, stream);
  if (fused != cudaSuccess) return fused;
  finalize<<<(a.requests + 3) / 4, 128, 0, stream>>>(a);  // a warp per request
  return cudaGetLastError();
}
}  // namespace exactsample
