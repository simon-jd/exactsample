"""Independent NumPy float64 semantic oracle (never imports torch or CUDA)."""
import math
import operator

import numpy as np

from .rng import gumbel, uniform
from .types import PathUsed, RequestParams, RngKey, Status, VerifyOutput


def _scalar_integer(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("booleans are not integer parameters")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError("expected integer parameter") from exc


def _packed_mask(mask, vocab):
    if mask is None:
        return np.ones(vocab, dtype=bool)
    words = np.asarray(mask)
    if words.shape != ((vocab + 31) // 32,) or words.dtype.kind not in "iu" or words.dtype.itemsize != 4:
        raise ValueError("mask must contain ceil(vocab/32) int32 or uint32 words")
    words = words.astype(np.uint32, copy=False)
    return np.array([bool((int(words[t // 32]) >> (t % 32)) & 1) for t in range(vocab)])


def distribution(logits, *, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, mask=None):
    """Return (float64 probabilities, Status) for one target row.

    Processing order is mask, temperature, min-p, exact stable top-k, minimal
    top-p prefix, normalization. NaN/+inf anywhere in the row is invalid,
    including masked entries. Negative infinity has zero mass.
    """
    values = np.asarray(logits)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "fiu":
        raise ValueError("logits must be a nonempty real vector")
    values = values.astype(np.float64, copy=True)
    probs = np.zeros(values.size, dtype=np.float64)
    allowed = _packed_mask(mask, values.size)
    try:
        k = _scalar_integer(top_k)
        temperature, top_p, min_p = float(temperature), float(top_p), float(min_p)
        valid = (math.isfinite(temperature) and temperature >= 0 and 0 <= k <= values.size
                 and math.isfinite(top_p) and 0 < top_p <= 1
                 and math.isfinite(min_p) and 0 <= min_p <= 1)
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid or np.isnan(values).any() or np.isposinf(values).any():
        return probs, Status.INVALID_INPUT
    active = allowed & np.isfinite(values)
    if not active.any():
        return probs, Status.NO_VALID_TOKEN
    if temperature == 0:
        probs[int(np.argmax(np.where(active, values, -np.inf)))] = 1.0
        return probs, Status.OK

    # Subtract before division: avoids +inf for tiny positive temperatures.
    maximum = np.max(values[active])
    with np.errstate(over="ignore", under="ignore"):
        shifted = (values - maximum) / temperature
        weights = np.where(active, np.exp(shifted), 0.0)
    active &= weights >= min_p
    active &= weights > 0.0
    # NumPy's stable sort preserves ascending token ID for equal values.
    ordered = np.flatnonzero(active)
    ordered = ordered[np.argsort(-values[ordered], kind="stable")]
    if k:
        ordered = ordered[:k]
    selected_weights = weights[ordered]
    denominator = math.fsum(float(x) for x in selected_weights)
    if top_p < 1:
        target = top_p * denominator
        prefix = []
        count = 0
        for weight in selected_weights:
            prefix.append(float(weight))
            count += 1
            if math.fsum(prefix) >= target:
                break
        ordered = ordered[:count]
        denominator = math.fsum(float(weights[x]) for x in ordered)
    probs[ordered] = weights[ordered] / denominator
    return probs, Status.OK


def proposal_distribution(logits):
    """Normalize actual q log weights, without applying target processing."""
    return distribution(logits)


def sample_distribution(probs, *, seed, stream_id, position, substream=0):
    """Sample categorical weights using per-token, addressable Gumbel noise."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0 or not np.isfinite(probs).all() or (probs < 0).any() or not (probs > 0).any():
        raise ValueError("probabilities must be nonnegative, finite, and have positive mass")
    best_token, best_value = -1, -math.inf
    for token in np.flatnonzero(probs > 0):
        score = math.log(float(probs[token])) + gumbel(seed, stream_id, position, substream, int(token))
        if score > best_value:
            best_token, best_value = int(token), score
    return best_token


def _per_request(value, count, name):
    # Object conversion preserves Python uint64 seeds. NumPy otherwise infers
    # float64 for e.g. [2**64-1, 0], irreversibly rounding the request identity.
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return [arr.item()] * count
    if arr.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape [{count}]")
    return list(arr)


def _bounded_int(value, bits):
    result = _scalar_integer(value)
    if not 0 <= result < (1 << bits):
        raise ValueError(f"integer outside uint{bits}")
    return result


def verify_reference(target_logits, cu_rows, draft_ids, *, params=None, rng: RngKey,
                     draft_logits=None, mask=None, mode="ratio"):
    """Verify a ragged batch against the exactsample v1 contract.

    Each request has D+1 target rows (D drafts and a bonus), and draft_ids
    has shape [requests, max_draft]. Full-q draft_logits has the same shape
    as target_logits and describes the actual proposal distribution, with
    all proposal filtering already encoded as -inf. Its bonus rows are
    ignored. All included target/draft rows are validated before any tokens
    can be returned for a request. Structural errors raise ValueError;
    request-local errors yield a status, -1 tokens and zero accepted_len.
    """
    if mode not in ("ratio", "coupled"):
        raise ValueError("mode must be 'ratio' or 'coupled'")
    params = RequestParams() if params is None else params
    if not isinstance(params, RequestParams) or not isinstance(rng, RngKey):
        raise TypeError("params/RNG must be RequestParams/RngKey")
    logits, offsets, drafts = map(np.asarray, (target_logits, cu_rows, draft_ids))
    if logits.ndim != 2 or logits.shape[1] == 0 or logits.dtype.kind not in "fiu":
        raise ValueError("target_logits must be a real [rows, vocab] array")
    rows, vocab = logits.shape
    if offsets.ndim != 1 or offsets.size == 0 or offsets.dtype.kind not in "iu":
        raise ValueError("cu_rows must be an integer [requests+1] vector")
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows or any(int(b) < int(a) for a, b in zip(offsets[:-1], offsets[1:])):
        raise ValueError("cu_rows must be monotone, start at zero, and end at rows")
    requests = offsets.size - 1
    if drafts.ndim != 2 or drafts.shape[0] != requests or drafts.dtype.kind not in "iu":
        raise ValueError("draft_ids must be an integer [requests, max_draft] array")
    max_draft = drafts.shape[1]
    if any(int(b) - int(a) > max_draft + 1 for a, b in zip(offsets[:-1], offsets[1:])):
        raise ValueError("ragged segment exceeds draft_ids capacity")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (rows, (vocab + 31) // 32) or mask.dtype.kind not in "iu" or mask.dtype.itemsize != 4:
            raise ValueError("mask must be packed int32 [rows, ceil(vocab/32)]")
    if draft_logits is not None:
        draft_logits = np.asarray(draft_logits)
        if draft_logits.shape != logits.shape or draft_logits.dtype.kind not in "fiu":
            raise ValueError("draft_logits must have target_logits shape and real dtype")
    positions = np.asarray(rng.positions)
    if positions.shape != (rows,):
        raise ValueError("rng.positions must have shape [rows]")
    fields = {name: _per_request(getattr(params, name), requests, name)
              for name in ("temperature", "top_k", "top_p", "min_p", "draft_kind")}
    seeds = _per_request(rng.seed, requests, "seed")
    streams = _per_request(rng.stream_id, requests, "stream_id")
    output = VerifyOutput(
        np.full((requests, max_draft + 1), -1, dtype=np.int64),
        np.zeros(requests, dtype=np.int32),
        np.full(requests, int(Status.OK), dtype=np.int32),
        np.full(requests, int(PathUsed.REFERENCE), dtype=np.int32),
    )
    for request in range(requests):
        start, end = int(offsets[request]), int(offsets[request + 1])
        count = end - start - 1
        kind = fields["draft_kind"][request]
        if count < 0 or kind not in ("one_hot", "full") or (kind == "full" and mode == "coupled"):
            output.status[request] = Status.INVALID_INPUT
            continue
        if count and kind == "full" and draft_logits is None:
            output.status[request] = Status.INVALID_INPUT
            continue
        try:
            seed = _bounded_int(seeds[request], 64)
            stream = _bounded_int(streams[request], 32)
            row_positions = [_bounded_int(pos, 32) for pos in positions[start:end]]
        except (ValueError, TypeError, OverflowError):
            output.status[request] = Status.INVALID_INPUT
            continue
        if any(not 0 <= int(drafts[request, index]) < vocab for index in range(count)):
            output.status[request] = Status.INVALID_INPUT
            continue
        row_probs, proposal_probs = [], []
        local_status = Status.OK
        for row in range(start, end):
            probabilities, local_status = distribution(
                logits[row], mask=None if mask is None else mask[row],
                **{name: fields[name][request] for name in ("temperature", "top_k", "top_p", "min_p")})
            if local_status != Status.OK:
                break
            row_probs.append(probabilities)
        if local_status == Status.OK:
            for index in range(count):
                token = int(drafts[request, index])
                if kind == "full":
                    proposal, local_status = proposal_distribution(draft_logits[start + index])
                    if local_status != Status.OK:
                        break
                    if proposal[token] == 0:
                        local_status = Status.DRAFT_ZERO_PROB
                        break
                    proposal_probs.append(proposal)
        if local_status != Status.OK:
            output.status[request] = local_status
            continue
        accepted = 0
        for index in range(count):
            p = row_probs[index]
            token = int(drafts[request, index])
            address = dict(seed=seed, stream_id=stream, position=row_positions[index])
            if mode == "coupled":
                sampled = sample_distribution(p, **address, substream=0)
                accept = sampled == token
            else:
                q_token = 1.0 if kind == "one_hot" else float(proposal_probs[index][token])
                accept = uniform(seed, stream, row_positions[index], 1, 0) < min(1.0, float(p[token]) / q_token)
                if not accept:
                    if kind == "one_hot":
                        residual = p.copy()
                        residual[token] = 0.0
                    else:
                        residual = np.maximum(p - proposal_probs[index], 0.0)
                    # This branch mathematically has positive residual mass.
                    # An empty residual is numeric invalidity, never token 0.
                    if not (residual > 0).any():
                        output.status[request] = Status.INVALID_INPUT
                        output.tokens[request] = -1
                        break
                    sampled = sample_distribution(residual, **address, substream=2)
            if not accept:
                output.tokens[request, index] = sampled
                break
            output.tokens[request, index] = token
            accepted += 1
        else:
            output.tokens[request, count] = sample_distribution(
                row_probs[count], seed=seed, stream_id=stream,
                position=row_positions[count], substream=0)
        if output.status[request] == Status.OK:
            output.accepted_len[request] = accepted
    return output
