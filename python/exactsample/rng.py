"""Pure Python Random123 Philox4x32-10 with the exactsample v1 counter layout.

There is no mutable generator state. Batch indices never enter the address.
"""
import math
import operator

_U32 = (1 << 32) - 1
_U64 = (1 << 64) - 1


def _uint(value, maximum, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, bool) or not 0 <= result <= maximum:
        raise ValueError(f"{name} outside [0, {maximum}]")
    return result


def philox4x32(counter, key):
    """Return four uint32 words for four counter words and two key words.

    All arithmetic uses Python integers, with explicit modulo-2**32 masking.
    This implementation is independent of the C++/CUDA implementation.
    """
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox requires 4 counter and 2 key words")
    c0, c1, c2, c3 = [_uint(x, _U32, "counter") for x in counter]
    k0, k1 = [_uint(x, _U32, "key") for x in key]
    for _ in range(10):
        p0 = c0 * 0xD2511F53
        p1 = c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & _U32,
            p1 & _U32,
            ((p0 >> 32) ^ c3 ^ k1) & _U32,
            p0 & _U32,
        )
        k0 = (k0 + 0x9E3779B9) & _U32
        k1 = (k1 + 0xBB67AE85) & _U32
    return c0, c1, c2, c3


def random_word(seed, stream_id, position, substream=0, token=0):
    seed = _uint(seed, _U64, "seed")
    stream_id = _uint(stream_id, _U32, "stream_id")
    position = _uint(position, _U32, "position")
    substream = _uint(substream, _U32, "substream")
    token = _uint(token, (1 << 34) - 1, "token")
    words = philox4x32(
        (stream_id, position, substream, token // 4),
        (seed & _U32, seed >> 32),
    )
    return words[token % 4]


def uniform(seed, stream_id, position, substream=0, token=0):
    """An exactly representable, open-interval double in (0, 1)."""
    return (random_word(seed, stream_id, position, substream, token) + 0.5) / (1 << 32)


def gumbel(seed, stream_id, position, substream=0, token=0):
    return -math.log(-math.log(uniform(seed, stream_id, position, substream, token)))
