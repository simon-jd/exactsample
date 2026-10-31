"""Rational probability-space enumerator, independent of exactsample and NumPy.

This enumerates the *joint experiment*: draw draft from q, split its probability
mass into accept and reject rectangles, then distribute rejected mass. It is
not a same-key oracle and imports no production filtering or RNG code.
"""
from fractions import Fraction as F
from itertools import product


def compositions(total, width):
    if width == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, width - 1):
            yield (first,) + rest


def rational_distributions(width, denominator):
    for weights in compositions(denominator, width):
        yield tuple(F(weight, denominator) for weight in weights)


def one_step_joint(p, q):
    """Map (proposed, emitted, accepted) to its exact probability."""
    if len(p) != len(q) or sum(p) != 1 or sum(q) != 1:
        raise ValueError("p and q must be same-sized probability distributions")
    excess = tuple(max(a - b, F(0)) for a, b in zip(p, q))
    rejected = sum(excess)
    joint = {}
    for draft in range(len(q)):
        shared_mass = min(p[draft], q[draft])
        if shared_mass:
            joint[draft, draft, True] = shared_mass
        mass = q[draft] - shared_mass
        if mass:
            for token, weight in enumerate(excess):
                if weight:
                    joint[draft, token, False] = mass * weight / rejected
    return joint


def sequence_joint(target, proposals):
    """Enumerate emitted token tuple and accepted length for K <= 3.

    Targets here are fixed independent rows (not an autoregressive model). The
    last target row is the bonus. Integrating out all proposed draft sequences
    gives this exact sequential probability tree.
    """
    if len(target) != len(proposals) + 1:
        raise ValueError("one target row per draft plus one bonus required")
    active = {((), 0): F(1)}
    finished = {}
    for p, q in zip(target, proposals):
        following = {}
        for (prefix, length), mass in active.items():
            for (_, token, accepted), prob in one_step_joint(p, q).items():
                key = (prefix + (token,), length + int(accepted))
                bucket = following if accepted else finished
                bucket[key] = bucket.get(key, F(0)) + mass * prob
        active = following
    for (prefix, length), mass in active.items():
        for token, prob in enumerate(target[-1]):
            if prob:
                key = (prefix + (token,), length)
                finished[key] = finished.get(key, F(0)) + mass * prob
    return finished
