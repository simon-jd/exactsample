"""Exact finite-state speculative verification, using fractions only.

Unlike the executable float64 oracle, this module enumerates branches rather
than drawing random numbers and imports neither NumPy nor exactsample. It therefore
checks the mathematical distribution, not equality to a copied implementation.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product


@dataclass(frozen=True)
class ExactOutcome:
    tokens: tuple[int, ...]
    accepted_len: int


def _distribution(weights):
    weights = tuple(Fraction(str(x)) if isinstance(x, float) else Fraction(x) for x in weights)
    if not weights or any(x < 0 for x in weights) or sum(weights) != 1:
        raise ValueError("exact probabilities must be nonnegative and sum exactly to one")
    return weights


def conditional_outcomes(target_probs, draft_ids, q_probs=None, *, mode="ratio"):
    """Enumerate all token-prefix/accepted-length outcomes for fixed drafts.

    target_probs contains K+1 distributions, q_probs (if full) contains K.
    Fractions or integer/string rational values are preferred to floats.
    With no q_probs, proposals are point masses at the provided draft_ids.
    Coupled one-hot and ratio one-hot have the same outcome probabilities;
    they deliberately differ in their relation to a seeded ordinary sampler.
    """
    targets = tuple(_distribution(p) for p in target_probs)
    drafts = tuple(draft_ids)
    if not targets or len(targets) != len(drafts) + 1:
        raise ValueError("exactly K+1 target distributions are required")
    vocab = len(targets[0])
    if vocab > 8 or len(drafts) > 3:
        raise ValueError("small-vocabulary enumerator supports V <= 8 and K <= 3")
    if any(len(p) != vocab for p in targets):
        raise ValueError("vocabulary sizes must match")
    if mode not in ("ratio", "coupled") or (mode == "coupled" and q_probs is not None):
        raise ValueError("coupled supports only one-hot proposals")
    if q_probs is not None:
        proposals = tuple(_distribution(q) for q in q_probs)
        if len(proposals) != len(drafts) or any(len(q) != vocab for q in proposals):
            raise ValueError("q_probs must have shape [K,V]")
    else:
        proposals = tuple(tuple(Fraction(int(t == d)) for t in range(vocab)) for d in drafts)
    outcomes = defaultdict(Fraction)
    surviving = Fraction(1)
    for index, (draft, p, q) in enumerate(zip(drafts, targets, proposals)):
        if not isinstance(draft, int) or not 0 <= draft < vocab:
            raise ValueError("draft outside vocabulary")
        if q[draft] == 0:
            raise ValueError("DRAFT_ZERO_PROB")
        accepted = min(Fraction(1), p[draft] / q[draft])
        rejection = surviving * (1 - accepted)
        if rejection:
            residual = tuple(max(px - qx, 0) for px, qx in zip(p, q))
            denominator = sum(residual)
            if denominator == 0:
                raise AssertionError("positive rejection requires positive residual")
            for token, mass in enumerate(residual):
                if mass:
                    outcomes[ExactOutcome(drafts[:index] + (token,), index)] += rejection * mass / denominator
        surviving *= accepted
    for token, mass in enumerate(targets[-1]):
        if mass and surviving:
            outcomes[ExactOutcome(drafts + (token,), len(drafts))] += surviving * mass
    if sum(outcomes.values()) != 1:
        raise AssertionError("outcome mass was not conserved")
    return dict(outcomes)


def enumerate_ratio_joint(target_probs, q_probs):
    """Marginalize every draft sequence and every accept/reject/token branch.

    For fixed per-position p and q. Autoregressive callers must provide each
    context's distributions and invoke the same enumeration at that context;
    a matrix of logits is not a model for unseen autoregressive branches.
    """
    targets = tuple(_distribution(p) for p in target_probs)
    proposals = tuple(_distribution(q) for q in q_probs)
    if not targets or len(targets) != len(proposals) + 1:
        raise ValueError("target must include exactly one bonus distribution")
    vocab = len(targets[0])
    if vocab > 8 or len(proposals) > 3 or any(len(p) != vocab for p in targets + proposals):
        raise ValueError("require consistent V <= 8 and K <= 3")
    result = defaultdict(Fraction)
    for drafts in product(range(vocab), repeat=len(proposals)):
        mass = Fraction(1)
        for draft, q in zip(drafts, proposals):
            mass *= q[draft]
        if not mass:
            continue
        for outcome, conditional in conditional_outcomes(targets, drafts, proposals).items():
            result[outcome] += mass * conditional
    if sum(result.values()) != 1:
        raise AssertionError("joint outcome mass was not conserved")
    return dict(result)


def rational_simplex(vocab, denominator):
    """All categorical distributions on an integer probability lattice."""
    if vocab < 1 or denominator < 1:
        raise ValueError("vocab and denominator must be positive")
    def compositions(total, slots, prefix=()):
        if slots == 1:
            yield prefix + (total,)
        else:
            for mass in range(total + 1):
                yield from compositions(total - mass, slots - 1, prefix + (mass,))
    for counts in compositions(denominator, vocab):
        yield tuple(Fraction(n, denominator) for n in counts)


def verify_lattice(max_vocab=4, denominator=4, max_draft=3):
    """Exhaust all p/q lattice pairs and independently check correction law.

    In addition to first-token p, accepted mass must equal sum(min(p,q)).
    Chains cover every deterministic draft in a mixed rational distribution.
    """
    cases = 0
    for vocab in range(1, max_vocab + 1):
        grid = tuple(rational_simplex(vocab, denominator))
        for p, q in product(grid, repeat=2):
            outcomes = enumerate_ratio_joint((p, p), (q,))
            marginal = tuple(sum(m for o, m in outcomes.items() if o.tokens[0] == t) for t in range(vocab))
            if marginal != p:
                raise AssertionError(("wrong categorical law", p, q, marginal))
            accepted = sum(m for o, m in outcomes.items() if o.accepted_len == 1)
            if accepted != sum(min(px, qx) for px, qx in zip(p, q)):
                raise AssertionError("wrong acceptance law")
            cases += 1
        p = tuple(Fraction(i + 1, vocab * (vocab + 1) // 2) for i in range(vocab))
        for count in range(max_draft + 1):
            for drafts in product(range(vocab), repeat=count):
                ratio = conditional_outcomes((p,) * (count + 1), drafts)
                coupled = conditional_outcomes((p,) * (count + 1), drafts, mode="coupled")
                if ratio != coupled:
                    raise AssertionError("one-hot coupling changed the law")
                expected = Fraction(1)
                for draft in drafts:
                    expected *= p[draft]
                if sum(m for o, m in ratio.items() if o.accepted_len == count) != expected:
                    raise AssertionError("wrong chain acceptance law")
                cases += 1
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-vocab", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--denominator", type=int, default=4)
    parser.add_argument("--max-draft", type=int, default=3, choices=range(4))
    args = parser.parse_args()
    count = verify_lattice(args.max_vocab, args.denominator, args.max_draft)
    print(f"PASS: {count} exact rational cases (CPU mathematics only)")


if __name__ == "__main__":
    main()
