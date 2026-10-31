# Independent CPU oracles

Two different checks live here and in `python/exactsample/reference.py`. Neither calls
CUDA, PyTorch, or a third-party sampling implementation.

The executable NumPy oracle uses float64 arithmetic, packed grammar masks, the
contract's filtering order, and independently implemented Philox. It handles
ragged batches with ordinary sampling, one-hot ratio/coupled verification, and
full-q ratio verification. It accepts strided NumPy views. Use an explicit
`verify_reference` call so a CPU result cannot be mistaken for a GPU result.

```python
import numpy as np
from exactsample import RequestParams, RngKey, verify_reference

result = verify_reference(
    np.array([[1., 2., 0.], [0., 1., 2.]]),
    np.array([0, 2]),                  # one draft row plus one bonus row
    np.array([[1]]),
    params=RequestParams(top_k=2),
    rng=RngKey(seed=7, stream_id=0, positions=np.array([20, 21])),
    mode="coupled",
)
assert result.status[0] == 0
print(result.tokens, result.accepted_len)
```

`reference/exact.py` uses only Python's `Fraction` and finite branch enumeration.
It does not import the NumPy oracle or share its filtering/sampling code. For
V <= 8 and K <= 3 it enumerates all rejection positions, replacement tokens,
accepted prefixes, and bonus tokens. `enumerate_ratio_joint` additionally
marginalizes every draft sequence drawn from q. Inputs are exact rational
probabilities, so it checks probability conservation and the p/q correction
law independently of floating-point arithmetic and RNG implementation.

```python
from fractions import Fraction as F
from reference.exact import enumerate_ratio_joint

law = enumerate_ratio_joint(
    target_probs=[[F(3, 4), F(1, 4)], [F(1, 2), F(1, 2)]],
    q_probs=[[F(1, 2), F(1, 2)]],
)
assert sum(law.values()) == 1
assert sum(mass for event, mass in law.items() if event.tokens[0] == 0) == F(3, 4)
```

Run the deterministic rational lattice and chain checks from the repository:

```sh
python reference/exact.py --max-vocab 4 --denominator 4 --max-draft 3
```

The default covers 1,620 cases: all p/q pairs on the denominator-4 lattice for
V=1..4, plus every one-hot draft chain of length 0..3 for one nonuniform
distribution per vocabulary size. Increase the flags to cover a larger lattice;
the number of cases grows combinatorially. This is an exhaustive check over
that finite lattice, not an exhaustive proof over all real distributions.

The chain enumerator assumes the provided per-position distributions are fixed.
Real autoregressive branches may have different distributions; unseen model
contexts cannot be recovered from a single saved logits matrix. Its exact
categorical laws are ideal continuous-uniform laws. Philox uniforms have a
finite 32-bit grid, whose quantization is evaluated separately by RNG and
statistical tests.

Passing these CPU checks does not establish CUDA compilation, device behavior,
sanitizer correctness, integration, or performance.
