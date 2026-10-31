"""exactsample semantic oracle and lazily loaded CUDA entry points."""
from .reference import distribution, proposal_distribution, sample_distribution, verify_reference
from .rng import gumbel, philox4x32, random_word, uniform
from .types import PathUsed, RequestParams, RngKey, Status, VerifyOutput

__version__ = "0.1.0"


def plan(*args, **kwargs):
    from .cuda import plan as implementation
    return implementation(*args, **kwargs)


def verify(*args, **kwargs):
    from .cuda import verify as implementation
    return implementation(*args, **kwargs)


__all__ = ["PathUsed", "RequestParams", "RngKey", "Status", "VerifyOutput",
           "distribution", "proposal_distribution", "sample_distribution",
           "verify_reference", "philox4x32", "random_word", "uniform", "gumbel",
           "plan", "verify"]
