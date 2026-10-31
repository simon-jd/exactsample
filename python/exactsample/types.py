"""Public, framework-independent exactsample data model."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Status(IntEnum):
    OK = 0
    NO_VALID_TOKEN = 1
    INVALID_INPUT = 2
    DRAFT_ZERO_PROB = 3


class PathUsed(IntEnum):
    REFERENCE = 0
    FUSED = 1


@dataclass(frozen=True)
class RequestParams:
    """Each field is scalar or has one value per request; top_k=0 disables it."""

    temperature: Any = 1.0
    top_k: Any = 0
    top_p: Any = 1.0
    min_p: Any = 0.0
    draft_kind: Any = "one_hot"


@dataclass(frozen=True)
class RngKey:
    """seed:uint64 and stream_id:uint32 are scalar/per-request; positions:[rows]."""

    seed: Any
    stream_id: Any
    positions: Any


@dataclass
class VerifyOutput:
    tokens: Any
    accepted_len: Any
    status: Any
    path_used: Any
