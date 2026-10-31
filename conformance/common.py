import importlib.metadata


class Unavailable(RuntimeError):
    """A required runtime/dependency/version is missing, not a passing test."""


class UnsupportedCase(ValueError):
    """This adapter cannot provide an equal-semantics comparison for this case."""


def cuda_runtime():
    try:
        import torch
    except ImportError as exc:
        raise Unavailable('PyTorch is not installed') from exc
    if not torch.cuda.is_available():
        raise Unavailable(f'CUDA is unavailable (torch {torch.__version__})')
    return torch


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise Unavailable(f'{name} is not installed') from exc
