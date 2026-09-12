"""Valkyr Python bindings — pure Zig + Vulkan LLM inference, torch-like API."""

from .tensor import Tensor
from .model import Model
from . import _native

__version__ = "0.1.0"
__all__ = ["Tensor", "Model", "tensor", "cat", "native_version"]


def tensor(data, dtype=None, device="vulkan"):
    """Create a Tensor from array-like data.

    Parameters
    ----------
    data : array-like
        Numpy array, list, or Tensor.
    dtype : str or numpy dtype, optional
        e.g. ``"float32"``, ``"f16"``, ``"int32"``.
    device : str, optional
        ``"vulkan"`` (default) or ``"cpu"``.
    """
    return Tensor(data, dtype=dtype, device=device)


def cat(tensors, dim=0):
    """Concatenate a sequence of Tensors along a dimension."""
    import numpy as _np
    arrs = [t._data if isinstance(t, Tensor) else _np.asarray(t) for t in tensors]
    return Tensor(_np.concatenate(arrs, axis=dim))


def native_version():
    """Return the valkyr_core shared library version tuple."""
    v = _native.vkr_version()
    return (v >> 16, (v >> 8) & 0xFF, v & 0xFF)