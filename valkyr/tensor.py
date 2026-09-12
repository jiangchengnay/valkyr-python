"""Tensor with a torch-like surface. v0.1 computes on CPU (numpy);
payload ops will move to Vulkan kernels in a later iteration."""
import numpy as _np

_DTYPE_MAP = {
    "float32": _np.float32,
    "f32": _np.float32,
    "float16": _np.float16,
    "f16": _np.float16,
    "int32": _np.int32,
    "i32": _np.int32,
    "int64": _np.int64,
    "i64": _np.int64,
    "uint8": _np.uint8,
    "u8": _np.uint8,
    "bool": _np.bool_,
}


def _resolve_dtype(dtype):
    if isinstance(dtype, str):
        try:
            return _DTYPE_MAP[dtype]
        except KeyError:
            raise ValueError(f"unknown dtype {dtype!r}; choose from {set(_DTYPE_MAP)}")
    if isinstance(dtype, type):
        return dtype
    return _np.dtype(dtype)


class Tensor:
    """A multi-dimensional array with a PyTorch-like surface.

    v0.1 stores data on CPU (numpy).  GPU inference goes through
    ``Model``, not through Tensor ops; the ``device`` parameter is
    accepted for API compatibility but all arithmetic happens in
    numpy on the host.
    """

    def __init__(self, data, dtype=None, device="vulkan"):
        if isinstance(data, Tensor):
            data = data._data
        if isinstance(data, _np.ndarray):
            arr = data
        else:
            arr = _np.asarray(data, dtype=_np.float32)
        if dtype is not None:
            arr = arr.astype(_resolve_dtype(dtype))
        if device not in ("cpu", "vulkan", "gpu"):
            raise ValueError(f"unsupported device {device!r}; use 'cpu' or 'vulkan'")
        self._data = arr
        self._device = "vulkan" if device in ("vulkan", "gpu") else "cpu"

    @property
    def shape(self):
        return tuple(self._data.shape)

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def numel(self):
        return self._data.size

    @property
    def dtype(self):
        return str(self._data.dtype)

    @property
    def device(self):
        return self._device

    def to(self, device):
        """Return a copy on the target device (v0.1: labels only, still CPU)."""
        if device not in ("cpu", "vulkan", "gpu"):
            raise ValueError(f"unsupported device {device!r}")
        t = Tensor(self._data.copy())
        t._device = "vulkan" if device in ("vulkan", "gpu") else "cpu"
        return t

    def numpy(self):
        """Return a CPU numpy copy of the data."""
        return self._data.copy()

    def __array__(self, dtype=None):
        return self._data.astype(dtype) if dtype else self._data

    # ── Basic ops ─────────────────────────────────────────────────

    def add(self, other):
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data + other)

    def matmul(self, other):
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(_np.matmul(self._data, other))

    def relu(self):
        return Tensor(_np.maximum(self._data, 0.0))

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return Tensor(self._data + other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data - other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data * other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __repr__(self):
        dn = self.dtype
        dev = "vulkan" if self._device == "vulkan" else "cpu"
        return f"Tensor(shape={self.shape}, dtype={dn}, device={dev})"

    def __bool__(self):
        raise TypeError("use .numpy() to convert Tensor to a scalar")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return Tensor(self._data[key])

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value._data
        self._data[key] = value