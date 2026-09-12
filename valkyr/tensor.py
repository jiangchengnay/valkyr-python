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


def _as_arr(other):
    """Coerce a Tensor / array-like into a numpy array (view where safe)."""
    if isinstance(other, Tensor):
        return other._data
    return _np.asarray(other)


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

    # ── Introspection ────────────────────────────────────────────────

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

    @property
    def T(self):
        """Transpose for 1-D / 2-D tensors (torch-compatible)."""
        return self.transpose()

    def dim(self):
        return self._data.ndim

    def size(self, dim=None):
        """Return the shape, or the size along one dimension."""
        if dim is None:
            return self.shape
        return self._data.shape[dim]

    def is_floating_point(self):
        return _np.issubdtype(self._data.dtype, _np.floating)

    # ── Device / conversion ──────────────────────────────────────────

    def to(self, device):
        """Return a copy on the target device (v0.1: labels only, still CPU)."""
        if device not in ("cpu", "vulkan", "gpu"):
            raise ValueError(f"unsupported device {device!r}")
        t = Tensor(self._data.copy())
        t._device = "vulkan" if device in ("vulkan", "gpu") else "cpu"
        return t

    def cpu(self):
        """Return a CPU copy (v0.1 data is already on CPU)."""
        return self.to("cpu")

    def numpy(self):
        """Return a CPU numpy copy of the data."""
        return self._data.copy()

    def item(self):
        """Return the single value as a Python scalar. Errors if not scalar."""
        if self._data.size != 1:
            raise ValueError(
                f"only one element tensors can be converted to Python scalars, "
                f"but got {self.shape}"
            )
        return self._data.item()

    def tolist(self):
        """Return the data as a (possibly nested) Python list."""
        return self._data.tolist()

    def clone(self):
        """Return a copy of this tensor."""
        return Tensor(self._data.copy(), device=self._device)

    def detach(self):
        """No autograd in v0.1 — returns a copy with the same values."""
        return self.clone()

    def __array__(self, dtype=None):
        return self._data.astype(dtype) if dtype else self._data

    # ── In-place helpers ─────────────────────────────────────────────

    def copy_(self, other):
        """Copy values from another Tensor / array-like in place."""
        self._data[...] = _as_arr(other)
        return self

    def fill_(self, value):
        self._data[...] = value
        return self

    def zero_(self):
        self._data[...] = 0
        return self

    # ── Shape ops ────────────────────────────────────────────────────

    def reshape(self, *shape):
        """Return a new tensor with the given shape (same data order)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self._data.reshape(shape), device=self._device)

    def view(self, *shape):
        """Alias of reshape (v0.1 numpy backend does not enforce views)."""
        return self.reshape(*shape)

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten dimensions [start_dim, end_dim] into one."""
        shape = self.shape
        if end_dim < 0:
            end_dim += len(shape)
        new_shape = shape[:start_dim] + (
            int(_np.prod(shape[start_dim:end_dim + 1])),
        ) + shape[end_dim + 1:]
        return Tensor(self._data.reshape(new_shape), device=self._device)

    def squeeze(self, dim=None):
        """Remove size-1 dimensions (all, or a single one via dim)."""
        if dim is None:
            return Tensor(self._data.squeeze(), device=self._device)
        return Tensor(self._data.squeeze(axis=dim), device=self._device)

    def unsqueeze(self, dim):
        """Insert a size-1 dimension at position dim."""
        return Tensor(_np.expand_dims(self._data, axis=dim), device=self._device)

    def transpose(self, dim0=0, dim1=1):
        """Swap two dimensions (default: transpose a matrix)."""
        return Tensor(_np.swapaxes(self._data, dim0, dim1), device=self._device)

    def permute(self, *dims):
        """Reorder dimensions."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return Tensor(_np.transpose(self._data, dims), device=self._device)

    def swapaxes(self, dim0, dim1):
        return self.transpose(dim0, dim1)

    # ── Basic ops ────────────────────────────────────────────────────

    def add(self, other):
        return Tensor(self._data + _as_arr(other), device=self._device)

    def sub(self, other):
        return Tensor(self._data - _as_arr(other), device=self._device)

    def mul(self, other):
        return Tensor(self._data * _as_arr(other), device=self._device)

    def div(self, other):
        return Tensor(self._data / _as_arr(other), device=self._device)

    def pow(self, exponent):
        return Tensor(_np.power(self._data, _as_arr(exponent)), device=self._device)

    def matmul(self, other):
        return Tensor(_np.matmul(self._data, _as_arr(other)), device=self._device)

    def relu(self):
        return Tensor(_np.maximum(self._data, 0.0), device=self._device)

    def clamp(self, min=None, max=None):
        """Clamp values into [min, max] (either bound may be None)."""
        return Tensor(_np.clip(self._data, min, max), device=self._device)

    # ── Unary math ───────────────────────────────────────────────────

    def neg(self):
        return Tensor(-self._data, device=self._device)

    def abs(self):
        return Tensor(_np.abs(self._data), device=self._device)

    def sqrt(self):
        return Tensor(_np.sqrt(self._data), device=self._device)

    def exp(self):
        return Tensor(_np.exp(self._data), device=self._device)

    def log(self):
        return Tensor(_np.log(self._data), device=self._device)

    def log2(self):
        return Tensor(_np.log2(self._data), device=self._device)

    def log10(self):
        return Tensor(_np.log10(self._data), device=self._device)

    def sigmoid(self):
        return Tensor(1.0 / (1.0 + _np.exp(-self._data)), device=self._device)

    def tanh(self):
        return Tensor(_np.tanh(self._data), device=self._device)

    # ── Reduction ops ────────────────────────────────────────────────

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.sum(self._data), device=self._device)
        return Tensor(
            _np.sum(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.mean(self._data), device=self._device)
        return Tensor(
            _np.mean(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def max(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.max(self._data), device=self._device)
        return Tensor(
            _np.max(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def min(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.min(self._data), device=self._device)
        return Tensor(
            _np.min(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def argmax(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.argmax(self._data), device=self._device)
        return Tensor(
            _np.argmax(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def argmin(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.argmin(self._data), device=self._device)
        return Tensor(
            _np.argmin(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def all(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(bool(_np.all(self._data)), device=self._device)
        return Tensor(
            _np.all(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def any(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(bool(_np.any(self._data)), device=self._device)
        return Tensor(
            _np.any(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    def prod(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_np.prod(self._data), device=self._device)
        return Tensor(
            _np.prod(self._data, axis=dim, keepdims=keepdim), device=self._device
        )

    # ── Comparison ───────────────────────────────────────────────────

    def eq(self, other):
        return Tensor(self._data == _as_arr(other), device=self._device)

    def ne(self, other):
        return Tensor(self._data != _as_arr(other), device=self._device)

    def lt(self, other):
        return Tensor(self._data < _as_arr(other), device=self._device)

    def le(self, other):
        return Tensor(self._data <= _as_arr(other), device=self._device)

    def gt(self, other):
        return Tensor(self._data > _as_arr(other), device=self._device)

    def ge(self, other):
        return Tensor(self._data >= _as_arr(other), device=self._device)

    # ── Python operators ─────────────────────────────────────────────

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return Tensor(_as_arr(other) + self._data, device=self._device)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return Tensor(_as_arr(other) - self._data, device=self._device)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return Tensor(_as_arr(other) * self._data, device=self._device)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return Tensor(_as_arr(other) / self._data, device=self._device)

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __rpow__(self, base):
        return Tensor(_np.power(_as_arr(base), self._data), device=self._device)

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs()

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        return Tensor(_np.matmul(_as_arr(other), self._data), device=self._device)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __repr__(self):
        dn = self.dtype
        dev = "vulkan" if self._device == "vulkan" else "cpu"
        return f"Tensor(shape={self.shape}, dtype={dn}, device={dev})"

    def __bool__(self):
        raise TypeError("use .numpy() to convert a Tensor to a scalar")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return Tensor(self._data[key], device=self._device)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value._data
        self._data[key] = value

    def __iter__(self):
        # Iterate over the leading dimension, wrapping each row as a Tensor.
        for i in range(len(self._data)):
            yield Tensor(self._data[i], device=self._device)
