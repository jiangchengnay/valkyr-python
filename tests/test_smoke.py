"""Smoke tests for the valkyr Python package (no real model needed)."""
import os
import sys
import tempfile
import traceback

# Ensure we test THIS checkout, not a stale site-packages install.
# Without this, `import valkyr` resolves to any previously
# `pip install`ed copy, silently testing old code.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Test helpers ─────────────────────────────────────────────────────

_pass = 0
_fail = 0
_FAILURES = []

def ok(name, msg="OK"):
    global _pass
    _pass += 1
    print(f"  ✓ {name}" + (f"  — {msg}" if msg != "OK" else ""))

def fail(name, msg):
    global _fail
    _fail += 1
    _FAILURES.append((name, msg))
    print(f"  ✗ {name}  — {msg}")

# ── Tests ────────────────────────────────────────────────────────────

def test_import():
    """import valkyr"""
    try:
        import valkyr
        ok("import valkyr", f"version={valkyr.__version__}")
        # Guard against silently testing a stale site-packages install.
        pkg_file = os.path.abspath(valkyr.__file__)
        if not pkg_file.startswith(ROOT):
            fail("import valkyr", f"resolved outside checkout: {pkg_file}")
            raise AssertionError(f"valkyr imported from {pkg_file}")
    except Exception as e:
        fail("import valkyr", str(e))
        traceback.print_exc()
        raise

def test_tensor_create():
    """Tensor creation and basic properties"""
    import valkyr
    import numpy as np

    t = valkyr.tensor([[1.0, 2.0], [3.0, 4.0]])
    ok("tensor shape", repr(t.shape))
    assert t.shape == (2, 2), f"shape={t.shape}"
    assert t.numel == 4
    assert t.device == "vulkan"
    ok("tensor dtype", t.dtype)
    assert t.ndim == 2
    assert t.dim() == 2
    assert t.size() == (2, 2)
    assert t.size(1) == 2
    assert t.T.shape == (2, 2)
    assert not t.is_floating_point() or t.dtype == "float32"

def test_tensor_dtype():
    """Tensor dtype acceptance"""
    import valkyr
    t = valkyr.tensor([1, 2, 3], dtype="float32")
    ok("tensor float32", f"dtype={t.dtype}")
    t2 = valkyr.tensor([1, 2, 3], dtype="int32")
    ok("tensor int32", f"dtype={t2.dtype}")
    try:
        valkyr.tensor([1], dtype="not-a-dtype")
        fail("tensor bad dtype", "should raise ValueError")
    except ValueError:
        ok("tensor bad dtype", "raises ValueError")

def test_tensor_ops():
    """Tensor add/matmul/relu"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = valkyr.tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a.add(b)
    assert c.shape == (2, 2)
    assert np.allclose(c.numpy(), [[6, 8], [10, 12]])
    ok("tensor.add", "ok")

    d = a.matmul(b)
    assert d.shape == (2, 2)
    assert np.allclose(d.numpy(), [[19, 22], [43, 50]])
    ok("tensor.matmul", "ok")

    e = a.relu()
    ok("tensor.relu", "ok")

def test_tensor_arithmetic_ops():
    """Tensor sub/mul/div/pow/clamp + operator overloads"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([6.0, 9.0, 4.0])
    b = valkyr.tensor([2.0, 3.0, 2.0])
    assert np.allclose((a - b).numpy(), [4, 6, 2])
    assert np.allclose((a * b).numpy(), [12, 27, 8])
    assert np.allclose((a / b).numpy(), [3, 3, 2])
    assert np.allclose((a ** 2).numpy(), [36, 81, 16])
    assert np.allclose((2 ** a).numpy(), [64, 512, 16])  # __rpow__
    assert np.allclose(a.sub(b).numpy(), [4, 6, 2])
    assert np.allclose(a.mul(b).numpy(), [12, 27, 8])
    assert np.allclose(a.div(b).numpy(), [3, 3, 2])
    assert np.allclose(a.pow(2).numpy(), [36, 81, 16])
    assert np.allclose(a.clamp(min=5.0, max=8.0).numpy(), [6, 8, 5])
    assert np.allclose((10 - a).numpy(), [4, 1, 6])   # __rsub__
    assert np.allclose((10 / a).numpy(), [10/6, 10/9, 2.5])  # __rtruediv__
    ok("tensor arithmetic ops", "ok")

def test_tensor_unary_math():
    """Tensor neg/abs/sqrt/exp/log/sigmoid/tanh"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([-2.0, 0.5, 4.0])
    assert np.allclose((-a).numpy(), [2.0, -0.5, -4.0])
    assert np.allclose(abs(a).numpy(), [2.0, 0.5, 4.0])
    assert np.allclose(a.neg().numpy(), [2.0, -0.5, -4.0])
    assert np.allclose(a.abs().numpy(), [2.0, 0.5, 4.0])
    b = valkyr.tensor([1.0, 4.0, 9.0])
    assert np.allclose(b.sqrt().numpy(), [1.0, 2.0, 3.0])
    assert np.allclose(b.exp().numpy(), np.exp([1.0, 4.0, 9.0]))
    assert np.allclose(b.log().numpy(), np.log([1.0, 4.0, 9.0]))
    assert np.allclose(b.log2().numpy(), np.log2([1.0, 4.0, 9.0]))
    assert np.allclose(b.log10().numpy(), np.log10([1.0, 4.0, 9.0]))
    s = valkyr.tensor([0.0, 1.0])
    assert np.allclose(s.sigmoid().numpy(), [0.5, 1/(1+np.exp(-1))])
    assert np.allclose(s.tanh().numpy(), np.tanh([0.0, 1.0]))
    ok("tensor unary math", "ok")

def test_tensor_reductions():
    """Tensor sum/mean/max/min/argmax/prod/all/any"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert abs(a.sum().item() - 10.0) < 1e-6
    assert abs(a.mean().item() - 2.5) < 1e-6
    assert a.max().item() == 4.0
    assert a.min().item() == 1.0
    assert a.argmax().item() == 3
    assert a.argmin().item() == 0
    assert a.prod().item() == 24.0
    assert a.sum(dim=0).tolist() == [4.0, 6.0]
    assert a.sum(dim=1).tolist() == [3.0, 7.0]
    assert a.max(dim=1).tolist() == [2.0, 4.0]
    b = valkyr.tensor([True, False, True])
    assert bool(b.any().item()) is True
    assert bool(valkyr.tensor([True, True]).all().item()) is True
    ok("tensor reductions", "ok")

def test_tensor_shape_ops():
    """Tensor reshape/view/flatten/squeeze/unsqueeze/transpose/permute"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r = a.reshape(3, 2)
    assert r.shape == (3, 2)
    assert r.view(6).shape == (6,)
    assert a.flatten().shape == (6,)
    assert a.T.shape == (3, 2)
    assert a.transpose(0, 1).shape == (3, 2)
    assert a.permute(1, 0).shape == (3, 2)
    s = valkyr.tensor([[[1.0]], [[2.0]]])
    assert s.squeeze().shape == (2,)
    assert s.squeeze(2).shape == (2, 1)
    v = valkyr.tensor([1.0, 2.0])
    assert v.unsqueeze(0).shape == (1, 2)
    assert v.unsqueeze(1).shape == (2, 1)
    assert np.allclose(r.numpy(), [[1, 2], [3, 4], [5, 6]])
    ok("tensor shape ops", "ok")

def test_tensor_comparisons():
    """Tensor eq/ne/lt/le/gt/ge"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([1.0, 5.0, 3.0])
    assert a.eq(3).tolist() == [False, False, True]
    assert a.ne(3).tolist() == [True, True, False]
    assert a.lt(3).tolist() == [True, False, False]
    assert a.le(3).tolist() == [True, False, True]
    assert a.gt(3).tolist() == [False, True, False]
    assert a.ge(3).tolist() == [False, True, True]
    assert (a == 3).tolist() == [False, False, True]
    assert (a != 3).tolist() == [True, True, False]
    assert (a < 3).tolist() == [True, False, False]
    assert (a >= 3).tolist() == [False, True, True]
    ok("tensor comparisons", "ok")

def test_tensor_scalar_conversion():
    """Tensor item/tolist/clone/copy_/fill_/zero_"""
    import valkyr
    import numpy as np
    s = valkyr.tensor(3.5)
    assert abs(s.item() - 3.5) < 1e-6
    a = valkyr.tensor([[1.0, 2.0]])
    assert a.tolist() == [[1.0, 2.0]]
    c = a.clone()
    assert np.allclose(c.numpy(), a.numpy())
    assert c is not a
    c.fill_(7.0)
    assert c.tolist() == [[7.0, 7.0]]
    c.zero_()
    assert c.tolist() == [[0.0, 0.0]]
    d = valkyr.tensor([1.0, 2.0])
    d.copy_([9.0, 8.0])
    assert d.tolist() == [9.0, 8.0]
    # __getitem__ / __setitem__ / __iter__
    row = a[0]
    assert row.shape == (2,)
    a[0, 0] = 42.0
    assert a[0, 0].item() == 42.0
    vals = [t.item() for t in valkyr.tensor([1.0, 2.0])]
    assert vals == [1.0, 2.0]
    try:
        bool(a)
        fail("tensor bool", "should raise TypeError")
    except TypeError:
        ok("tensor bool raises", "ok")
    ok("tensor scalar conversion", "ok")

def test_tensor_cat():
    """cat function"""
    import valkyr
    import numpy as np
    a = valkyr.tensor([1, 2])
    b = valkyr.tensor([3, 4])
    c = valkyr.cat([a, b])
    assert c.shape == (4,)
    assert np.allclose(c.numpy(), [1, 2, 3, 4])
    ok("cat", "ok")

def test_tensor_to():
    """Tensor.to(device)"""
    import valkyr
    t = valkyr.tensor([1, 2, 3], device="cpu")
    assert t.device == "cpu"
    t2 = t.to("vulkan")
    assert t2.device == "vulkan"
    t3 = t.cpu()
    assert t3.device == "cpu"
    ok("tensor.to", "ok")

def test_tensor_numpy():
    """Tensor.numpy() round-trip"""
    import valkyr
    import numpy as np
    t = valkyr.tensor([1.5, 2.5])
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [1.5, 2.5])
    ok("tensor.numpy", "ok")

def test_model_constructor_validation():
    """Model parameter validation happens before touching the native lib"""
    import valkyr
    # Bad device
    try:
        valkyr.Model("/some/path", device="cuda")
        fail("model bad device", "should raise ValueError")
    except ValueError as e:
        ok("model bad device", str(e)[:50])
    # Bad precision
    try:
        valkyr.Model("/some/path", precision="fp64")
        fail("model bad precision", "should raise ValueError")
    except ValueError as e:
        ok("model bad precision", str(e)[:50])
    # Bad max_pos
    try:
        valkyr.Model("/some/path", max_pos=777)
        fail("model bad max_pos", "should raise ValueError")
    except ValueError as e:
        ok("model bad max_pos", str(e)[:50])
    # No native lib needed for these — validation is pure Python.

def test_model_loaded_validation():
    """Model instance validation without a real GPU model.

    Uses object.__new__ to skip native engine creation; verifies that
    argument validation and lifecycle guards work at the Python layer.
    """
    import valkyr
    m = object.__new__(valkyr.Model)
    m._closed = True
    # generate on a closed model
    try:
        m.generate("hello")
        fail("generate closed", "should raise RuntimeError")
    except RuntimeError as e:
        ok("generate closed", str(e)[:40])
    # non-str prompt
    m2 = object.__new__(valkyr.Model)
    m2._closed = False
    try:
        m2.generate(12345)
        fail("generate non-str", "should raise TypeError")
    except TypeError as e:
        ok("generate non-str", str(e)[:40])
    # chat: non-list
    try:
        m2.chat("not a list")
        fail("chat non-list", "should raise TypeError")
    except TypeError as e:
        ok("chat non-list", str(e)[:40])
    # chat: bad message shape
    try:
        m2.chat([{"role": "user"}])
        fail("chat bad msg", "should raise TypeError")
    except TypeError as e:
        ok("chat bad msg", str(e)[:40])
    # context manager closes (handle 0 is a no-op in the native lib)
    m3 = object.__new__(valkyr.Model)
    m3._closed = False
    m3._handle = 0
    with m3:
        pass
    assert m3._closed, "context manager should close the model"
    ok("context manager", "ok")

def test_model_load_error():
    """Model.load with nonexistent path raises RuntimeError"""
    import valkyr
    try:
        m = valkyr.Model.load("/nonexistent/path/to/model")
        fail("model.load nonexistent", "should have raised RuntimeError")
    except RuntimeError as e:
        # Expected: Zig build may fail earlier, but we handle the error
        ok("model.load error", str(e)[:60])
    except Exception as e:
        msg = str(e)
        # setup.py build may not have run yet, so valkyr_core might not
        # exist.  Accept that as a graceful error too.
        if "not found" in msg.lower() or "not built" in msg.lower():
            ok("model.load (no native lib)", msg[:60])
        else:
            fail("model.load error", msg)

def test_native_version():
    """native_version() returns a tuple"""
    import valkyr
    v = valkyr.native_version()
    assert isinstance(v, tuple) and len(v) == 3
    ok("native_version", f"{v}")

def test_lib_candidates():
    """The ctypes loader resolves a real library file"""
    import valkyr
    import os
    from valkyr import _native
    lib_path = _native._find_lib()
    assert os.path.isfile(lib_path), f"lib not found: {lib_path}"
    assert os.path.basename(lib_path) in _native._LIB_CANDIDATES
    ok("lib resolved", os.path.basename(lib_path))

# ── Runner ───────────────────────────────────────────────────────────

def main():
    global _pass, _fail
    print(f"\n{'='*50}")
    print("Valkyr Python bindings — smoke tests")
    print(f"{'='*50}\n")

    tests = [
        ("import", test_import),
        ("Tensor create", test_tensor_create),
        ("Tensor dtype", test_tensor_dtype),
        ("Tensor ops", test_tensor_ops),
        ("Tensor arithmetic", test_tensor_arithmetic_ops),
        ("Tensor unary math", test_tensor_unary_math),
        ("Tensor reductions", test_tensor_reductions),
        ("Tensor shape ops", test_tensor_shape_ops),
        ("Tensor comparisons", test_tensor_comparisons),
        ("Tensor scalar conversion", test_tensor_scalar_conversion),
        ("Tensor cat", test_tensor_cat),
        ("Tensor to", test_tensor_to),
        ("Tensor numpy", test_tensor_numpy),
        ("Model ctor validation", test_model_constructor_validation),
        ("Model loaded validation", test_model_loaded_validation),
        ("Model load error", test_model_load_error),
        ("native_version", test_native_version),
        ("lib candidates", test_lib_candidates),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            fail(name, f"unhandled: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()

    total = _pass + _fail
    print(f"\n{'='*50}")
    print(f"Results: {_pass}/{total} passed", end="")
    if _fail:
        print(f", {_fail} failed:")
        for name, msg in _FAILURES:
            print(f"  ✗ {name}: {msg}")
    else:
        print(" — all good!")
    print(f"{'='*50}\n")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
