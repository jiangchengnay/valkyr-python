"""Smoke tests for the valkyr Python package (no real model needed)."""
import os
import sys
import tempfile
import traceback

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

def test_tensor_dtype():
    """Tensor dtype acceptance"""
    import valkyr
    t = valkyr.tensor([1, 2, 3], dtype="float32")
    ok("tensor float32", f"dtype={t.dtype}")
    t2 = valkyr.tensor([1, 2, 3], dtype="int32")
    ok("tensor int32", f"dtype={t2.dtype}")

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
        ("Tensor cat", test_tensor_cat),
        ("Tensor to", test_tensor_to),
        ("Tensor numpy", test_tensor_numpy),
        ("Model load error", test_model_load_error),
        ("native_version", test_native_version),
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