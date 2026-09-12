"""End-to-end binding test: load a real model, generate text, chat.

Usage:
    python tests/test_e2e.py [model-dir]

Defaults to ../valkyr-engine/Qwen3-0.6B. Model load takes ~45 s, so this
is NOT part of the fast smoke suite.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

DEFAULT_MODEL = os.path.join(
    os.path.dirname(HERE), "..", "valkyr-engine", "Qwen3-0.6B"
)

_pass = 0
_fail = 0


def ok(name, msg="OK"):
    global _pass
    _pass += 1
    print(f"  OK   {name}" + (f"  -- {msg}" if msg != "OK" else ""))


def fail(name, msg):
    global _fail
    _fail += 1
    print(f"  FAIL {name}  -- {msg}")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model_path = os.path.abspath(model_path)
    print("=" * 60)
    print("Valkyr binding -- end-to-end")
    print("=" * 60)
    print(f"model: {model_path}")

    import valkyr

    ok("import", f"version={valkyr.__version__}")
    print(f"  native_version = {valkyr.native_version()}")

    if not os.path.isdir(model_path):
        fail("model dir", f"not found: {model_path}")
        return 1

    m = None
    try:
        m = valkyr.Model.load(model_path, precision="bf16", max_pos=1024,
                              max_new_tokens=16)
        ok("Model.load", repr(m))
        print(f"  device_name = {m.device_name}")
    except Exception as e:
        fail("Model.load", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    try:
        out = m.generate("The capital of France is", max_tokens=64)
        ok("generate", repr(out))
        # Qwen3 is a thinking model: a short cap only opens the
        # <think> block, so assert on coherent non-empty text rather
        # than requiring the answer to appear within the cap.
        if len(out.strip()) >= 8:
            ok("generate content", f"{len(out)} chars")
        else:
            fail("generate content", f"too short: {out!r}")
    except Exception as e:
        fail("generate", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        out = m.chat(
            [{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
        )
        ok("chat", repr(out))
    except Exception as e:
        fail("chat", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        m.close()
        ok("close")
    except Exception as e:
        fail("close", f"{type(e).__name__}: {e}")

    print("=" * 60)
    print(f"Results: {_pass} passed, {_fail} failed")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
