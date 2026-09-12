"""End-to-end binding test: load a real model, generate text, chat.

Verifies not just that the binding runs, but that the model's output is
*logically correct*: for "capital of France" it must actually produce
"Paris" (outside any <think> block), with no mojibake and no
meaningless text.

Usage:
    python tests/test_e2e.py [model-dir [model-dir...]]

Defaults to the two known-good local models (Qwen3-0.6B and
DeepSeek-R1-Distill-Qwen-1.5B). Each model load takes ~45 s, so this is
NOT part of the fast smoke suite.
"""
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENGINE = os.path.join(os.path.dirname(HERE), "..", "valkyr-engine")

DEFAULT_MODELS = [
    os.path.join(ENGINE, "Qwen3-0.6B"),
    os.path.join(ENGINE, "DeepSeek-R1-Distill-Qwen-1.5B"),
]

# Strip a model's <think>...</think> reasoning block, keeping only the
# final answer text.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text):
    return _THINK_RE.sub("", text).strip()


def is_reasonable(text):
    """Output must be non-trivial and free of mojibake/garble."""
    if len(text) < 3:
        return False, f"too short: {text!r}"
    # Replacement char / heavy control chars / stray box-drawing noise
    if "\ufffd" in text:
        return False, "contains U+FFFD (mojibake)"
    # NUL bytes and friends are never valid model text
    for ch in ("\x00", "\x01", "\x02", "\x03"):
        if ch in text:
            return False, f"contains control char {ord(ch)}"
    return True, ""


_pass = 0
_fail = 0
_FAILURES = []


def ok(name, msg="OK"):
    global _pass
    _pass += 1
    print(f"  OK   {name}" + (f"  -- {msg}" if msg != "OK" else ""))


def fail(name, msg):
    global _fail
    _fail += 1
    _FAILURES.append((name, msg))
    print(f"  FAIL {name}  -- {msg}")


def run_model(model_path):
    name = os.path.basename(model_path)
    print(f"\n--- model: {name} ({model_path}) ---")

    import valkyr

    if not os.path.isdir(model_path):
        fail(f"[{name}] model dir", f"not found: {model_path}")
        return

    m = None
    try:
        m = valkyr.Model.load(model_path, precision="bf16", max_pos=1024,
                              max_new_tokens=16)
        ok(f"[{name}] Model.load", repr(m))
        ok(f"[{name}] device_name", m.device_name)
    except Exception as e:
        fail(f"[{name}] Model.load", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # 1) generate: must produce the correct factual answer (Paris)
    try:
        out = m.generate("What is the capital of France? Answer in one word.",
                         max_tokens=256)
        ok(f"[{name}] generate", f"{len(out)} chars")
        answer = strip_think(out)
        ok(f"[{name}] generate answer (no <think>)", repr(answer[:120]))
        if "paris" in answer.lower():
            ok(f"[{name}] generate correct answer", "contains 'Paris'")
        else:
            fail(f"[{name}] generate correct answer",
                 f"'Paris' not in answer: {answer[:160]!r}")
        good, why = is_reasonable(answer)
        if good:
            ok(f"[{name}] generate reasonable", "")
        else:
            fail(f"[{name}] generate reasonable", why)
    except Exception as e:
        fail(f"[{name}] generate", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    # 2) chat: multi-turn with a follow-up that depends on the first turn
    try:
        out = m.chat(
            [{"role": "user", "content": "Name a country in Europe."},
             {"role": "assistant", "content": "France."},
             {"role": "user", "content": "What is its capital? One word."}],
            max_tokens=256,
        )
        ok(f"[{name}] chat", f"{len(out)} chars")
        answer = strip_think(out)
        ok(f"[{name}] chat answer (no <think>)", repr(answer[:120]))
        if "paris" in answer.lower():
            ok(f"[{name}] chat correct answer", "contains 'Paris'")
        else:
            fail(f"[{name}] chat correct answer",
                 f"'Paris' not in answer: {answer[:160]!r}")
        good, why = is_reasonable(answer)
        if good:
            ok(f"[{name}] chat reasonable", "")
        else:
            fail(f"[{name}] chat reasonable", why)
    except Exception as e:
        fail(f"[{name}] chat", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        m.close()
        ok(f"[{name}] close")
    except Exception as e:
        fail(f"[{name}] close", f"{type(e).__name__}: {e}")


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    models = [os.path.abspath(p) for p in models]
    print("=" * 60)
    print("Valkyr binding -- end-to-end (logic correctness)")
    print("=" * 60)

    import valkyr
    ok("import", f"version={valkyr.__version__}")
    print(f"  native_version = {valkyr.native_version()}")

    for mp in models:
        run_model(mp)

    print("=" * 60)
    print(f"Results: {_pass} passed, {_fail} failed")
    for name, msg in _FAILURES:
        print(f"  FAIL {name}: {msg}")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
