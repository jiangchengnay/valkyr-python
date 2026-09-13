"""Vulkan LoRA training test via the Python bindings.

Usage: python tests/test_lora_train.py [model-dir]
Default: ../valkyr-engine/Qwen3-0.6B (the only trainer-supported family
here — full attention, no bias). Requires GPU; ~1-3 min for 20 steps.
"""
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENGINE = os.path.join(os.path.dirname(HERE), "..", "valkyr-engine")
DEFAULT_MODEL = os.path.join(ENGINE, "Qwen3-0.6B")

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
    model_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    print("=" * 60)
    print("Valkyr binding -- Vulkan LoRA training")
    print("=" * 60)
    print(f"model: {model_path}")

    import valkyr

    if not os.path.isdir(model_path):
        fail("model dir", f"not found: {model_path}")
        return 1

    m = None
    try:
        m = valkyr.Model.load(model_path, precision="bf16", max_pos=1024,
                              max_new_tokens=16)
        ok("Model.load", repr(m))
    except Exception as e:
        fail("Model.load", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Training data: repeated simple facts so the probe should shift.
    examples = [
        "法国的首都是巴黎。The capital of France is Paris.",
        "法国的首都是巴黎。The capital of France is Paris.",
        "法国的首都是巴黎。The capital of France is Paris.",
        "法国的首都是巴黎。The capital of France is Paris.",
    ]
    probe = "法国的首都是"  # will be sampled before/after

    tmpdir = tempfile.mkdtemp(prefix="valkyr_lora_")
    out_path = os.path.join(tmpdir, "test.lvkpt")

    try:
        summary = m.train_lora(
            examples, out_path,
            steps=20, lr=1e-5, rank=16, alpha=32.0,
            targets="all_attn", n_pos=16, rotate=True,
            probe=probe, n_gen=20,
        )
        ok("train_lora ran", "")
        print(summary)

        # Summary must contain the CE markers.
        if "CE initial" in summary and "CE final" in summary:
            ok("summary has CE markers", "")
        else:
            fail("summary has CE markers", summary[:200])

        if "saved .lvkpt" in summary and os.path.isfile(out_path):
            sz = os.path.getsize(out_path)
            ok("lvkpt saved", f"{sz} bytes")
        else:
            fail("lvkpt saved", f"missing or not mentioned: {out_path!r}")

        # CE must decrease (training worked).
        try:
            import re
            init_m = re.search(r"CE initial: ([0-9.]+)", summary)
            final_m = re.search(r"CE final: ([0-9.]+)", summary)
            if init_m and final_m:
                ce_init = float(init_m.group(1))
                ce_final = float(final_m.group(1))
                if ce_final < ce_init:
                    ok("CE decreased", f"{ce_init:.4f} -> {ce_final:.4f}")
                else:
                    fail("CE decreased", f"{ce_init:.4f} -> {ce_final:.4f}")
        except Exception as e:
            fail("CE parse", str(e))

        # Probe after should differ from before (model shifted).
        try:
            before = summary.split("probe before:")[1].split("probe after:")[0].strip()
            after = summary.split("probe after:")[1].split("saved .lvkpt")[0].strip()
            ok("probe before", repr(before[:60]))
            ok("probe after", repr(after[:60]))
            if after and after != before and len(after) >= 3:
                ok("probe shifted", "")
            else:
                fail("probe shifted", f"before={before!r} after={after!r}")
        except Exception as e:
            fail("probe parse", str(e))
    except Exception as e:
        fail("train_lora", f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            m.close()
            ok("close")
        except Exception:
            pass
        # Keep the .lvkpt for inspection; note its path.
        print(f"  lvpt kept at: {out_path}")

    print("=" * 60)
    print(f"Results: {_pass} passed, {_fail} failed")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
