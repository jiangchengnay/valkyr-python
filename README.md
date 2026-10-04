# valkyr — Python bindings

Python bindings for [Valkyr](https://github.com/Foundation42/valkyr), a cross-vendor LLM
inference and training engine written in pure Zig + Vulkan. The goal is a
PyTorch-shaped surface (`Tensor`, `Model`, `tensor()`, `cat()`) that runs
without any CUDA dependency.

Licensed under [GPL-3.0-or-later](LICENSE). The upstream Valkyr engine
is Apache-2.0 (or a commercial license from Foundation42), which is
compatible with GPLv3.

## What actually works today

| Surface | Status |
|---|---|
| `valkyr.Model.load(...)` / `.generate()` / `.chat()` | **Runs on Vulkan** via the `vkr_*` C-ABI entry points |
| `Model.train_lora(...)` | **Runs on Vulkan** — LoRA fine-tuning (Adam), writes `.lvkpt` checkpoints |
| `valkyr.Tensor` arithmetic (`add`, `matmul`, `relu`, …) | **CPU/numpy only.** `device` is a label; `.to("vulkan")` does not move data |
| `Model.generate(temperature=...)` | Accepted but **ignored** — the Session sampler is greedy-only |

## Architecture

```
valkyr/__init__.py    public surface: Tensor, Model, tensor, cat, native_version
valkyr/tensor.py      torch-like Tensor (currently numpy-backed)
valkyr/model.py       Model lifecycle + generation
valkyr/_native.py     ctypes declarations and error mapping
src/bindings.zig      C ABI: eight vkr_* exports, opaque Engine handle
```

The C ABI is intentionally tiny — all type marshalling (dict/list → JSON)
happens on the Python side, so the build has no pybind11 or C++ layer:

| Symbol | Purpose |
|---|---|
| `vkr_version()` | Packed `major<<16 \| minor<<8 \| patch` |
| `vkr_create_engine(path, precision, max_pos, max_new_tokens)` | Builds Model→Tokenizer→vk.Context→GpuModel→Session→Recorder→InferenceRunner, returns an opaque handle |
| `vkr_generate(handle, prompt, buf, cap, out_len, max_tokens)` | Single-turn generation |
| `vkr_chat(handle, json_messages, buf, cap, out_len, max_tokens)` | Multi-turn chat |
| `vkr_lora_train(handle, examples, output, ...)` | Vulkan LoRA fine-tuning (forward / backward / Adam) |
| `vkr_free_engine(handle)` | Release |
| `vkr_device_name(handle, buf, cap)` | Vulkan device name |
| `vkr_last_error(buf, cap)` | Human-readable message for the last failure |

Convention: every function returns `0` on success and `-1` on failure;
strings are NUL-terminated UTF-8; output buffers are caller-owned.

## Building

Requirements: **Zig 0.14.1+**, the **LunarG Vulkan SDK** (for `glslc` and
`vulkan-1`), and a Vulkan 1.3 GPU. Python 3.9+ with numpy.

```sh
# From this directory:
zig build -Doptimize=ReleaseFast      # → zig-out/bin/valkyr_core.dll
pip install .                         # builds + installs the wheel
```

`setup.py` locates Zig in this order: `$VALKYR_ZIG`, then
`<workspace>/.tools/**/zig[.exe]`, then `zig` on `PATH`.

The engine is consumed as a Zig package dependency (`../valkyr-engine`),
declared in `build.zig.zon` — see the engine's `build.zig`, which exposes
`b.addModule("valkyr_gpu", ...)`.

## Quick check

```python
import valkyr

m = valkyr.Model.load("/path/to/hf-model-dir", precision="bf16", max_pos=1024)
print(m.device_name)
print(m.generate("The capital of France is", max_tokens=32))
print(m.chat([{"role": "user", "content": "你好"}], max_tokens=64))
m.close()
```

```sh
python tests/test_smoke.py     # 30 checks, no model required
```

> Note: `tests/test_smoke.py` forces `sys.path` to this checkout so it
> never silently tests a stale `pip install`ed copy. If you get odd
> failures, make sure no old `valkyr` wheel shadows the local tree
> (`python -m pip uninstall valkyr` if needed).

## Known gaps

- `pyproject.toml` declares the wheel as `py3-none-any`, but
  `valkyr_core.dll` is a platform-specific Windows x64 binary. Other
  platforms must build their own; consider retagging as
  `py3-none-win_amd64` before publishing (see `setup.py` — the wheel is
  built from the checked-in sources, and `zig build` in this directory
  produces the platform library).
- `Model.load` on a model directory loads the families supported by the
  engine (Gemma / Llama / Mistral / Qwen3 / Qwen3.5 / Qwen2.5).
- `Model.generate(temperature=...)` is accepted but currently ignored
  (the Session sampler is greedy-only).  Sampling is a roadmap item.
- `Model.train_lora` covers the full-attention families (Qwen3-class
  with per-head Q/K-norm, Qwen2/Qwen2.5 with attention projection bias).
