"""High-level Model API — Valkyr GPU inference, torch-like API surface."""

import json as _json
from . import _native

_PRECISION_MAP = {
    "fp32": 0, "fp32_all": 0,
    "bf16": 1, "bf16_matmul": 1,
    "q4_0": 2, "q4_0_matmul": 2,
    "q4_k": 3, "q4_k_matmul": 3,
}
_VALID_MAX_POS = (256, 512, 1024, 2048, 4096, 8192, 16384)


class Model:
    """A loaded Valkyr model. All inference runs on the Vulkan GPU.

    Parameters
    ----------
    path : str
        Directory containing ``config.json``, ``tokenizer.json``, and
        ``.safetensors`` shards (HuggingFace format).
    device : str, optional
        ``"vulkan"`` (default) or ``"gpu"``.
    precision : str, optional
        Weight upload precision: ``"fp32"``, ``"bf16"`` (default),
        ``"q4_0"``, or ``"q4_k"``.
    max_pos : int, optional
        Maximum sequence length (KV cache size). 1024 default.
    max_new_tokens : int, optional
        Default generation cap. 256.
    """

    def __init__(self, path, device="vulkan", precision="bf16",
                 max_pos=1024, max_new_tokens=256):
        self._path = path
        self._precision = precision
        self._max_pos = max_pos
        self._max_new_tokens = max_new_tokens

        if device not in ("vulkan", "gpu"):
            raise ValueError(f"device={device!r}; only 'vulkan'/'gpu' supported")
        try:
            prec_code = _PRECISION_MAP[precision]
        except KeyError:
            raise ValueError(
                f"unknown precision {precision!r}; "
                f"choose from {sorted(set(_PRECISION_MAP.values()))}"
            )
        if max_pos not in _VALID_MAX_POS:
            raise ValueError(
                f"max_pos={max_pos!r}; choose from {_VALID_MAX_POS}"
            )

        self._handle = _native.create_engine(path, prec_code, max_pos, max_new_tokens)
        self._closed = False

    @classmethod
    def load(cls, path, device="vulkan", precision="bf16",
             max_pos=1024, max_new_tokens=256):
        """Load a model from a HuggingFace-format directory."""
        return cls(path, device=device, precision=precision,
                   max_pos=max_pos, max_new_tokens=max_new_tokens)

    @property
    def device_name(self):
        """GPU name as reported by Vulkan."""
        return _native.device_name(self._handle).decode("utf-8", "replace")

    @property
    def source_path(self):
        return self._path

    # ── Generation ─────────────────────────────────────────────────

    def generate(self, prompt, max_tokens=100, temperature=0.8):
        """Generate text from a user prompt.

        .. note::
            *temperature* is accepted for API compatibility with PyTorch.
            The current Valkyr Session sampler only supports greedy
            decoding; non-greedy temperature values fall back to greedy.
            Full temperature / top-k / top-p sampling will be added in a
            future release.
        """
        if getattr(self, '_closed', False):
            raise RuntimeError("model is closed")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        # temperature accepted but debited to greedy (Session limitation)
        return _native.generate(self._handle, prompt, max_tokens)

    def chat(self, messages, max_tokens=256, temperature=0.8):
        """Multi-turn chat via the model's chat template.

        Parameters
        ----------
        messages : list of dict
            ``[{"role": "user", "content": "..."}, ...]``.
            Roles: ``system``, ``user``, ``assistant``.
        """
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise TypeError("each message must be {'role': ..., 'content': ...}")
        j = _json.dumps(messages, ensure_ascii=False)
        return _native.chat(self._handle, j, max_tokens)

    def train_lora(self, dataset, epochs=3, lr=1e-4, **kwargs):
        """LoRA fine-tuning (not yet wired in v0.1).

        Raises ``NotImplementedError``.  Use the CLI ``valkyr --finetune``
        in the meantime.
        """
        raise NotImplementedError(
            "train_lora is not wired up in the Python binding v0.1. "
            "Valkyr exposes low-level LoRA kernels (recordLoraForward/Backward) "
            "but no high-level training loop yet. Use the CLI::\n\n"
            "  valkyr --finetune <model-dir> --lora-ckpt <lora.ckpt>\n\n"
            "Tracked as the next milestone."
        )

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self):
        if not getattr(self, '_closed', True):
            _native.free_engine(self._handle)
            self._closed = True

    def __del__(self):
        # 防御：__init__ 在 create_engine 前失败（或对象经 __new__ 构造）
        # 时可能没有 _handle；只在两者都齐备时才释放。
        if hasattr(self, '_closed') and not self._closed and hasattr(self, '_handle'):
            _native.free_engine(self._handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (
            f"<Model path={self._path!r} precision={self._precision!r}"
            f" max_pos={self._max_pos}>"
        )