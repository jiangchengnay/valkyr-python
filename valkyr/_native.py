"""ctypes glue for the valkyr_core shared library (built by `zig build`)."""
import ctypes
import os

_LIB_CANDIDATES = ["valkyr_core.dll", "libvalkyr_core.so", "libvalkyr_core.dylib"]

def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    for n in _LIB_CANDIDATES:
        p = os.path.join(here, n)
        if os.path.exists(p):
            return p
        p2 = os.path.join(here, "..", "zig-out", "bin", n)
        if os.path.exists(p2):
            return p2
        p3 = os.path.join(here, "..", "zig-out", "lib", n)
        if os.path.exists(p3):
            return p3
    raise OSError(
        "valkyr_core native library not found. Build it with:\n"
        "  zig build -Doptimize=ReleaseFast"
    )

_lib = ctypes.CDLL(_find_lib())


def _f(name, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


vkr_version = _f("vkr_version", ctypes.c_uint32)
vkr_create_engine = _f(
    "vkr_create_engine",
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_uint32,
)
vkr_generate = _f(
    "vkr_generate",
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_uint32,
)
vkr_chat = _f(
    "vkr_chat",
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_uint32,
)
vkr_free_engine = _f("vkr_free_engine", None, ctypes.c_size_t)
vkr_device_name = _f(
    "vkr_device_name", ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
)
vkr_lora_train = _f(
    "vkr_lora_train",
    ctypes.c_int,
    ctypes.c_size_t,          # handle
    ctypes.c_char_p,          # examples_json (["text", ...])
    ctypes.c_char_p,          # out_path (.lvkpt)
    ctypes.c_uint32,          # steps
    ctypes.c_float,           # lr
    ctypes.c_uint32,          # rank
    ctypes.c_float,           # alpha
    ctypes.c_uint32,          # n_pos
    ctypes.c_uint32,          # targets_mask
    ctypes.c_int,             # rotate
    ctypes.c_char_p,          # probe ("" = none)
    ctypes.c_uint32,          # n_gen
    ctypes.c_void_p,          # out buffer
    ctypes.c_size_t,          # out cap
    ctypes.POINTER(ctypes.c_size_t),  # out_len
)
vkr_last_error = _f(
    "vkr_last_error", ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
)


def _last_error():
    buf = ctypes.create_string_buffer(512)
    n = vkr_last_error(buf, 512)
    return buf.value.decode("utf-8", "replace") if n > 0 else "unknown error"


def create_engine(model_path, precision, max_pos, max_new_tokens):
    handle = ctypes.c_size_t(0)
    rc = vkr_create_engine(
        ctypes.byref(handle),
        model_path.encode("utf-8"),
        precision,
        max_pos,
        max_new_tokens,
    )
    if rc != 0:
        raise RuntimeError(_last_error())
    return handle.value


def generate(handle, prompt, max_tokens):
    cap = max((max_tokens + 1) * 64, 4096)
    buf = ctypes.create_string_buffer(cap)
    n = ctypes.c_size_t(0)
    rc = vkr_generate(handle, prompt.encode("utf-8"), buf, cap, ctypes.byref(n), max_tokens)
    if rc != 0:
        raise RuntimeError(_last_error())
    return buf.raw[: n.value].decode("utf-8", "replace")


def chat(handle, json_str, max_tokens):
    cap = max((max_tokens + 1) * 64, 4096)
    buf = ctypes.create_string_buffer(cap)
    n = ctypes.c_size_t(0)
    rc = vkr_chat(handle, json_str.encode("utf-8"), buf, cap, ctypes.byref(n), max_tokens)
    if rc != 0:
        raise RuntimeError(_last_error())
    return buf.raw[: n.value].decode("utf-8", "replace")


def free_engine(handle):
    vkr_free_engine(handle)


def lora_train(handle, examples, out_path, steps, lr, rank, alpha,
               n_pos, targets_mask, rotate, probe, n_gen):
    """Vulkan LoRA fine-tune. Returns the training summary text.

    examples : list of str — training texts (packed with EOS separators).

    Runs on a dedicated thread with a large stack: the Zig training
    stack (Runner.init / Context / TrainWeights) allocates sizeable
    structs on the call stack, which overflows the ~1 MB default Python
    thread stack (engine CLI runs on the process main thread, which has
    a much bigger stack).
    """
    import json as _json
    import threading

    cap = 65536
    buf = ctypes.create_string_buffer(cap)
    n = ctypes.c_size_t(0)
    examples_json = _json.dumps(examples, ensure_ascii=False).encode("utf-8")

    result = {}
    def _work():
        result["rc"] = vkr_lora_train(
            handle, examples_json, out_path.encode("utf-8"),
            steps, lr, rank, alpha, n_pos, targets_mask,
            1 if rotate else 0,
            (probe or "").encode("utf-8"), n_gen,
            buf, cap, ctypes.byref(n),
        )

    try:
        _old_stack = threading.stack_size()
        threading.stack_size(64 * 1024 * 1024)  # 64 MiB for the worker
    except (ValueError, RuntimeError):
        _old_stack = None
    try:
        t = threading.Thread(target=_work)
        t.start()
        t.join()
    finally:
        if _old_stack is not None:
            try:
                threading.stack_size(_old_stack)
            except (ValueError, RuntimeError):
                pass

    rc = result.get("rc")
    if rc is None:
        raise RuntimeError("lora_train worker failed to run")
    if rc != 0:
        raise RuntimeError(_last_error())
    return buf.raw[: n.value].decode("utf-8", "replace")


def device_name(handle):
    buf = ctypes.create_string_buffer(256)
    n = vkr_device_name(handle, buf, 256)
    if n < 0:
        raise RuntimeError(_last_error())
    return buf.value if buf.value else b"unknown"