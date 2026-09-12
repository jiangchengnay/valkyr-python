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


def device_name(handle):
    buf = ctypes.create_string_buffer(256)
    n = vkr_device_name(handle, buf, 256)
    if n < 0:
        raise RuntimeError(_last_error())
    return buf.value if buf.value else b"unknown"