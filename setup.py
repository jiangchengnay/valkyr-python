"""Build the `valkyr` Python package, compiling `valkyr_core` from Zig.

This project owns its own `build.zig` + `build.zig.zon`; the engine it
binds lives next door at `../valkyr-engine` and is pulled in as a Zig
package dependency. That means the build here is a single

    zig build -Doptimize=ReleaseFast

run in this directory, which produces
`zig-out/bin/valkyr_core.dll` (or `libvalkyr_core.{so,dylib}`).

Zig discovery order:
  1. ``$VALKYR_ZIG``                      — explicit override
  2. ``<workspace>/.tools/**/zig[.exe]``  — the workspace-local toolchain
  3. ``zig`` on ``PATH``
"""

import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Zig toolchain discovery ──────────────────────────────────────────


def _find_zig():
    override = os.environ.get("VALKYR_ZIG")
    if override and os.path.isfile(override):
        return os.path.abspath(override)

    # The workspace keeps a pinned Zig under <workspace>/tools/zig-0.14.1/.
    # From projects/valkyr-python-bindings/ that is two levels up.
    workspace = os.path.normpath(os.path.join(HERE, "..", ".."))
    tools = os.path.join(workspace, "tools", "zig-0.14.1")
    if os.path.isdir(tools):
        fn = "zig.exe" if os.name == "nt" else "zig"
        for dirpath, _, _ in os.walk(tools):
            candidate = os.path.join(dirpath, fn)
            if os.path.isfile(candidate):
                return candidate

    found = shutil.which("zig")
    if found:
        return found

    raise RuntimeError(
        "Zig compiler not found. Set VALKYR_ZIG, place Zig 0.14.1+ under "
        "<workspace>/.tools/, or put `zig` on PATH."
    )


# ── Native library build ─────────────────────────────────────────────

_LIB_NAMES = ("valkyr_core.dll", "libvalkyr_core.so", "libvalkyr_core.dylib")


def _build_native():
    zig = _find_zig()

    env = dict(os.environ)
    # The engine's build.zig compiles shaders with glslc and @cIncludes
    # <vulkan/vulkan.h>; both come from the LunarG SDK.
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        env["PATH"] = os.path.join(sdk, "Bin") + os.pathsep + env.get("PATH", "")

    # Keep caches inside the project tree so builds work under sandboxes
    # that block the user-global cache directories.
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", os.path.join(HERE, ".zig-global-cache"))
    env.setdefault("ZIG_LOCAL_CACHE_DIR", os.path.join(HERE, ".zig-cache"))

    subprocess.check_call([zig, "build", "-Doptimize=ReleaseFast"], cwd=HERE, env=env)

    for sub in ("bin", "lib"):
        for name in _LIB_NAMES:
            path = os.path.join(HERE, "zig-out", sub, name)
            if os.path.exists(path):
                return path

    raise RuntimeError(
        "zig build succeeded but no valkyr_core shared library was found "
        "under zig-out/bin or zig-out/lib"
    )


class BuildPy(build_py):
    """Compile the Zig shared library, then ship it inside the package."""

    def run(self):
        lib = _build_native()

        # Also drop it next to the .py sources so `pip install -e .` and
        # `pytest` find it without an extra step.
        in_tree = os.path.join(HERE, "valkyr", os.path.basename(lib))
        shutil.copy2(lib, in_tree)

        super().run()

        pkg_dir = self.get_package_dir("valkyr")
        if pkg_dir and os.path.isdir(pkg_dir):
            shutil.copy2(lib, os.path.join(pkg_dir, os.path.basename(lib)))


setup(cmdclass={"build_py": BuildPy})  # metadata lives in pyproject.toml
