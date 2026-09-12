//! Build for `valkyr_core` — the C-ABI shared library consumed by the
//! `valkyr` Python package (ctypes).
//!
//! This project is deliberately separate from `../valkyr-engine`: the
//! engine is the vendored/patched upstream Zig + Vulkan LLM runtime, and
//! this is the Python-facing shim on top of it. It links the engine's
//! `valkyr_gpu` module and exports the seven `vkr_*` symbols declared in
//! `src/bindings.zig`.
//!
//!   zig build -Doptimize=ReleaseFast
//!
//! produces zig-out/bin/valkyr_core.dll (Windows) / libvalkyr_core.so
//! (Linux) / libvalkyr_core.dylib (macOS).

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── The engine, as a package dependency ───────────────────────────
    const valkyr_dep = b.dependency("valkyr", .{
        .target = target,
        .optimize = optimize,
    });
    const valkyr_gpu_mod = valkyr_dep.module("valkyr_gpu");

    // ── Shared library ────────────────────────────────────────────────
    const shared = b.addSharedLibrary(.{
        .name = "valkyr_core",
        .root_source_file = b.path("src/bindings.zig"),
        .target = target,
        .optimize = optimize,
    });
    shared.root_module.addImport("valkyr_gpu", valkyr_gpu_mod);

    // Vulkan headers + loader. Mirrors the engine's own Windows handling:
    // the LunarG SDK installer sets VULKAN_SDK, which is where
    // vulkan/vulkan.h and the vulkan-1 import library live.
    if (target.result.os.tag == .windows) {
        if (std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK")) |sdk| {
            shared.addIncludePath(.{ .cwd_relative = b.fmt("{s}/Include", .{sdk}) });
            shared.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/Lib", .{sdk}) });
        } else |_| {}
        shared.linkSystemLibrary("vulkan-1");
    } else {
        shared.linkSystemLibrary("vulkan");
    }
    shared.linkLibC();

    b.installArtifact(shared);

    // ── Install into the Python package ───────────────────────────────
    // `valkyr/_native.py` looks for the library *inside the package
    // directory first*, then falls back to zig-out. Installing only to
    // zig-out therefore lets a stale `valkyr/valkyr_core.dll` silently
    // shadow a fresh build, which is exactly how a fixed recorder went
    // unnoticed. Copy the artifact into the package tree so a plain
    // `zig build` is always what Python loads.
    const py_install = b.addUpdateSourceFiles();
    py_install.addCopyFileToSource(shared.getEmittedBin(), b.fmt("valkyr/{s}", .{shared.out_filename}));
    b.getInstallStep().dependOn(&py_install.step);
}
