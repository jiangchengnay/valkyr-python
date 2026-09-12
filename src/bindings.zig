//! valkyr_core — C ABI binding layer between the `valkyr` Python package
//! (ctypes) and Valkyr's `valkyr_gpu` Zig module.
//!
//! Owns the full model lifecycle behind one opaque `Engine` handle:
//!   Model.load -> Tokenizer -> vk.Context -> GpuModel.upload -> Session
//!   -> Recorder -> InferenceRunner
//! Generation goes through the runner's chat protocol; the caller drains
//! token events until `finish`. All strings cross the boundary as
//! null-terminated UTF-8; output buffers are caller-owned byte arrays.
//!
//! Error convention: every exported function returns 0 on success and -1
//! on failure; the human-readable message is fetched via vkr_last_error.

const std = @import("std");

const vkr = @import("valkyr_gpu");

const model_mod = vkr.model;
const tokenizer_mod = vkr.tokenizer;
const gpu_model = vkr.gpu_model;
const vk_mod = vkr.vk;
const session_mod = vkr.session;
const recorder_mod = vkr.recorder;
const runner_mod = vkr.inference.runner;
const chat_template_mod = vkr.chat_template;

const MAX_ENGINES: usize = 16;
var g_engines: [MAX_ENGINES]?*Engine = [_]?*Engine{null} ** MAX_ENGINES;

const MAX_DEV_NAME = 256;

const Engine = struct {
    gpa: std.mem.Allocator,

    cpu: model_mod.Model,
    tok: tokenizer_mod.Tokenizer,
    ctx: *vk_mod.Context,
    gm: *gpu_model.GpuModel,
    sess: *session_mod.Session,
    rec: *recorder_mod.Recorder,
    run: *runner_mod.InferenceRunner,

    corr: u64,
    dev_name: [MAX_DEV_NAME]u8 = undefined,
    dev_name_len: usize = 0,
};

// ── Error reporting (Python reads via vkr_last_error) ────────────────

var g_err: [512]u8 = undefined;
var g_err_len: usize = 0;

fn setErrFmt(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&g_err, fmt, args) catch return;
    g_err_len = s.len;
}

// ── Engine lifecycle ─────────────────────────────────────────────────

const Precision = enum(c_int) {
    fp32_all = 0,
    bf16_matmul = 1,
    q4_0_matmul = 2,
    q4_k_matmul = 3,
};

fn toGpuPrecision(p: Precision) gpu_model.Precision {
    return switch (p) {
        .fp32_all => .fp32_all,
        .bf16_matmul => .bf16_matmul,
        .q4_0_matmul => .q4_0_matmul,
        .q4_k_matmul => .q4_k_matmul,
    };
}

fn engineCreate(dir_path: []const u8, precision: gpu_model.Precision, max_pos: u32, max_new_tokens: u32) !*Engine {
    const gpa = std.heap.c_allocator;

    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{
        .gpa = gpa,
        .cpu = undefined,
        .tok = undefined,
        .ctx = undefined,
        .gm = undefined,
        .sess = undefined,
        .rec = undefined,
        .run = undefined,
        .corr = 1,
    };

    e.cpu = try model_mod.Model.load(gpa, dir_path);
    errdefer e.cpu.deinit();

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    e.tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    errdefer e.tok.deinit();

    e.ctx = try gpa.create(vk_mod.Context);
    errdefer gpa.destroy(e.ctx);
    e.ctx.* = try vk_mod.Context.init(gpa);
    errdefer e.ctx.deinit();

    e.gm = try gpa.create(gpu_model.GpuModel);
    errdefer gpa.destroy(e.gm);
    e.gm.* = try gpu_model.GpuModel.upload(gpa, e.ctx, &e.cpu, precision);
    errdefer e.gm.deinit(e.ctx.device);

    const sess_cfg = session_mod.Config{
        .max_new_tokens = max_new_tokens,
        .max_pos = @intCast(max_pos),
        .sampler = .greedy,
    };
    e.sess = try gpa.create(session_mod.Session);
    errdefer gpa.destroy(e.sess);
    e.sess.* = try session_mod.Session.init(gpa, e.ctx, e.gm, &e.tok, sess_cfg);
    errdefer e.sess.deinit();

    const sets_per_step: u32 = @intCast(@max(@as(usize, 512), e.cpu.config.num_hidden_layers * 32));
    e.rec = try gpa.create(recorder_mod.Recorder);
    errdefer gpa.destroy(e.rec);
    e.rec.* = try recorder_mod.Recorder.init(e.ctx, sets_per_step, 2048);
    errdefer e.rec.deinit();

    const rcfg = runner_mod.RunnerConfig{
        .default_max_tokens = max_new_tokens,
        // Blocking Python calls: let one tick chew a whole model forward.
        .default_budget = .{ .either = .{ .layers = 4096, .microseconds = 86_400_000_000 } },
    };
    e.run = try gpa.create(runner_mod.InferenceRunner);
    errdefer gpa.destroy(e.run);
    e.run.* = try runner_mod.InferenceRunner.initBorrow(gpa, e.sess, e.rec, rcfg);
    errdefer e.run.deinit();

    const dn_slice = std.mem.span(e.ctx.deviceName());
    const dn_len = @min(dn_slice.len, MAX_DEV_NAME - 1);
    @memcpy(e.dev_name[0..dn_len], dn_slice[0..dn_len]);
    e.dev_name[dn_len] = 0;
    e.dev_name_len = dn_len;

    return e;
}

fn engineFree(e: *Engine) void {
    const gpa = e.gpa;
    const device = e.ctx.device;
    e.run.deinit();
    gpa.destroy(e.run);
    e.rec.deinit();
    gpa.destroy(e.rec);
    e.sess.deinit();
    gpa.destroy(e.sess);
    e.gm.deinit(device);
    gpa.destroy(e.gm);
    e.ctx.deinit();
    gpa.destroy(e.ctx);
    e.tok.deinit();
    e.cpu.deinit();
    gpa.destroy(e);
}

fn getEngine(handle: usize) ?*Engine {
    if (handle == 0) return null;
    const idx = handle - 1;
    if (idx >= MAX_ENGINES) return null;
    return g_engines[idx];
}

// ── Generate / chat ──────────────────────────────────────────────────

fn drainToFinish(e: *Engine, out: []u8) !usize {
    var written: usize = 0;
    var idle_ticks: u32 = 0;
    const max_idle_ticks: u32 = 100;
    while (true) {
        try e.run.tickInline();
        var had_event = false;
        while (e.run.pollEvent()) |ev| {
            had_event = true;
            switch (ev.kind) {
                .token => |t| {
                    const dec = e.run.resolve(t.decoded);
                    const n = @min(dec.len, out.len - written);
                    @memcpy(out[written..][0..n], dec[0..n]);
                    written += n;
                },
                .finish => return written,
                .err => return error.RunnerFailed,
                .accepted, .arena_swap => {},
            }
        }
        if (!had_event) {
            idle_ticks += 1;
            if (idle_ticks > max_idle_ticks) return error.RunnerIdleTimeout;
        } else {
            idle_ticks = 0;
        }
    }
}

fn generateInner(e: *Engine, prompt: []const u8, out: []u8, max_new_tokens: u32) !usize {
    var msgs: [1]chat_template_mod.Message = undefined;
    msgs[0] = .{ .role = .user, .content = prompt };
    e.corr += 1;
    const cmd = runner_mod.Command{ .chat = .{
        .corr = e.corr,
        .messages = &msgs,
        .sampler = .greedy,
        .max_tokens = max_new_tokens,
    } };
    try e.run.submit(cmd);
    return drainToFinish(e, out);
}

/// `json` is a [{role, content}, ...] array; role in {system,user,assistant}.
fn chatInner(e: *Engine, json: []const u8, out: []u8, max_new_tokens: u32) !usize {
    var arena = std.heap.ArenaAllocator.init(e.gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    const arr = switch (parsed.value) {
        .array => |x| x,
        else => return error.InvalidJsonMessageArray,
    };
    if (arr.items.len == 0) return error.EmptyMessages;
    const msgs = try a.alloc(chat_template_mod.Message, arr.items.len);
    for (arr.items, 0..) |item, i| {
        const obj = switch (item) {
            .object => |o| o,
            else => return error.InvalidMessageEntry,
        };
        const rv = obj.get("role") orelse return error.MissingMessageRole;
        const cv = obj.get("content") orelse return error.MissingMessageContent;
        const role_s = switch (rv) {
            .string => |s| s,
            else => return error.InvalidMessageRole,
        };
        const content = switch (cv) {
            .string => |s| s,
            else => return error.InvalidMessageContent,
        };
        const role: chat_template_mod.Role = if (std.mem.eql(u8, role_s, "user"))
            .user
        else if (std.mem.eql(u8, role_s, "assistant"))
            .assistant
        else if (std.mem.eql(u8, role_s, "system"))
            .system
        else
            return error.UnknownMessageRole;
        msgs[i] = .{ .role = role, .content = content };
    }

    e.corr += 1;
    const cmd = runner_mod.Command{ .chat = .{
        .corr = e.corr,
        .messages = msgs,
        .sampler = .greedy,
        .max_tokens = max_new_tokens,
    } };
    try e.run.submit(cmd);
    return drainToFinish(e, out);
}

// ── Exported C ABI (consumed by valkyr/_native.py via ctypes) ────────

pub export fn vkr_version() u32 {
    return 0x0001_0000;
}

pub export fn vkr_create_engine(
    out_handle: *usize,
    model_path: [*:0]const u8,
    precision: c_int,
    max_pos: u32,
    max_new_tokens: u32,
) c_int {
    if (precision < 0 or precision > 3) {
        setErrFmt("invalid precision code {d}", .{precision});
        return -1;
    }
    var free_slot: ?usize = null;
    for (g_engines, 0..) |slot, i| {
        if (slot == null) {
            free_slot = i;
            break;
        }
    }
    const slot = free_slot orelse {
        setErrFmt("engine table full (max {d})", .{MAX_ENGINES});
        return -1;
    };

    const prec: Precision = @enumFromInt(precision);
    const e = engineCreate(std.mem.span(model_path), toGpuPrecision(prec), max_pos, max_new_tokens) catch |err| {
        setErrFmt("engine create failed: {s}", .{@errorName(err)});
        return -1;
    };
    g_engines[slot] = e;
    out_handle.* = slot + 1;
    return 0;
}

pub export fn vkr_generate(
    handle: usize,
    prompt: [*:0]const u8,
    out: [*]u8,
    out_cap: usize,
    out_len: *usize,
    max_tokens: u32,
) c_int {
    const e = getEngine(handle) orelse {
        setErrFmt("invalid engine handle {d}", .{handle});
        return -1;
    };
    out_len.* = generateInner(e, std.mem.span(prompt), out[0..out_cap], max_tokens) catch |err| {
        setErrFmt("generate failed: {s}", .{@errorName(err)});
        return -1;
    };
    return 0;
}

pub export fn vkr_chat(
    handle: usize,
    json: [*:0]const u8,
    out: [*]u8,
    out_cap: usize,
    out_len: *usize,
    max_tokens: u32,
) c_int {
    const e = getEngine(handle) orelse {
        setErrFmt("invalid engine handle {d}", .{handle});
        return -1;
    };
    out_len.* = chatInner(e, std.mem.span(json), out[0..out_cap], max_tokens) catch |err| {
        setErrFmt("chat failed: {s}", .{@errorName(err)});
        return -1;
    };
    return 0;
}

pub export fn vkr_free_engine(handle: usize) void {
    if (getEngine(handle)) |e| {
        const idx = handle - 1;
        g_engines[idx] = null;
        engineFree(e);
    }
}

pub export fn vkr_device_name(handle: usize, buf: [*]u8, cap: usize) c_int {
    const e = getEngine(handle) orelse {
        setErrFmt("invalid engine handle {d}", .{handle});
        return -1;
    };
    const n = @min(e.dev_name_len, cap);
    @memcpy(buf[0..n], e.dev_name[0..n]);
    if (n < cap) buf[n] = 0;
    return @intCast(n);
}

pub export fn vkr_last_error(buf: [*]u8, cap: usize) c_int {
    const n = @min(g_err_len, cap);
    @memcpy(buf[0..n], g_err[0..n]);
    if (n < cap) buf[n] = 0;
    return @intCast(n);
}