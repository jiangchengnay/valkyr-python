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

// Training stack (real-model transformer, the same one `--lora-finetune` uses).
const train_transformer = vkr.train.transformer;
const train_dataset = vkr.train.dataset;
const train_load_real = vkr.train.load_real;
const train_sampling = vkr.train.sampling;

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
        // Point the Session at the model directory so ChatTemplate can
        // load the model's own Jinja `chat_template` from
        // tokenizer_config.json. Without this the Session falls back to
        // the family-hardcoded template, which fails with
        // NoStartOfTurn for models whose chat markers are absent from
        // the vocab (e.g. DeepSeek-R1: LlamaTokenizerFast under a Qwen2
        // architecture — no <|im_start|>).
        .model_dir = dir_path,
    };
    e.sess = try gpa.create(session_mod.Session);
    errdefer gpa.destroy(e.sess);
    e.sess.* = try session_mod.Session.init(gpa, e.ctx, e.gm, &e.tok, sess_cfg);
    errdefer e.sess.deinit();

    // The recorder must be large enough for the worst-case single tick.
    // `drainToFinish` runs a blocking call with `default_budget.layers`
    // = 4096, so one tick can record every layer of a full forward plus
    // the sample step. Sizing this below the budget overruns the Vulkan
    // descriptor pool, which surfaces as VK_ERROR_OUT_OF_POOL_MEMORY
    // (-1000069000) at submit time. Mirror `valkyr --serve`, which uses
    // 4096 sets / 16384 descriptors for the same blocking usage.
    e.rec = try gpa.create(recorder_mod.Recorder);
    errdefer gpa.destroy(e.rec);
    e.rec.* = try recorder_mod.Recorder.init(e.ctx, 4096, 16384);
    errdefer e.rec.deinit();

    const rcfg = runner_mod.RunnerConfig{
        .default_max_tokens = max_new_tokens,
        // Blocking Python calls: let one tick chew a whole model forward.
        // NOTE: `layers` must stay within the recorder's 4096-set pool
        // (see Recorder.init above). 4096 layers is fine for the 28-layer
        // Qwen3-0.6B but each layer dispatches several kernels, so the
        // layer cap alone does not bound the set count. Keep the µs cap
        // as the primary bound — matching `valkyr --serve`, the only
        // configuration verified end-to-end against the engine.
        .default_budget = .{ .either = .{ .layers = 8, .microseconds = 5_000 } },
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
    // Wall-clock ceiling as a backstop against a wedged GPU / runner
    // that never emits `finish`. Generous: a 28-layer model at
    // 8 layers/tick with 256 new tokens fits comfortably in minutes.
    const t_start = std.time.nanoTimestamp();
    const max_total_ns: i128 = 10 * std.time.ns_per_min;
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
        // A request in flight legitimately produces long stretches of
        // eventless ticks — the whole prefill phase (embedding every
        // prompt token, one tick per 8 layers / 5 ms) emits nothing,
        // so a long multi-turn prompt can take hundreds of eventless
        // ticks before the first token. Counting those as "idle" made
        // every long-prompt chat time out spuriously. Only count idle
        // when the runner is truly idle (no active request AND nothing
        // queued); that is the only state in which it will never emit.
        const busy = e.run.active != null or e.run.backlog.items.len > 0;
        if (!had_event and !busy) {
            idle_ticks += 1;
            if (idle_ticks > max_idle_ticks) return error.RunnerIdleTimeout;
        } else {
            idle_ticks = 0;
        }
        const now = std.time.nanoTimestamp();
        if (now - t_start > max_total_ns) return error.RunnerTimedOut;
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

/// Mean per-position cross-entropy (mirror of the helper in
/// commands/finetune.zig — fp64 accumulation for the per-position log-Z).
fn computeMeanCe(logits: []const f32, target_ids: []const u32, n_pos: u32, vocab: usize) f32 {
    var ce_sum: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        ce_sum += log_z - @as(f64, logits[off + target_ids[p]]);
    }
    return @floatCast(ce_sum / @as(f64, @floatFromInt(n_pos)));
}

/// Vulkan LoRA fine-tuning through the Python bindings — the same
/// training stack as `valkyr --lora-finetune`. Uses the engine already
/// loaded behind `handle` (CPU model + tokenizer), materializes fp32
/// train tensors, packs `examples_json` (a JSON array of strings) into
/// a dataset, runs `steps` Adam steps on the GPU, and saves a `.lvkpt`
/// LoRA checkpoint. A human-readable training summary (CE before/after,
/// ms/step, optional probe before/after) is written to `out`.
///
/// Architectural note: the trainer supports full-attention models
/// WITHOUT attention bias (Qwen3-class; Qwen2/Qwen2.5 have bias and are
/// rejected with AttentionBiasNotSupported). Q/K-norm is optional.
pub export fn vkr_lora_train(
    handle: usize,
    examples_json: [*:0]const u8,
    out_path: [*:0]const u8,
    steps: u32,
    lr: f32,
    rank: u32,
    alpha: f32,
    n_pos: u32,
    targets_mask: u32,
    rotate: c_int,
    probe: [*:0]const u8,
    n_gen: u32,
    out: [*]u8,
    out_cap: usize,
    out_len: *usize,
) c_int {
    const e = getEngine(handle) orelse {
        setErrFmt("invalid engine handle {d}", .{handle});
        return -1;
    };
    out_len.* = loraTrainInner(e, std.mem.span(examples_json), std.mem.span(out_path), steps, lr, rank, alpha, n_pos, targets_mask, rotate != 0, std.mem.span(probe), n_gen, out[0..out_cap]) catch |err| {
        setErrFmt("lora train failed: {s}", .{@errorName(err)});
        return -1;
    };
    return 0;
}

fn loraTrainInner(
    e: *Engine,
    examples_json: []const u8,
    out_path: []const u8,
    steps: u32,
    lr: f32,
    rank: u32,
    alpha: f32,
    n_pos: u32,
    targets_mask: u32,
    rotate: bool,
    probe: []const u8,
    n_gen: u32,
    out: []u8,
) !usize {
    // ── Parse the examples array (["text", ...]).
    var arena = std.heap.ArenaAllocator.init(e.gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = try std.json.parseFromSlice(std.json.Value, a, examples_json, .{});
    const arr = switch (parsed.value) {
        .array => |x| x,
        else => return error.ExamplesMustBeArray,
    };
    if (arr.items.len == 0) return error.EmptyExamples;
    const examples = try a.alloc([]const u8, arr.items.len);
    for (arr.items, 0..) |item, i| {
        examples[i] = switch (item) {
            .string => |s| s,
            else => return error.ExampleMustBeString,
        };
    }

    // ── fp32 train tensors from the already-loaded CPU model.
    var weights = try train_load_real.loadTrainWeights(e.gpa, &e.cpu, n_pos);
    defer weights.deinit();
    var cfg = weights.cfg;
    cfg.lr = lr;
    cfg.lora_targets = targets_mask;
    cfg.lora_rank = rank;
    cfg.lora_alpha = alpha;

    // ── GPU bring-up (training context is separate from inference).
    var ctx = try vk_mod.Context.init(e.gpa);
    defer ctx.deinit();
    var runner = try train_transformer.Runner.init(e.gpa, &ctx, cfg, weights.view());
    defer runner.deinit();

    // ── Dataset: pack examples with the session's end-of-turn token.
    const eot: u32 = e.sess.template.end_of_turn;
    var ds = try train_dataset.buildFromExamples(e.gpa, &e.tok, examples, n_pos, eot);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;

    const vocab: usize = cfg.vocab_size;
    const logits = try e.gpa.alloc(f32, @as(usize, n_pos) * vocab);
    defer e.gpa.free(logits);

    var summary = std.ArrayList(u8).init(e.gpa);
    defer summary.deinit();
    const sw = summary.writer();

    // ── Probe sample before training (optional).
    var probe_pre_text: ?[]const u8 = null;
    var probe_ids: ?[]u32 = null;
    defer if (probe_pre_text) |t| e.gpa.free(t);
    defer if (probe_ids) |ids| e.gpa.free(ids);
    if (probe.len > 0) {
        const ids = try e.tok.encode(e.gpa, probe);
        if (ids.len == 0 or ids.len >= n_pos) {
            e.gpa.free(ids);
            try sw.print("WARN probe tokenizes to > n_pos; skipping probe\n", .{});
        } else {
            probe_ids = ids;
            const pre = try train_sampling.greedyDecode(e.gpa, &runner, ids, n_gen, n_pos, vocab, eot);
            defer e.gpa.free(pre);
            probe_pre_text = try train_sampling.decodeIdsToText(e.gpa, &e.tok, pre);
            try sw.print("probe before: {s}\n", .{probe_pre_text.?});
        }
    }

    // ── Initial CE on batch 0.
    const input_ids = try e.gpa.alloc(u32, n_pos);
    defer e.gpa.free(input_ids);
    const target_ids = try e.gpa.alloc(u32, n_pos);
    defer e.gpa.free(target_ids);
    try ds.batch(0, input_ids, target_ids);
    try runner.forwardLogits(input_ids, logits);
    const ce_init = computeMeanCe(logits, target_ids, n_pos, vocab);
    try sw.print("CE initial: {d:.4} (batch 0, {d} examples)\n", .{ ce_init, examples.len });

    // ── Training loop.
    const t_start = std.time.nanoTimestamp();
    var step: u32 = 0;
    while (step < steps) : (step += 1) {
        const b: usize = if (rotate)
            @intCast(@mod(@as(usize, step), ds.numBatches()))
        else
            0;
        try ds.batch(b, input_ids, target_ids);
        try runner.step(input_ids, target_ids);
    }
    const t_end = std.time.nanoTimestamp();
    const train_ms: f64 = @as(f64, @floatFromInt(t_end - t_start)) / 1.0e6;
    const ms_per_step: f64 = train_ms / @as(f64, @floatFromInt(@max(steps, 1)));

    // ── Final CE on batch 0.
    try ds.batch(0, input_ids, target_ids);
    try runner.forwardLogits(input_ids, logits);
    const ce_final = computeMeanCe(logits, target_ids, n_pos, vocab);
    const drop_pct: f64 = if (ce_init > 0) 100.0 * (@as(f64, ce_init) - @as(f64, ce_final)) / @as(f64, ce_init) else 0.0;
    try sw.print("CE final: {d:.4} (drop {d:.2}%, {d:.1} ms/step)\n", .{ ce_final, drop_pct, ms_per_step });

    // ── Probe sample after training.
    if (probe_ids) |ids| {
        const post = try train_sampling.greedyDecode(e.gpa, &runner, ids, n_gen, n_pos, vocab, eot);
        defer e.gpa.free(post);
        const post_text = try train_sampling.decodeIdsToText(e.gpa, &e.tok, post);
        defer e.gpa.free(post_text);
        try sw.print("probe after:  {s}\n", .{post_text});
    }

    // ── Save the LoRA checkpoint.
    try runner.saveLoraCheckpoint(e.gpa, out_path);
    try sw.print("saved .lvkpt: {s}\n", .{out_path});

    const result = summary.items;
    const n = @min(result.len, out.len);
    @memcpy(out[0..n], result[0..n]);
    return n;
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