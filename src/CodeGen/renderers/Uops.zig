//! ============================================================================
//!  Zant IR – *single–assignment* micro–operation layer
//! ============================================================================
//!  This file is meant to live in `src/ir.zig`.
//!  It defines **everything** a back-end needs to know about one UOp:
//!    • the tag (`UOpType`)                     – *what* the op does
//!    • the element type (`DType`)              – *with which* scalar type
//!    • its operands (`src` as indices)         – *who* produces the inputs
//!    • an optional payload (`Any`)             – *extra metadata*
//!
//!
//!  ─ SSA (Static Single Assignment) ───────────────────────────────────────
//!  •  Each `UOp` appears **once** in the slice and never mutates.
//!  •  Later ops reference earlier ones by *index* (the `src` array).
//!  •  Guarantees there is exactly **one definition** per temporary, which
//!     simplifies constant propagation, alias analysis, and code-gen.
//!
//! ============================================================================

const std = @import("std");

// ─────────────────────────────────────────────────────────────────────────────
// 1. UOpType – every primitive micro-op in Zant
//    (Comments give a one-line intuitive meaning.)
// ─────────────────────────────────────────────────────────────────────────────
pub const UOpType = enum {
    // Data movement / buffer mgmt
    DEFINE_GLOBAL, // allocate a top-level tensor/buffer (inputs, outputs)
    DEFINE_ACC, // allocate and zero a reduction accumulator
    LOAD, // read one element from memory
    STORE, // write one element to memory
    CONST, // scalar literal (f32/i32/…)

    // Pure arithmetic & logical ops (element-wise)
    ADD,
    SUB,
    MUL,
    FDIV,
    POW,
    EXP2,
    NEG,
    MAX,
    MIN,
    CLIP, // CLIP clamps to [min,max] (limits in Any)
    CMPLT, // compare <  (returns bool)
    WHERE, // ternary select (cond ? a : b)
    MULACC, // fused multiply-add into an accumulator x += y*z

    // Reductions
    REDUCE_ADD,
    REDUCE_MAX,

    // Loop / control
    RANGE, // begin counted loop   (bounds in Any.loop_bounds)
    ENDRANGE, // end   counted loop
    IF, //we already have where keep it or not ?
    ENDIF,

    // Addressing & view manipulation (no data copies)
    GEP, // Get element pointer GEP calculates the actual position inside a flat memory buffer,
    //taking into account the strides of each dimension — including broadcasting.
    VIEW, // Create a view of a tensor
    COPY, // Copy a tensor
    RESHAPE, // Reshape a tensor
    PAD, // Pad a tensor
    PERMUTE, // Permute a tensor
    EXPAND, // Expand a tensor

    // Shape & bookkeeping
    SHAPE, // Get the shape of a tensor
    CAST, // Cast a tensor to a different type

    // Scheduling hints (inserted by auto-tuner)
    TILE_M, // Tile the tensor in the M dimension
    TILE_N, // Tile the tensor in the N dimension
    VECTORIZE, // Vectorize the tensor
    UNROLL_K, // Unroll the tensor in the K dimension

    // Graph–level utilities
    FUSE, // marks a fused element-wise chain (payload = ops list)

};

pub const GEPInfo = struct {
    base: usize, // base UOp ID
    offset: usize, // offset from base
    stride: usize, // stride for the GEP operation
};
// ─────────────────────────────────────────────────────────────────────────────
// 2. Any – single-slot, type-safe payload attached to `UOp.arg`
// ─────────────────────────────────────────────────────────────────────────────
pub const Any = union(enum) {
    // ── 2 · 1  Scalar immediates ────────────────────────────────────────
    int: usize,
    float: f32,
    bool: bool,

    // ── 2 · 2  Tiny metadata blobs ──────────────────────────────────────
    label: []const u8,
    shape: []const usize, // runtime shape vector

    // ── 2 · 3  Control–flow helpers ─────────────────────────────────────
    loop_bounds: struct { // • used by RANGE / ENDRANGE
        start: usize,
        end: usize,
    },

    // ── 2 · 3 · 1 View based range ──────────────────────────────────────────────
    loop_bounds_view: struct {
        start: usize,
        end: usize,
        stride_index: usize,
        view_id: usize,
    },

    // ── 2 · 4  Addressing info ──────────────────────────────────────────
    mem_info: struct { // • used by GEP
        base: usize, // base UOp ID
        offset: usize, // offset from base
        stride: usize, // stride for the GEP operation
    },

    mem_info_gep_info: GEPInfo,

    // ── 2 · 5  NEW ──────────────────────────────────────
    /// Carries **both** the logical shape and the per-dimensional strides
    /// (stride == 0 means "broadcast this dimension").
    view_meta: struct { // • used by VIEW
        shape: []const usize,
        strides: []const isize,
    },

    cast_meta: struct {
        to: DType, // target scalar tSype
        saturate: bool, // obey float-8 saturation tables (opset-23 attr)
    },

    clip_bounds: struct {
        type: DType,
        min: DTypeValue,
        max: DTypeValue,
    },

    stride_id: usize,

    // 👉  add more variants when a new op requires metadata
};

pub const DTypeValue = union(DType) {
    f32: f32,
    i32: i32,
    i8: i8,
    bool: bool,
    u16: u16,

    pub fn getDType(self: DTypeValue) DType {
        return switch (self) {
            DTypeValue.f32 => DType.f32,
            DTypeValue.i32 => DType.i32,
            DTypeValue.i8 => DType.i8,
            DTypeValue.bool => DType.bool,
            DTypeValue.u16 => DType.u16,
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. DType – minimalist scalar element types
// ─────────────────────────────────────────────────────────────────────────────
pub const DType = enum { f32, i32, i8, bool, u16 };

pub const DTypeInfo = struct {
    pub fn asString(dtype: DType) []const u8 {
        return switch (dtype) {
            .f32 => "f32",
            .i32 => "i32",
            .i8 => "i8",
            .bool => "bool",
            .u16 => "u16",
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. UOp – ONE single-assignment micro-op stored in a linear slice
// ─────────────────────────────────────────────────────────────────────────────
pub const UOp = struct {
    id: usize, // equals position in program slice (redundant but handy)
    op: UOpType, // opcode tag
    dtype: DType, // element type of the result
    src: []const usize, // producer IDs; each ID < id  (topologically sorted)
    arg: ?Any = null, // optional payload (constants, bounds, …)

    /// Pretty-print for REPL / unit tests
    pub fn dump(self: UOp, w: anytype) !void {
        try w.print("{d:>3}  {s}", .{ self.id, @tagName(self.op) });
        if (self.src.len > 0) try w.print("  src={any}", .{self.src});
        if (self.arg) |a| try w.print("  arg={any}", .{a});
        try w.print("\n", .{});
    }
};

pub const UOpBuilder = struct {
    alloc: std.mem.Allocator,
    list: std.ArrayList(UOp),

    pub fn init(a: std.mem.Allocator) UOpBuilder {
        return .{ .alloc = a, .list = .init(a) };
    }

    /// Push that dupes `src` safely.
    /// NEW: Also dupes arg.view_meta.shape and arg.view_meta.strides for VIEW ops.
    pub fn push(self: *UOpBuilder, op: UOpType, dt: DType, src: []const usize, arg: ?Any) usize {
        const id = self.list.items.len;

        // Duplicate src slice
        const src_copy = if (src.len == 0)
            &[_]usize{} // empty slice → static, no alloc
        else
            self.alloc.dupe(usize, src) catch unreachable;

        // Handle arg duplication based on op type
        var final_arg = arg;
        if (arg) |arg_val| {
            // Use switch for type-safe union payload access
            if (op == .VIEW) {
                switch (arg_val) {
                    .view_meta => |vm| {
                        // Duplicate shape and strides for VIEW ops
                        const shape_copy = if (vm.shape.len == 0) &[_]usize{} else self.alloc.dupe(usize, vm.shape) catch unreachable;
                        const strides_copy = if (vm.strides.len == 0) &[_]isize{} else self.alloc.dupe(isize, vm.strides) catch unreachable;
                        // Create a new Any with the copied slices
                        final_arg = Any{ .view_meta = .{ .shape = shape_copy, .strides = strides_copy } };
                    },
                    else => {}, // VIEW op with unexpected arg type? Ignore for now.
                }
            }
            // Add other cases here if other ops have args needing duplication
            // else if (op == .SOME_OTHER_OP) { ... }
        }

        // Append the UOp with copied src and potentially copied arg contents
        self.list.append(.{ .id = id, .op = op, .dtype = dt, .src = src_copy, .arg = final_arg }) catch unreachable;
        return id;
    }

    /// Transfer ownership of the slice (caller must later free each src* AND specific arg* payloads)
    pub fn toOwnedSlice(self: *UOpBuilder) ![]UOp {
        const owned_slice = try self.list.toOwnedSlice();
        // Reset the builder's list to prevent double-free in deinit
        self.list = std.ArrayList(UOp).init(self.alloc);
        return owned_slice;
    }

    /// Free every `src` slice + the array buffer itself.
    /// NEW: Also frees duplicated arg payloads (currently only view_meta shape/strides).
    pub fn deinit(self: *UOpBuilder) void {
        std.debug.print("DEBUG: UOpBuilder.deinit freeing {d} uops\n", .{self.list.items.len});
        for (self.list.items) |uop| {
            // Free src (only if non-empty)
            if (uop.src.len > 0) {
                self.alloc.free(@constCast(uop.src));
            }
            // Free duplicated arg payloads (only if non-null and relevant type)
            if (uop.arg) |arg_val| {
                // Use switch for type-safe union payload access
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            // Only free if non-empty
                            if (vm.shape.len > 0) self.alloc.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) self.alloc.free(@constCast(vm.strides));
                        },
                        else => {}, // VIEW op with unexpected arg type? Ignore.
                    }
                }
                // Add else if for other duplicated args
                // else if (uop.op == .SOME_OTHER_OP) { ... }
            }
        }
        self.list.deinit();
    }
};

// Team 1 ZantSyntaxTree - From Onnx to ZantSyntaxTree
// Mirko
// Pietro
// Filippo

// ---Optimization--- Constant folding kernel fusion ---

// Team 2 IR - From High Level IR Math to Low Level IR Math (e.g.LowerRelu LowerAdd)
// Marco
// Mattia
// Alessandro/Adriano

// Team 3 Renderer -From Uops to zig code
// Burak
// Matteo
// Alessandro/Adriano
