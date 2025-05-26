const std = @import("std");
const zant = @import("../../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const Tensor = zant.core.tensor.Tensor; // Import Tensor type

// --------- standard CEIL
/// Compute element-wise the ceil of the given tensor.
/// If x is integral, +0, -0, NaN, or infinite, x itself is returned.
pub fn ceil(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    // Verify that T is among the supported types:
    // tensor(double), tensor(float), tensor(float16)
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16))) {
        @compileError("Unsupported type in ceil_lean");
    };

    // Allocate output tensor with the same shape as the input
    var result = try Tensor(T).fromShape(input.allocator, input.shape);

    // Perform element-wise ceil computation
    try ceil_lean(T, input, &result);

    return result;
}

// --------- lean CEIL
pub inline fn ceil_lean(comptime T: anytype, input: *Tensor(T), result: *Tensor(T)) !void {
    for (0..input.size) |i| {
        const x = input.data[i];
        if (std.math.isNan(x) or std.math.isInf(x)) {
            result.data[i] = x;
        } else {
            result.data[i] = @ceil(x);
        }
    }
}

pub fn get_ceil_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}
pub fn lowerCeil(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    out_shape: []const usize,
    out_dtype: DType, // promoted element type
) usize { // returns id of result buffer

    // ── Set-up phase ────────────────────────────────────────────────────
    _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = 1 } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── Flat element loop ───────────────────────────────────────────────
    var nelem: usize = 1;
    for (out_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

    const id_ceil = b.push(.CLIP, out_dtype, &.{id_loadA}, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_ceil }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
}
