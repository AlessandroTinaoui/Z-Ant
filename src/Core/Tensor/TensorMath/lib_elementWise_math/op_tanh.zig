const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

/// Compute element-wise the hyperbolic tangent of the given tensor.
pub fn tanh(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    // Verify that T is among the supported types:
    // tensor(double), tensor(float), tensor(float16)
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16))) {
        @compileError("Unsupported type in tanh_lean");
    };

    // Allocating output tensor with the same shape of the input
    var result = try Tensor(T).fromShape(input.allocator, input.shape);

    try tanh_lean(T, input, &result);

    return result;
}

// --------- lean TANH
pub inline fn tanh_lean(comptime T: anytype, input: *Tensor(T), result: *Tensor(T)) !void {
    // Compute tanh(x) for each element of the tensor
    for (0..input.size) |i| {
        result.data[i] = std.math.tanh(input.data[i]);
    }
}

pub inline fn get_tanh_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try zant.utils.allocator.allocator.alloc(usize, input_shape.len);
    errdefer zant.utils.allocator.allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}

/// https://onnx.ai/onnx/operators/onnx__Tanh.html
pub fn lowerTanh(
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

    const id_tanh = b.push(.TANH, out_dtype, &.{id_loadA}, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_tanh }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
}
