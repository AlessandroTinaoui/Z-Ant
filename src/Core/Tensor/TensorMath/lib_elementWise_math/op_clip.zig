const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const DTypeValue = Uops.DTypeValue;
const Any = Uops.Any;

/// Returns the shape of the output tensor for the clip operation.
/// For clip operation, the output shape is identical to the input shape.
pub fn get_clip_output_shape(comptime T: type, inputTensor: *const Tensor(T), minTensor: ?*const Tensor(T), maxTensor: ?*const Tensor(T)) ![]usize {
    const allocator = inputTensor.allocator;
    _ = minTensor;
    _ = maxTensor;
    const shape = try allocator.alloc(usize, inputTensor.shape.len);
    @memcpy(shape, inputTensor.shape);
    return shape;
}

/// Clips tensor elements element-wise into the range [min_val, max_val].
/// Writes results into outputTensor. Does not perform allocations or extensive checks.
pub inline fn lean_clip(
    comptime T: type,
    inputTensor: *const Tensor(T),
    minTensor: ?*const Tensor(T),
    maxTensor: ?*const Tensor(T),
    outputTensor: *Tensor(T),
) !void {
    // Get default min/max values based on type
    const min_val = if (minTensor) |t| t.data[0] else switch (@typeInfo(T)) {
        .int => std.math.minInt(T),
        .float => -std.math.floatMax(T),
        else => @as(T, 0),
    };

    const max_val = if (maxTensor) |t| t.data[0] else switch (@typeInfo(T)) {
        .int => std.math.maxInt(T),
        .float => std.math.floatMax(T),
        else => @as(T, 0),
    };

    // Handle the case where min > max: set all values to max
    if (min_val > max_val) {
        // According to ONNX spec, if min > max, fill with min value.
        @memset(outputTensor.data, min_val);
        return;
    }

    // Use SIMD for larger sizes if applicable
    const vector_len = std.simd.suggestVectorLength(T) orelse 4;
    const use_simd = vector_len > 1 and inputTensor.size >= vector_len;

    if (use_simd) {
        // Process in chunks for better cache locality
        const chunk_size = 16; // Process 16 elements at a time
        const chunks = inputTensor.size / chunk_size;
        var i: usize = 0;

        // Process chunks
        while (i < chunks * chunk_size) : (i += chunk_size) {
            inline for (0..chunk_size) |offset| {
                const val = inputTensor.data[i + offset];
                // Apply standard clamp logic: Max(input, min) -> Min(result, max)
                const temp = @max(val, min_val);
                outputTensor.data[i + offset] = @min(temp, max_val);
            }
        }

        // Handle remaining elements
        while (i < inputTensor.size) : (i += 1) {
            const val = inputTensor.data[i];
            const temp = @max(val, min_val);
            outputTensor.data[i] = @min(temp, max_val);
        }
    } else {
        // Scalar loop for small sizes
        for (0..inputTensor.size) |i| {
            const val = inputTensor.data[i];
            // Apply standard clamp logic: Max(input, min) -> Min(result, max)
            const temp = @max(val, min_val);
            outputTensor.data[i] = @min(temp, max_val);
        }
    }
}

/// Clips tensor elements element-wise into the range [min_val, max_val].
/// Allocates and returns a new tensor. Performs input validation.
pub fn clip(
    comptime T: type,
    allocator: std.mem.Allocator,
    inputTensor: *const Tensor(T),
    minTensor: ?*const Tensor(T),
    maxTensor: ?*const Tensor(T),
) !Tensor(T) {
    // --- Checks ---
    if (inputTensor.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (minTensor) |t| {
        // Consider a tensor a scalar if its size is 1
        if (t.size != 1) return TensorMathError.InputTensorNotScalar;
        if (t.size == 0) return TensorError.EmptyTensor;
    }
    if (maxTensor) |t| {
        // Consider a tensor a scalar if its size is 1
        if (t.size != 1) return TensorMathError.InputTensorNotScalar;
        if (t.size == 0) return TensorError.EmptyTensor;
    }

    // Create a copy of the input tensor with the same shape
    var result = try Tensor(T).init(inputTensor.allocator);
    errdefer result.deinit();

    // Allocate memory for shape and data
    result.shape = try allocator.alloc(usize, inputTensor.shape.len);
    @memcpy(result.shape, inputTensor.shape);

    // Calculate total size
    var total_size: usize = 1;
    for (result.shape) |dim| {
        total_size *= dim;
    }

    // Allocate data
    result.data = try allocator.alloc(T, total_size);
    result.size = total_size;

    // Perform the clip operation
    try lean_clip(T, inputTensor, minTensor, maxTensor, &result);

    return result;
}

/// https://onnx.ai/onnx/operators/onnx__Clip.html
pub fn lowerClip(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    out_shape: []const usize,
    strideA: []const isize,
    out_dtype: DType, // promoted element type
    min: DTypeValue,
    max: DTypeValue,
) usize { // returns id of result buffer

    // ── Set-up phase ────────────────────────────────────────────────────
    _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── Flat element loop ───────────────────────────────────────────────
    var nelem: usize = 1;
    for (out_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

    const id_tanh = b.push(.CLIP, out_dtype, &.{id_loadA}, Any{ .clip_bounds = .{ .type = out_dtype, .min = min, .max = max } });

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_tanh }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
}
