const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

const Q_MIN_U8: f32 = 0.0;
const Q_MAX_U8: f32 = 255.0;

/// DynamicQuantizeLinear: Computes scale, zero point, and quantizes FP32 input to UINT8.
/// Returns an array containing [quantized_tensor, scale_tensor, zero_point_tensor].
/// The caller must free the returned array and its tensors.
pub fn dynamicQuantizeLinear(x: *Tensor(f32)) ![]*Tensor(anyopaque) {
    // 1. Get output shapes
    const output_shapes = try get_dynamicQuantizeLinear_output_shape(x.shape);
    defer {
        for (output_shapes) |shape| {
            if (shape.len > 0) { // Don't free potentially static empty shapes
                x.allocator.free(shape);
            }
        }
        x.allocator.free(output_shapes);
    }

    // 2. Allocate output tensors
    // We need 3 outputs: y (u8), y_scale (f32), y_zero_point (u8)
    const y_data = try x.allocator.alloc(u8, Tensor(u8).calculateSize(output_shapes[0]));
    errdefer x.allocator.free(y_data);
    var y = Tensor(u8){
        .allocator = x.allocator,
        .shape = &[_]usize{}, // Will be assigned later
        .data = y_data,
        .size = y_data.len,
    };

    const y_scale_data = try x.allocator.alloc(f32, Tensor(f32).calculateSize(output_shapes[1]));
    errdefer x.allocator.free(y_scale_data);
    var y_scale = Tensor(f32){
        .allocator = x.allocator,
        .shape = &[_]usize{}, // Will be assigned later
        .data = y_scale_data,
        .size = y_scale_data.len,
    };

    const y_zero_point_data = try x.allocator.alloc(u8, Tensor(u8).calculateSize(output_shapes[2]));
    errdefer x.allocator.free(y_zero_point_data);
    var y_zero_point = Tensor(u8){
        .allocator = x.allocator,
        .shape = &[_]usize{}, // Will be assigned later
        .data = y_zero_point_data,
        .size = y_zero_point_data.len,
    };

    // 3. Call the lean implementation
    try dynamicQuantizeLinear_lean(x, &y, &y_scale, &y_zero_point);

    // 4. Package results
    const results = try x.allocator.alloc(*Tensor(anyopaque), 3);
    errdefer x.allocator.free(results); // Free array if packing fails

    results[0] = @ptrCast(&y);
    results[1] = @ptrCast(&y_scale);
    results[2] = @ptrCast(&y_zero_point);

    // Transfer ownership of shapes (allocated in get_dynamicQuantizeLinear_output_shape)
    // to the output tensors. The deferred free in this function won't run.
    y.shape = output_shapes[0];
    y_scale.shape = output_shapes[1];
    y_zero_point.shape = output_shapes[2];

    // Prevent the deferred free of shapes now that ownership is transferred
    output_shapes[0] = &.{};
    output_shapes[1] = &.{};
    output_shapes[2] = &.{};

    return results;
}

/// Lean implementation of DynamicQuantizeLinear. Assumes output tensors are pre-allocated
/// with the correct shapes and memory.
pub fn dynamicQuantizeLinear_lean(
    x: *const Tensor(f32),
    y: *Tensor(u8),
    y_scale: *Tensor(f32),
    y_zero_point: *Tensor(u8),
) !void {
    if (x.size == 0) {
        // Handle empty input tensor
        y_scale.data[0] = 1.0; // Default scale
        y_zero_point.data[0] = 0; // Default zero point
        return;
    }

    // 1. Find min and max of x
    var x_min: f32 = x.data[0];
    var x_max: f32 = x.data[0];
    for (x.data) |val| {
        x_min = @min(x_min, val);
        x_max = @max(x_max, val);
    }

    // 2. Adjust range to include 0
    const x_min_adj = @min(x_min, 0.0);
    const x_max_adj = @max(x_max, 0.0);

    // 3. Calculate scale
    const x_range = x_max_adj - x_min_adj;
    const scale: f32 = if (x_range == 0) 1.0 else x_range / Q_MAX_U8; // Avoid division by zero
    y_scale.data[0] = scale;

    // 4. Calculate zero point
    const initial_zero_point_fp: f32 = if (scale == 0) 0.0 else Q_MIN_U8 - x_min_adj / scale;
    // Clip to u8 range [0, 255]
    const clipped_zero_point_fp = std.math.clamp(initial_zero_point_fp, Q_MIN_U8, Q_MAX_U8);
    // Round to nearest, ties to even (std.math.round behavior)
    const rounded_zero_point_fp = std.math.round(clipped_zero_point_fp);
    // Cast to u8
    // We assume std.math.round already produced a value within [0, 255] due to prior clipping
    const zero_point: u8 = @as(u8, @intFromFloat(@as(f32, rounded_zero_point_fp)));
    y_zero_point.data[0] = zero_point;

    // 5. Quantize x -> y
    const zero_point_f32 = @as(f32, @floatFromInt(zero_point));
    const inv_scale = if (scale == 0) 0.0 else 1.0 / scale; // Precompute inverse scale

    for (x.data, 0..) |x_val, i| {
        const x_scaled = x_val * inv_scale;
        const shifted = x_scaled + zero_point_f32;
        const rounded = std.math.round(shifted); // Round ties to even
        const clamped = std.math.clamp(rounded, Q_MIN_U8, Q_MAX_U8); // Saturate
        y.data[i] = @as(u8, @intFromFloat(clamped));
    }
}

/// Calculates the output shapes for DynamicQuantizeLinear.
/// Returns [y_shape, y_scale_shape, y_zero_point_shape].
/// The caller owns the returned shapes array and the shape slices within it.
pub fn get_dynamicQuantizeLinear_output_shape(input_shape: []const usize) ![][]usize {
    const allocator = pkg_allocator; // Use a default allocator for shapes

    // Allocate the array to hold the three shape slices
    const output_shapes = try allocator.alloc([]usize, 3);
    errdefer allocator.free(output_shapes); // Free array if subsequent allocations fail

    // 1. y_shape (same as input_shape)
    const y_shape = try allocator.dupe(usize, input_shape);
    errdefer allocator.free(y_shape);
    output_shapes[0] = y_shape;

    // 2. y_scale_shape (scalar) - Represented as shape {1} for a single value
    const scalar_shape_slice = try allocator.alloc(usize, 1);
    errdefer allocator.free(scalar_shape_slice);
    errdefer allocator.free(output_shapes[0]); // Free y_shape if scalar alloc fails
    scalar_shape_slice[0] = 1;
    output_shapes[1] = scalar_shape_slice; // y_scale_shape

    // 3. y_zero_point_shape (scalar) - Represented as shape {1}
    const zp_shape_slice = try allocator.alloc(usize, 1);
    // If this fails, free the previously allocated shapes
    errdefer allocator.free(output_shapes[1]);
    errdefer allocator.free(output_shapes[0]);
    zp_shape_slice[0] = 1;
    output_shapes[2] = zp_shape_slice; // y_zero_point_shape

    return output_shapes;
}
