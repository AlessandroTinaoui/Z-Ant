const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

pub const AutoPadType = enum {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

pub fn get_pooling_output_shape(input_shape: []const usize, kernel: [2]usize, stride: [2]usize) ![4]usize {
    // std.log.debug("\n[DEBUG] get_pooling_output_shape - Input shape: {any}", .{input_shape});
    // std.log.debug("\n[DEBUG] get_pooling_output_shape - Kernel: {any}", .{kernel});
    // std.log.debug("\n[DEBUG] get_pooling_output_shape - Stride: {any}", .{stride});

    if (input_shape.len != 4) {
        std.log.warn("\n[DEBUG] ERROR: Invalid dimensions - input shape length is not 4", .{});
        return TensorMathError.InvalidDimensions;
    }

    if (kernel[0] == 0 or kernel[1] == 0 or stride[0] == 0 or stride[1] == 0) {
        std.log.warn("\n[DEBUG] ERROR: Wrong stride - kernel or stride contains zeros", .{});
        return TensorMathError.WrongStride;
    }

    if (kernel[0] > input_shape[2] or kernel[1] > input_shape[3]) {
        std.log.warn("\n[DEBUG] WARNING: Kernel size ({},{}) > input dimensions ({},{})", .{ kernel[0], kernel[1], input_shape[2], input_shape[3] });

        // SPECIAL CASE: Instead of error, return a shape with 1x1 spatial dimensions
        // when kernel is larger than input
        std.log.debug("\n[DEBUG] Using special case: returning output with 1x1 spatial dimensions", .{});
        const batch_size = input_shape[0];
        const channels = input_shape[1];

        return [4]usize{ batch_size, channels, 1, 1 };
    }

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const out_height = (input_shape[2] - kernel[0]) / stride[0] + 1;
    const out_width = (input_shape[3] - kernel[1]) / stride[1] + 1;

    // std.log.debug("\n[DEBUG] Calculated output dimensions - height: {}, width: {}", .{ out_height, out_width });
    return [4]usize{ batch_size, channels, out_height, out_width };
}

// POOLING -----------------------------------------------------------------------------------------------------------------------
pub fn pool_tensor(
    comptime T: type,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !Tensor(T) {
    var outputTensorShape = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(outputTensorShape);

    for (0..input.shape.len - 2) |i| {
        outputTensorShape[i] = input.shape[i];
    }
    const height = input.shape.len - 2;
    const width = input.shape.len - 1;

    outputTensorShape[height] = (input.shape[height] - kernel[0] + 1) / stride[0];
    outputTensorShape[width] = (input.shape[width] - kernel[1] + 1) / stride[1];

    var output = try Tensor(T).fromShape(&pkg_allocator, outputTensorShape);
    errdefer output.deinit();

    const location = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(location);
    @memset(location, 0);

    try multidim_pooling(
        T,
        input,
        used_windows,
        &output,
        0,
        location,
        kernel,
        stride,
        poolingType,
    );

    return output;
}

pub fn multidim_pooling(
    comptime T: anytype,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    output: *Tensor(T),
    current_depth: usize,
    location: []usize,
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !void {
    if (current_depth == output.shape.len - 2) {
        const allocator = pkg_allocator;

        var temp_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(temp_location);
        var window_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(window_location);
        var window_values = try allocator.alloc(T, kernel[0] * kernel[1]);
        defer allocator.free(window_values);

        var output_row_counter: usize = 0;
        var output_col_counter: usize = 0;

        @memcpy(temp_location, location);
        @memcpy(window_location, location);

        temp_location[current_depth] = 0;
        temp_location[current_depth + 1] = 0;

        const input_rows = input.shape[current_depth];
        const input_cols = input.shape[current_depth + 1];
        const output_cols = output.shape[current_depth + 1];

        while (temp_location[current_depth] + kernel[0] <= input_rows) : (temp_location[current_depth] += stride[0]) {
            temp_location[current_depth + 1] = 0; // Reset column counter
            while (temp_location[current_depth + 1] + kernel[1] <= input_cols) : (temp_location[current_depth + 1] += stride[1]) {
                const kernel_rows = kernel[0];
                const kernel_cols = kernel[1];
                const w = output_row_counter * output_cols + output_col_counter;

                // Reset window location for each window
                @memcpy(window_location, temp_location);

                var window_idx: usize = 0;
                for (0..kernel_rows) |i| {
                    for (0..kernel_cols) |j| {
                        window_location[current_depth] = temp_location[current_depth] + i;
                        window_location[current_depth + 1] = temp_location[current_depth + 1] + j;
                        window_values[window_idx] = try input.get_at(window_location);
                        window_idx += 1;
                    }
                }

                switch (poolingType) {
                    .Max => {
                        var max = window_values[0];
                        var max_idx: usize = 0;
                        for (window_values, 0..) |val, i| {
                            if (val > max) {
                                max = val;
                                max_idx = i;
                            }
                        }

                        const row = max_idx / kernel_cols;
                        const col = max_idx % kernel_cols;
                        const max_r = temp_location[current_depth] + row;
                        const max_c = temp_location[current_depth + 1] + col;

                        used_windows.data[w * (input_rows * input_cols) + max_r * input_cols + max_c] = 1;

                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, max);
                    },
                    .Min => {
                        var min = window_values[0];
                        var min_idx: usize = 0;
                        for (window_values, 0..) |val, i| {
                            if (val < min) {
                                min = val;
                                min_idx = i;
                            }
                        }

                        const row = min_idx / kernel_cols;
                        const col = min_idx % kernel_cols;
                        const min_r = temp_location[current_depth] + row;
                        const min_c = temp_location[current_depth + 1] + col;

                        used_windows.data[w * (input_rows * input_cols) + min_r * input_cols + min_c] = 1;

                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, min);
                    },
                    .Avg => {
                        var sum: T = 0;
                        for (window_values) |val| {
                            sum += val;
                        }
                        const avg = sum / @as(T, @floatFromInt(window_values.len));

                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, avg);

                        // Mark all values in window as used for average pooling
                        for (0..kernel_rows) |i| {
                            for (0..kernel_cols) |j| {
                                const avg_r = temp_location[current_depth] + i;
                                const avg_c = temp_location[current_depth + 1] + j;
                                used_windows.data[w * (input_rows * input_cols) + avg_r * input_cols + avg_c] = 1;
                            }
                        }
                    },
                }

                output_col_counter += 1;
            }
            output_row_counter += 1;
            output_col_counter = 0;
        }
    } else {
        for (0..output.shape[current_depth]) |element_at_current_depth| {
            location[current_depth] = element_at_current_depth;
            try multidim_pooling(
                T,
                input,
                used_windows,
                output,
                current_depth + 1,
                location,
                kernel,
                stride,
                poolingType,
            );
        }
    }
}

/// Performs pooling operation on a 4D tensor (batch, channels, height, width)
pub fn pool_forward(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, poolingType: PoolingType) !struct { output: Tensor(T), used_input: Tensor(u8) } {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const input_rows = input.shape[2];
    const input_cols = input.shape[3];

    // Validate input dimensions
    if (input.shape.len != 4) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // Validate kernel and stride
    if (kernel[0] == 0 or kernel[1] == 0 or stride[0] == 0 or stride[1] == 0) {
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output dimensions
    const out_rows = (input_rows -% kernel[0]) / stride[0] +% 1;
    const out_cols = (input_cols -% kernel[1]) / stride[1] +% 1;

    // Validate output dimensions
    if (out_rows == 0 or out_cols == 0 or out_rows > input_rows or out_cols > input_cols) {
        return TensorMathError.InvalidDimensions;
    }

    // Allocate output tensor
    var output_shape = [_]usize{ batch_size, channels, out_rows, out_cols };
    var output = try Tensor(T).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    // Allocate used_input tensor
    var used_input_shape = [_]usize{ batch_size, channels, out_rows * out_cols, input_rows, input_cols };
    var used_input = try Tensor(u8).fromShape(&pkg_allocator, &used_input_shape);
    errdefer used_input.deinit();
    @memset(used_input.data, 0);

    // Main pooling loop
    for (0..batch_size) |b| {
        for (0..channels) |c| {
            const batch_offset = b * channels * input_rows * input_cols;
            const channel_offset = c * input_rows * input_cols;
            const out_batch_offset = b * channels * out_rows * out_cols;
            const out_channel_offset = c * out_rows * out_cols;

            for (0..out_rows) |out_r| {
                const r_start = out_r * stride[0];
                if (r_start >= input_rows) continue;

                for (0..out_cols) |out_c| {
                    const c_start = out_c * stride[1];
                    if (c_start >= input_cols) continue;

                    const window_index = out_r * out_cols + out_c;
                    const out_idx = out_batch_offset + out_channel_offset + out_r * out_cols + out_c;

                    switch (poolingType) {
                        .Max => {
                            var max_value: T = -std.math.floatMax(T);
                            var max_pos: usize = 0;
                            var found_valid = false;

                            // Scan kernel window
                            for (0..kernel[0]) |kr| {
                                const in_r = r_start + kr;
                                if (in_r >= input_rows) break;

                                for (0..kernel[1]) |kc| {
                                    const in_c = c_start + kc;
                                    if (in_c >= input_cols) break;

                                    const idx = batch_offset + channel_offset + in_r * input_cols + in_c;
                                    if (idx >= input.data.len) break;

                                    const val = input.data[idx];
                                    if (!found_valid or val > max_value) {
                                        max_value = val;
                                        max_pos = in_r * input_cols + in_c;
                                        found_valid = true;
                                    }
                                }
                            }

                            // Set output value
                            if (out_idx >= output.data.len) {
                                output.deinit();
                                used_input.deinit();
                                return TensorMathError.InvalidDimensions;
                            }
                            output.data[out_idx] = if (found_valid) max_value else 0;

                            // Mark used input
                            if (found_valid) {
                                const used_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                    c * out_rows * out_cols * input_rows * input_cols +
                                    window_index * input_rows * input_cols + max_pos;
                                if (used_idx >= used_input.data.len) {
                                    output.deinit();
                                    used_input.deinit();
                                    return TensorMathError.InvalidDimensions;
                                }
                                used_input.data[used_idx] = 1;
                            }
                        },
                        .Min => {
                            var min_value: T = if (T == f64) std.math.inf(f64) else std.math.floatMax(T);
                            var min_pos: usize = 0;
                            var found_valid = false;

                            for (0..kernel[0]) |kr| {
                                for (0..kernel[1]) |kc| {
                                    const in_r = r_start + kr;
                                    const in_c = c_start + kc;

                                    if (in_r < input_rows and in_c < input_cols) {
                                        const idx = b * channels * input_rows * input_cols +
                                            c * input_rows * input_cols +
                                            in_r * input_cols + in_c;
                                        const val = input.data[idx];
                                        if (!found_valid or val < min_value) {
                                            min_value = val;
                                            min_pos = in_r * input_cols + in_c;
                                            found_valid = true;
                                        }
                                    }
                                }
                            }

                            if (!found_valid) {
                                min_value = 0;
                            }

                            if (out_idx >= output.data.len) {
                                output.deinit();
                                used_input.deinit();
                                return TensorMathError.InvalidDimensions;
                            }
                            output.data[out_idx] = min_value;

                            if (found_valid) {
                                const used_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                    c * out_rows * out_cols * input_rows * input_cols +
                                    window_index * input_rows * input_cols + min_pos;
                                if (used_idx >= used_input.data.len) {
                                    output.deinit();
                                    used_input.deinit();
                                    return TensorMathError.InvalidDimensions;
                                }
                                used_input.data[used_idx] = 1;
                            }
                        },
                        .Avg => {
                            var sum: T = 0;
                            var count: usize = 0;

                            for (0..kernel[0]) |kr| {
                                for (0..kernel[1]) |kc| {
                                    const in_r = r_start + kr;
                                    const in_c = c_start + kc;

                                    if (in_r < input_rows and in_c < input_cols) {
                                        const idx = b * channels * input_rows * input_cols +
                                            c * input_rows * input_cols +
                                            in_r * input_cols + in_c;
                                        sum += input.data[idx];
                                        count += 1;

                                        const used_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                            c * out_rows * out_cols * input_rows * input_cols +
                                            window_index * input_rows * input_cols +
                                            in_r * input_cols + in_c;
                                        if (used_idx >= used_input.data.len) {
                                            output.deinit();
                                            used_input.deinit();
                                            return TensorMathError.InvalidDimensions;
                                        }
                                        used_input.data[used_idx] = 1;
                                    }
                                }
                            }
                            const avg = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;
                            if (out_idx >= output.data.len) {
                                output.deinit();
                                used_input.deinit();
                                return TensorMathError.InvalidDimensions;
                            }
                            output.data[out_idx] = avg;
                        },
                    }
                }
            }
        }
    }

    return .{ .output = output, .used_input = used_input };
}

/// Performs backward pass for pooling operation
pub fn pool_backward(comptime T: type, dValues: *Tensor(T), input_shape: []const usize, used_input: *Tensor(u8), kernel: [2]usize, stride: [2]usize) !Tensor(T) {
    const shape = try pkg_allocator.alloc(usize, input_shape.len);
    defer pkg_allocator.free(shape);
    @memcpy(shape, input_shape);

    var dInput = try Tensor(T).fromShape(&pkg_allocator, shape);
    errdefer dInput.deinit();

    @memset(dInput.data, 0);

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const input_rows = input_shape[2];
    const input_cols = input_shape[3];

    const out_rows = dValues.shape[2];
    const out_cols = dValues.shape[3];

    for (0..batch_size) |b| {
        for (0..channels) |c| {
            for (0..out_rows) |out_r| {
                for (0..out_cols) |out_c| {
                    const grad_idx = b * channels * out_rows * out_cols +
                        c * out_rows * out_cols +
                        out_r * out_cols + out_c;
                    const grad = dValues.data[grad_idx];
                    const window_index = out_r * out_cols + out_c;

                    for (0..kernel[0]) |kr| {
                        for (0..kernel[1]) |kc| {
                            const in_r = out_r * stride[0] + kr;
                            const in_c = out_c * stride[1] + kc;

                            if (in_r < input_rows and in_c < input_cols) {
                                const mask_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                    c * out_rows * out_cols * input_rows * input_cols +
                                    window_index * input_rows * input_cols +
                                    in_r * input_cols + in_c;
                                const mask_val = used_input.data[mask_idx];

                                if (mask_val == 1) {
                                    const input_idx = b * channels * input_rows * input_cols +
                                        c * input_rows * input_cols +
                                        in_r * input_cols + in_c;
                                    dInput.data[input_idx] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dInput;
}

pub fn get_onnx_maxpool_output_shape(
    input_shape: []const usize,
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
) ![]usize {
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - Input shape: {any}", .{input_shape});
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - Kernel shape: {any}", .{kernel_shape});
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - Strides: {any}", .{strides});
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - Dilations: {any}", .{dilations});
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - Pads: {any}", .{pads});
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - AutoPad: {}", .{auto_pad});
    // std.log.debug("\n[DEBUG] get_onnx_maxpool_output_shape - Ceil mode: {}", .{ceil_mode});

    if (input_shape.len != 4) {
        std.log.warn("\n[DEBUG] ERROR: Invalid dimensions - input shape length is not 4", .{});
        return TensorMathError.InvalidDimensions;
    }

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const input_height = input_shape[2];
    const input_width = input_shape[3];

    // Handle special case: kernel larger than input
    if (kernel_shape[0] > input_height or kernel_shape[1] > input_width) {
        std.log.warn("\n[DEBUG] WARNING: Kernel size ({},{}) > input dimensions ({},{})", .{ kernel_shape[0], kernel_shape[1], input_height, input_width });
        std.log.debug("\n[DEBUG] Using special case: returning output with 1x1 spatial dimensions", .{});

        var output_shape = try pkg_allocator.alloc(usize, 4);
        output_shape[0] = batch_size;
        output_shape[1] = channels;
        output_shape[2] = 1;
        output_shape[3] = 1;
        return output_shape;
    }

    var output_shape = try pkg_allocator.alloc(usize, 4);
    errdefer pkg_allocator.free(output_shape);

    output_shape[0] = batch_size;
    output_shape[1] = channels;

    var pad_h: usize = 0;
    var pad_w: usize = 0;
    var pad_top: usize = 0;
    var pad_left: usize = 0;
    var pad_bottom: usize = 0;
    var pad_right: usize = 0;

    // Calculate effective kernel size with dilations
    const effective_kernel_h = (kernel_shape[0] - 1) * dilations[0] + 1;
    const effective_kernel_w = (kernel_shape[1] - 1) * dilations[1] + 1;

    switch (auto_pad) {
        .NOTSET => {
            pad_top = pads[0];
            pad_left = pads[1];
            pad_bottom = pads[2];
            pad_right = pads[3];
            pad_h = pad_top + pad_bottom;
            pad_w = pad_left + pad_right;
        },
        .VALID => {
            pad_h = 0;
            pad_w = 0;
            pad_top = 0;
            pad_left = 0;
            pad_bottom = 0;
            pad_right = 0;
        },
        .SAME_UPPER, .SAME_LOWER => {
            // Calculate total padding needed
            const out_height = @divTrunc(input_height + strides[0] - 1, strides[0]);
            const out_width = @divTrunc(input_width + strides[1] - 1, strides[1]);

            pad_h = (out_height - 1) * strides[0] + effective_kernel_h - input_height;
            pad_w = (out_width - 1) * strides[1] + effective_kernel_w - input_width;

            if (auto_pad == .SAME_UPPER) {
                pad_top = pad_h / 2;
                pad_bottom = pad_h - pad_top;
                pad_left = pad_w / 2;
                pad_right = pad_w - pad_left;
            } else {
                pad_bottom = pad_h / 2;
                pad_top = pad_h - pad_bottom;
                pad_right = pad_w / 2;
                pad_left = pad_w - pad_right;
            }
        },
    }

    // Calculate output dimensions
    var out_height: usize = undefined;
    var out_width: usize = undefined;

    if (ceil_mode) {
        out_height = @divTrunc(input_height + pad_h - effective_kernel_h + strides[0] - 1, strides[0]) + 1;
        out_width = @divTrunc(input_width + pad_w - effective_kernel_w + strides[1] - 1, strides[1]) + 1;
    } else {
        out_height = @divTrunc(input_height + pad_h - effective_kernel_h, strides[0]) + 1;
        out_width = @divTrunc(input_width + pad_w - effective_kernel_w, strides[1]) + 1;
    }

    output_shape[2] = out_height;
    output_shape[3] = out_width;

    return output_shape;
}

/// Lean version of onnx_maxpool that takes a pre-allocated output tensor
pub fn lean_onnx_maxpool(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
) !void {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const input_height = input.shape[2];
    const input_width = input.shape[3];
    const out_height = output.shape[2];
    const out_width = output.shape[3];

    // Calculate padding based on auto_pad mode
    var pad_top: usize = 0;
    var pad_left: usize = 0;
    var pad_bottom: usize = 0;
    var pad_right: usize = 0;

    switch (auto_pad) {
        .NOTSET => {
            if (pads.len >= 4) {
                pad_top = pads[0];
                pad_left = pads[1];
                pad_bottom = pads[2];
                pad_right = pads[3];
            }
        },
        .VALID => {},
        .SAME_UPPER, .SAME_LOWER => {
            const total_pad_height = (out_height - 1) * strides[0] + kernel_shape[0] - input_height;
            const total_pad_width = (out_width - 1) * strides[1] + kernel_shape[1] - input_width;
            if (auto_pad == .SAME_UPPER) {
                pad_top = total_pad_height / 2;
                pad_left = total_pad_width / 2;
                pad_bottom = total_pad_height - pad_top;
                pad_right = total_pad_width - pad_left;
            } else {
                pad_bottom = total_pad_height / 2;
                pad_right = total_pad_width / 2;
                pad_top = total_pad_height - pad_bottom;
                pad_left = total_pad_width - pad_right;
            }
        },
    }

    // Process each batch and channel
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const input_offset = b * channels * input_height * input_width + c * input_height * input_width;
            const output_offset = b * channels * out_height * out_width + c * out_height * out_width;

            // Process each output pixel
            var oh: usize = 0;
            while (oh < out_height) : (oh += 1) {
                var ow: usize = 0;
                while (ow < out_width) : (ow += 1) {
                    // Calculate starting input position
                    const ih_start = @as(isize, @intCast(oh * strides[0])) - @as(isize, @intCast(pad_top));
                    const iw_start = @as(isize, @intCast(ow * strides[1])) - @as(isize, @intCast(pad_left));

                    // Find maximum value in the window
                    var max_val = -std.math.inf(T);
                    var kh: usize = 0;
                    while (kh < kernel_shape[0]) : (kh += 1) {
                        var kw: usize = 0;
                        while (kw < kernel_shape[1]) : (kw += 1) {
                            const ih = ih_start + @as(isize, @intCast(kh * dilations[0]));
                            const iw = iw_start + @as(isize, @intCast(kw * dilations[1]));
                            if (ih >= 0 and ih < @as(isize, @intCast(input_height)) and
                                iw >= 0 and iw < @as(isize, @intCast(input_width)))
                            {
                                const val = input.data[input_offset + @as(usize, @intCast(ih)) * input_width + @as(usize, @intCast(iw))];
                                max_val = @max(max_val, val);
                            }
                        }
                    }
                    // Store result
                    output.data[output_offset + oh * out_width + ow] = max_val;
                }
            }
        }
    }
}
// Modify the original onnx_maxpool to use the lean version
//TODO: implement "ceil_mode" and "storage_order" https://onnx.ai/onnx/operators/onnx__MaxPool.html

pub fn onnx_maxpool(
    comptime T: type,
    input: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
) !struct { output: Tensor(T), used_input: Tensor(u8) } {
    // Calculate output shape
    const output_shape = try get_onnx_maxpool_output_shape(
        input.shape,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        ceil_mode,
    );
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    var used_input = try Tensor(u8).fromShape(&pkg_allocator, input.shape);
    errdefer used_input.deinit();

    try lean_onnx_maxpool(
        T,
        input,
        &output,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
    );

    return .{ .output = output, .used_input = used_input };
}

pub fn lean_onnx_averagepool(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    count_include_pad: bool,
) !void {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const input_height = input.shape[2];
    const input_width = input.shape[3];
    const out_height = output.shape[2];
    const out_width = output.shape[3];

    // Calculate padding based on auto_pad mode
    var pad_top: usize = 0;
    var pad_left: usize = 0;
    var pad_bottom: usize = 0;
    var pad_right: usize = 0;

    switch (auto_pad) {
        .NOTSET => {
            if (pads.len >= 4) {
                pad_top = pads[0];
                pad_left = pads[1];
                pad_bottom = pads[2];
                pad_right = pads[3];
            }
        },
        .VALID => {},
        .SAME_UPPER, .SAME_LOWER => {
            const total_pad_height = (out_height - 1) * strides[0] + kernel_shape[0] - input_height;
            const total_pad_width = (out_width - 1) * strides[1] + kernel_shape[1] - input_width;
            if (auto_pad == .SAME_UPPER) {
                pad_top = total_pad_height / 2;
                pad_left = total_pad_width / 2;
                pad_bottom = total_pad_height - pad_top;
                pad_right = total_pad_width - pad_left;
            } else {
                pad_bottom = total_pad_height / 2;
                pad_right = total_pad_width / 2;
                pad_top = total_pad_height - pad_bottom;
                pad_left = total_pad_width - pad_right;
            }
        },
    }

    // Process each batch and channel
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const input_offset = b * channels * input_height * input_width + c * input_height * input_width;
            const output_offset = b * channels * out_height * out_width + c * out_height * out_width;

            // Process each output pixel
            var oh: usize = 0;
            while (oh < out_height) : (oh += 1) {
                var ow: usize = 0;
                while (ow < out_width) : (ow += 1) {
                    // Calculate starting input position
                    const ih_start = @as(isize, @intCast(oh * strides[0])) - @as(isize, @intCast(pad_top));
                    const iw_start = @as(isize, @intCast(ow * strides[1])) - @as(isize, @intCast(pad_left));

                    // Find average value in the window
                    var sum: T = 0;
                    var count: usize = 0;
                    var kh: usize = 0;
                    while (kh < kernel_shape[0]) : (kh += 1) {
                        var kw: usize = 0;
                        while (kw < kernel_shape[1]) : (kw += 1) {
                            const ih = ih_start + @as(isize, @intCast(kh * dilations[0]));
                            const iw = iw_start + @as(isize, @intCast(kw * dilations[1]));
                            if (ih >= 0 and ih < @as(isize, @intCast(input_height)) and
                                iw >= 0 and iw < @as(isize, @intCast(input_width)))
                            {
                                const idx = input_offset + @as(usize, @intCast(ih)) * input_width + @as(usize, @intCast(iw));
                                const val = input.data[idx];
                                sum += val;
                                count += 1;
                            } else if (count_include_pad) {
                                count += 1; // Pad as 0 so sum doesn't change
                            }
                        }
                    }
                    if (count == 0) {
                        std.log.warn("Warning: count == 0 at oh={}, ow={}, ih_start={}, iw_start={}, pad_top={}, pad_left={}\n", .{ oh, ow, ih_start, iw_start, pad_top, pad_left });
                    }
                    // Store result
                    output.data[output_offset + oh * out_width + ow] = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;
                }
            }
        }
    }
}

pub fn onnx_averagepool(
    comptime T: type,
    input: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
    count_include_pad: bool,
) !Tensor(T) {
    // Calculate output shape
    const output_shape = try get_onnx_averagepool_output_shape(
        input.shape,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        ceil_mode,
    );
    defer pkg_allocator.free(output_shape);

    // Allocate output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Call lean version
    try lean_onnx_averagepool(
        T,
        input,
        &output,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        count_include_pad,
    );

    return output;
}

pub fn get_onnx_averagepool_output_shape(
    input_shape: []const usize,
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
) ![]usize {
    if (input_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const input_height = input_shape[2];
    const input_width = input_shape[3];

    // if (kernel_shape[0] > input_height or kernel_shape[1] > input_width) {
    //     var output_shape = try pkg_allocator.alloc(usize, 4);
    //     output_shape[0] = batch_size;
    //     output_shape[1] = channels;
    //     output_shape[2] = 1;
    //     output_shape[3] = 1;
    //     return output_shape;
    // }

    var output_shape = try pkg_allocator.alloc(usize, 4);
    errdefer pkg_allocator.free(output_shape);

    output_shape[0] = batch_size;
    output_shape[1] = channels;

    var pad_h: usize = 0;
    var pad_w: usize = 0;
    var pad_top: usize = 0;
    var pad_left: usize = 0;
    var pad_bottom: usize = 0;
    var pad_right: usize = 0;

    const effective_kernel_h = (kernel_shape[0] - 1) * dilations[0] + 1;
    const effective_kernel_w = (kernel_shape[1] - 1) * dilations[1] + 1;

    var out_height: usize = undefined;
    var out_width: usize = undefined;

    switch (auto_pad) {
        .NOTSET => {
            pad_top = pads[0];
            pad_left = pads[1];
            pad_bottom = pads[2];
            pad_right = pads[3];
            pad_h = pad_top + pad_bottom;
            pad_w = pad_left + pad_right;

            if (ceil_mode) {
                out_height = @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(input_height + pad_h - effective_kernel_h)) / @as(f64, @floatFromInt(strides[0]))))) + 1;
                out_width = @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(input_width + pad_w - effective_kernel_w)) / @as(f64, @floatFromInt(strides[1]))))) + 1;
            } else {
                out_height = @as(usize, @intFromFloat(std.math.floor(@as(f64, @floatFromInt(input_height + pad_h - effective_kernel_h)) / @as(f64, @floatFromInt(strides[0]))))) + 1;
                out_width = @as(usize, @intFromFloat(std.math.floor(@as(f64, @floatFromInt(input_width + pad_w - effective_kernel_w)) / @as(f64, @floatFromInt(strides[1]))))) + 1;
            }
        },
        .VALID => {
            if (ceil_mode) {
                out_height = @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(input_height - effective_kernel_h + 1)) / @as(f64, @floatFromInt(strides[0])))));
                out_width = @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(input_width - effective_kernel_w + 1)) / @as(f64, @floatFromInt(strides[1])))));
            } else {
                out_height = @as(usize, @intFromFloat(std.math.floor(@as(f64, @floatFromInt(input_height - effective_kernel_h)) / @as(f64, @floatFromInt(strides[0]))))) + 1;
                out_width = @as(usize, @intFromFloat(std.math.floor(@as(f64, @floatFromInt(input_width - effective_kernel_w)) / @as(f64, @floatFromInt(strides[1]))))) + 1;
            }
        },
        .SAME_UPPER, .SAME_LOWER => {
            if (ceil_mode) {
                out_height = @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(input_height)) / @as(f64, @floatFromInt(strides[0])))));
                out_width = @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(input_width)) / @as(f64, @floatFromInt(strides[1])))));
            } else {
                out_height = @as(usize, @intFromFloat(std.math.floor(@as(f64, @floatFromInt(input_height - 1)) / @as(f64, @floatFromInt(strides[0]))))) + 1;
                out_width = @as(usize, @intFromFloat(std.math.floor(@as(f64, @floatFromInt(input_width - 1)) / @as(f64, @floatFromInt(strides[1]))))) + 1;
            }
            pad_h = (out_height - 1) * strides[0] + effective_kernel_h - input_height;
            pad_w = (out_width - 1) * strides[1] + effective_kernel_w - input_width;

            if (auto_pad == .SAME_UPPER) {
                pad_top = pad_h / 2;
                pad_bottom = pad_h - pad_top;
                pad_left = pad_w / 2;
                pad_right = pad_w - pad_left;
            } else {
                pad_bottom = pad_h / 2;
                pad_top = pad_h - pad_bottom;
                pad_right = pad_w / 2;
                pad_left = pad_w - pad_right;
            }
        },
    }

    output_shape[2] = out_height;
    output_shape[3] = out_width;

    return output_shape;
}

//--------------------------------------------------------------------------
// lowerMaxPool2d — translate an ONNX *MaxPool-2D* node into UOps.
//
// HOST-SIDE PRE-PASS  (everything happens **before** this function is called)
// ──────────────────────────────────────────────────────────────────────────
// • Shapes
//     X : (N, C, H, W)                    kernel : (kH, kW)
// • Attributes
//     pads[4]        = {padT, padL, padB, padR}   (default 0)
//     strides[2]     = {strH, strW}               (default 1)
//     dilations[2]   = {dilH, dilW}               (default 1)
//     ceil_mode      = 0 | 1
//
// • Output spatial size  OH × OW
//     if ceil_mode==0   OH = ⌊(H+padT+padB − dilH·(kH−1) − 1)/strH⌋ + 1
//                        OW = ⌊(W+padL+padR − dilW·(kW−1) − 1)/strW⌋ + 1
//     else               use ⌈…⌉ instead of ⌊…⌋
//
// • Stride vector for X   (row-major NCHW, elements)
//       in_stride = [C·H·W,  H·W,  W, 1]
//
// • DType promotion (ONNX rules) → `out_dtype`
//
// The host passes: `out_shape=[N,C,OH,OW]`, `in_stride`, all pads/strides/
// dilations, kernel size, and `out_dtype`.  The IR below is then *fully
// static*: no dynamic shape logic remains.
//--------------------------------------------------------------------------
pub fn lowerMaxPool2d(
    b: *UOpBuilder,
    X_id: usize, // input tensor X
    out_shape: []const usize, // [N, C, OH, OW]
    in_stride: []const isize, // X strides (len 4)
    pads: [2]usize, // {padT, padL}
    strides_hw: [2]usize, // {strideH, strideW}
    dil_hw: [2]usize, // {dilH, dilW}
    kHW: [2]usize, // {kH, kW}
    out_dtype: DType,
    ceil_mode: bool,
) usize {
    // ── helpers --------------------------------------------------------
    const H = struct {
        fn rng(bi: *UOpBuilder, end: usize) usize {
            return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
        }
        fn k(bi: *UOpBuilder, v: usize) usize {
            return bi.push(.CONST, .i32, &.{}, Any{ .int = v });
        }
        fn fneg_inf(bi: *UOpBuilder) usize { // -∞  as float32
            const negative_inf: f32 = -std.math.inf(f32);
            return bi.push(.CONST, .f32, &.{}, Any{ .float = negative_inf });
        }
    };

    // --- Derive input shape from strides ---
    if (in_stride.len != 4 or in_stride[3] != 1) {
        // Basic validation assuming dense NCHW row-major
        @panic("lowerMaxPool2d expects dense NCHW input strides (..., W, 1)");
    }
    const W_in = @as(usize, @intCast(in_stride[2]));
    const H_in = if (W_in == 0) @panic("Input W cannot be 0") else @as(usize, @intCast(in_stride[1])) / W_in;
    const C_in = if (H_in * W_in == 0) @panic("Input H*W cannot be 0") else @as(usize, @intCast(in_stride[0])) / (H_in * W_in);
    const N_in = out_shape[0]; // Assume input N matches output N
    const derived_in_shape = [_]usize{ N_in, C_in, H_in, W_in };
    // ---------------------------------------

    // --- MOVE VIEW CREATION UP ---
    // Create the VIEW of the input tensor first to ensure its ID is registered.
    _ = b.push(.SHAPE, .i32, &.{X_id}, null); // Keep for potential debug/info
    const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{
        .view_meta = .{
            .shape = &derived_in_shape, // <<< USE DERIVED INPUT SHAPE
            .strides = in_stride,
        },
    });
    // --- END MOVE ---

    // ── constants ------------------------------------------------------
    // Now create constants. Their IDs will be assigned *after* the VIEW.
    const padT = H.k(b, pads[0]);
    const padL = H.k(b, pads[1]);
    const strH = H.k(b, strides_hw[0]);
    const strW = H.k(b, strides_hw[1]);
    const dilH = H.k(b, dil_hw[0]);
    const dilW = H.k(b, dil_hw[1]);
    const Hlim = H.k(b, H_in); // Use INPUT height for limit
    const Wlim = H.k(b, W_in); // Use INPUT width for limit
    const negInf = H.fneg_inf(b); // -∞ literal

    if (ceil_mode) {
        // TODO: Implement ceil mode adjustment
    }

    // ── view of X ------------------------------------------------------
    // REMOVED: _ = b.push(.SHAPE, .i32, &.{X_id}, null); // debug only -- Already done above
    // REMOVED: const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = &.{ out_shape[0], out_shape[1], out_shape[2], out_shape[3] }, .strides = in_stride } }); -- Already done above

    // output buffer Y
    const id_Y = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── outer loops n · c · oh · ow ------------------------------------
    const n = H.rng(b, out_shape[0]);
    const c = H.rng(b, out_shape[1]);
    const oh = H.rng(b, out_shape[2]);
    const ow = H.rng(b, out_shape[3]);

    // accumulator seeded to -∞
    const acc = b.push(.DEFINE_ACC, out_dtype, &.{negInf}, null);

    // ── kernel loops  kh · kw  -----------------------------------------
    const kh = H.rng(b, kHW[0]);
    const kw = H.rng(b, kHW[1]);

    // ------ spatial indices ---------------------------------------
    const ih = b.push(.SUB, .i32, &.{ b.push(.ADD, .i32, &.{ b.push(.MUL, .i32, &.{ oh, strH }, null), b.push(.MUL, .i32, &.{ kh, dilH }, null) }, null), padT }, null);

    const iw = b.push(.SUB, .i32, &.{ b.push(.ADD, .i32, &.{ b.push(.MUL, .i32, &.{ ow, strW }, null), b.push(.MUL, .i32, &.{ kw, dilW }, null) }, null), padL }, null);

    // ------ bounds check : 0 <= ih < H  &&  0 <= iw < W --------------
    // Need boolean constants for WHERE
    const bool_true = b.push(.CONST, .bool, &.{}, Any{ .bool = true });
    const bool_false = b.push(.CONST, .bool, &.{}, Any{ .bool = false });

    // Calculate: inside_h = (ih < Hlim) AND (NOT (ih < 0))
    const cond_h_lo = b.push(.CMPLT, .bool, &.{ ih, H.k(b, 0) }, null); // B = (ih < 0)
    const cond_h_hi = b.push(.CMPLT, .bool, &.{ ih, Hlim }, null); // A = (ih < Hlim)
    const not_cond_h_lo = b.push(.WHERE, .bool, &.{ cond_h_lo, bool_false, bool_true }, null); // NOT B
    const inside_h = b.push(.WHERE, .bool, &.{ cond_h_hi, not_cond_h_lo, bool_false }, null); // A AND (NOT B)

    // Calculate: inside_w = (iw < Wlim) AND (NOT (iw < 0))
    const cond_w_lo = b.push(.CMPLT, .bool, &.{ iw, H.k(b, 0) }, null); // B = (iw < 0)
    const cond_w_hi = b.push(.CMPLT, .bool, &.{ iw, Wlim }, null); // A = (iw < Wlim)
    const not_cond_w_lo = b.push(.WHERE, .bool, &.{ cond_w_lo, bool_false, bool_true }, null); // NOT B
    const inside_w = b.push(.WHERE, .bool, &.{ cond_w_hi, not_cond_w_lo, bool_false }, null); // A AND (NOT B)

    // Calculate: in_bounds = inside_h AND inside_w
    const in_bounds = b.push(.WHERE, .bool, &.{ inside_h, inside_w, bool_false }, null); // inside_h AND inside_w

    _ = b.push(.IF, .bool, &.{in_bounds}, null);

    const pX = b.push(.GEP, out_dtype, &.{ id_viewX, n, c, ih, iw }, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });

    const x = b.push(.LOAD, out_dtype, &.{pX}, null);
    _ = b.push(.MAX, out_dtype, &.{ acc, x }, null);

    _ = b.push(.ENDIF, .bool, &.{}, null);

    // end kernel loops
    _ = b.push(.ENDRANGE, .bool, &.{kw}, null);
    _ = b.push(.ENDRANGE, .bool, &.{kh}, null);

    // ── write result ----------------------------------------------------
    const pY = b.push(.GEP, out_dtype, &.{ id_Y, n, c, oh, ow }, Any{ .mem_info = .{ .base = id_Y, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ pY, acc }, null);

    // close outer loops
    _ = b.push(.ENDRANGE, .bool, &.{ow}, null);
    _ = b.push(.ENDRANGE, .bool, &.{oh}, null);
    _ = b.push(.ENDRANGE, .bool, &.{c}, null);
    _ = b.push(.ENDRANGE, .bool, &.{n}, null);

    return id_Y;
}
