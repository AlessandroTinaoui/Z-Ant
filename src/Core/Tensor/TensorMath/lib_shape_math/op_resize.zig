const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

pub fn resize(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8) !Tensor(T) {
    //check if mode exixts:
    if (!(std.mem.eql(u8, mode, "nearest") or std.mem.eql(u8, mode, "linear") or std.mem.eql(u8, mode, "cubic") or std.mem.eql(u8, mode, "floor"))) {
        return TensorError.UnsupportedMode;
    }

    //check args: there should be one and only one between scales and sizes
    if (scales == null and sizes == null) {
        return TensorError.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return TensorError.InvalidInput;
    }

    // Create output tensor
    const output_shape = try get_resize_output_shape(t.shape, scales, sizes);
    defer t.allocator.free(output_shape);
    var output = try Tensor(T).fromShape(t.allocator, output_shape);

    //call rezise_lean
    if (scales) |s| {
        if (s.len != t.shape.len) {
            return TensorError.InvalidInput;
        } else {
            try rezise_lean(T, t, mode, scales, null, coordinate_transformation_mode, &output);
        }
    } else if (sizes) |sz| {
        if (sz.len != t.shape.len) {
            return TensorError.InvalidInput;
        } else {
            try rezise_lean(T, t, mode, null, sizes, coordinate_transformation_mode, &output);
        }
    }

    return output;
}

//resize lean
pub fn rezise_lean(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8, output_tensor: *Tensor(T)) !void {
    std.log.debug("rezise_lean\n", .{});
    std.log.debug("mode: {s}\n", .{mode});
    std.log.debug("scales: {any}\n", .{scales});
    std.log.debug("sizes: {any}\n", .{sizes});
    std.log.debug("coordinate_transformation_mode: {s}\n", .{coordinate_transformation_mode});

    // Perform interpolation based on mode
    if (std.mem.eql(u8, mode, "nearest")) {
        try nearest_interpolation(T, t, output_tensor.data, output_tensor.shape, coordinate_transformation_mode);
    } else if (std.mem.eql(u8, mode, "linear")) {
        try linear_interpolation(T, t, output_tensor.data, output_tensor.shape, coordinate_transformation_mode);
    } else if (std.mem.eql(u8, mode, "floor")) {
        try floor_interpolation(T, t, output_tensor.data, output_tensor.shape, coordinate_transformation_mode);
    } else { //cubic interpolation
        try cubic_interpolation(T, t, output_tensor.data, output_tensor.shape, coordinate_transformation_mode);
    }
}

pub fn get_resize_output_shape(input_shape: []const usize, scales: ?[]const f32, sizes: ?[]const usize) ![]usize {
    if (scales == null and sizes == null) {
        return TensorError.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return TensorError.InvalidInput;
    }

    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    if (scales) |s| {
        if (s.len != input_shape.len) {
            return TensorError.InvalidInput;
        }
        for (0..input_shape.len) |i| {
            output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(input_shape[i])) * s[i]));
        }
    } else if (sizes) |sz| {
        if (sz.len != input_shape.len) {
            return TensorError.InvalidInput;
        }
        @memcpy(output_shape, sz);
    }

    return output_shape;
}

fn nearest_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    @setEvalBranchQuota(10000);

    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);
    const output_strides = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_strides);

    // Calculate output strides
    var stride: usize = 1;
    var idx: usize = output_shape.len;
    while (idx > 0) {
        idx -= 1;
        output_strides[idx] = stride;
        stride *= output_shape[idx];
    }

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        var input_idx: usize = 0;

        for (0..output_shape.len) |i| {
            const scale = @as(f32, @floatFromInt(output_shape[i])) / @as(f32, @floatFromInt(self.shape[i]));
            var input_pos: f32 = undefined;

            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                input_pos = (@as(f32, @floatFromInt(output_indices[i])) + 0.5) / scale - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                input_pos = @as(f32, @floatFromInt(output_indices[i])) * @as(f32, @floatFromInt(self.shape[i] - 1)) / @as(f32, @floatFromInt(output_shape[i] - 1));
            } else { // asymmetric
                input_pos = @as(f32, @floatFromInt(output_indices[i])) / scale;
            }

            const input_idx_i = @as(i32, @intFromFloat(@round(input_pos)));
            const clamped_idx = @min(@max(input_idx_i, 0), @as(i32, @intCast(self.shape[i] - 1)));
            input_idx += @as(usize, @intCast(clamped_idx)) * input_strides[i];
            output_idx += output_indices[i] * output_strides[i];
        }

        output_data[output_idx] = self.data[input_idx];

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn linear_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    @setEvalBranchQuota(10000);

    // For now, implement only for 1D and 2D tensors
    if (self.shape.len > 2) return TensorError.UnsupportedDimension;

    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        if (output_shape.len == 1) {
            output_idx = output_indices[0];
        } else {
            output_idx = output_indices[0] * output_shape[1] + output_indices[1];
        }

        // Calculate interpolation coordinates
        var x: f32 = undefined;
        if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
            x = (@as(f32, @floatFromInt(output_indices[0])) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
        } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
            x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
        } else { // asymmetric
            x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
        }

        const x_floor = @floor(x);
        const x0 = @as(usize, @intFromFloat(@max(0, x_floor)));
        const x1 = @min(x0 + 1, self.shape[0] - 1);
        const dx = x - x_floor;

        if (self.shape.len == 1) {
            var v0: f32 = undefined;
            var v1: f32 = undefined;

            if (@typeInfo(T) == .float) {
                v0 = self.data[x0];
                v1 = self.data[x1];
            } else {
                v0 = @as(f32, @floatFromInt(self.data[x0]));
                v1 = @as(f32, @floatFromInt(self.data[x1]));
            }

            const interpolated = v0 * (1 - dx) + v1 * dx;

            if (@typeInfo(T) == .float) {
                output_data[output_idx] = @floatCast(interpolated);
            } else {
                output_data[output_idx] = @intFromFloat(@round(interpolated));
            }
        } else {
            var y: f32 = undefined;
            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                y = (@as(f32, @floatFromInt(output_indices[1])) + 0.5) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1])) - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1] - 1)) / @as(f32, @floatFromInt(output_shape[1] - 1));
            } else { // asymmetric
                y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1]));
            }

            const y_floor = @floor(y);
            const y0 = @as(usize, @intFromFloat(@max(0, y_floor)));
            const y1 = @min(y0 + 1, self.shape[1] - 1);
            const dy = y - y_floor;

            var v00: f32 = undefined;
            var v01: f32 = undefined;
            var v10: f32 = undefined;
            var v11: f32 = undefined;

            if (@typeInfo(T) == .float) {
                v00 = self.data[x0 * self.shape[1] + y0];
                v01 = self.data[x0 * self.shape[1] + y1];
                v10 = self.data[x1 * self.shape[1] + y0];
                v11 = self.data[x1 * self.shape[1] + y1];
            } else {
                v00 = @as(f32, @floatFromInt(self.data[x0 * self.shape[1] + y0]));
                v01 = @as(f32, @floatFromInt(self.data[x0 * self.shape[1] + y1]));
                v10 = @as(f32, @floatFromInt(self.data[x1 * self.shape[1] + y0]));
                v11 = @as(f32, @floatFromInt(self.data[x1 * self.shape[1] + y1]));
            }

            const tmp1 = v00 * (1 - dx) * (1 - dy);
            const tmp2 = v01 * (1 - dx) * dy;
            const tmp3 = v10 * dx * (1 - dy);
            const tmp4 = v11 * dx * dy;

            const interpolated = tmp1 + tmp2 + tmp3 + tmp4;

            if (@typeInfo(T) == .float) {
                output_data[output_idx] = @floatCast(interpolated);
            } else {
                output_data[output_idx] = @intFromFloat(@round(interpolated));
            }
        }

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn floor_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    @setEvalBranchQuota(10000);

    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);
    const output_strides = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_strides);

    // Calculate output strides
    var stride: usize = 1;
    var idx: usize = output_shape.len;
    while (idx > 0) {
        idx -= 1;
        output_strides[idx] = stride;
        stride *= output_shape[idx];
    }

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        var input_idx: usize = 0;

        for (0..output_shape.len) |i| {
            const scale = @as(f32, @floatFromInt(output_shape[i])) / @as(f32, @floatFromInt(self.shape[i]));
            var input_pos: f32 = undefined;

            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                input_pos = (@as(f32, @floatFromInt(output_indices[i])) + 0.5) / scale - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                input_pos = @as(f32, @floatFromInt(output_indices[i])) * @as(f32, @floatFromInt(self.shape[i] - 1)) / @as(f32, @floatFromInt(output_shape[i] - 1));
            } else { // asymmetric
                input_pos = @as(f32, @floatFromInt(output_indices[i])) / scale;
            }

            // Use floor instead of round for this mode
            const input_idx_i = @as(i32, @intFromFloat(@floor(input_pos)));
            const clamped_idx = @min(@max(input_idx_i, 0), @as(i32, @intCast(self.shape[i] - 1)));
            input_idx += @as(usize, @intCast(clamped_idx)) * input_strides[i];
            output_idx += output_indices[i] * output_strides[i];
        }

        output_data[output_idx] = self.data[input_idx];

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn cubic_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    @setEvalBranchQuota(10000);

    // For simplicity, implement only for 1D tensors initially
    if (self.shape.len != 1) return TensorError.UnsupportedDimension;

    var output_idx: usize = 0;
    while (output_idx < output_shape[0]) : (output_idx += 1) {
        var x: f32 = undefined;
        if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
            x = (@as(f32, @floatFromInt(output_idx)) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
        } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
            x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
        } else { // asymmetric
            x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
        }

        const x0 = @as(i32, @intFromFloat(@floor(x)));
        const dx = x - @as(f32, @floatFromInt(x0));

        var sum: f32 = 0;
        var weight_sum: f32 = 0;

        var i: i32 = -1;
        while (i < 3) : (i += 1) {
            const idx = x0 + i;
            if (idx >= 0 and idx < @as(i32, @intCast(self.shape[0]))) {
                const w = cubic_weight(dx - @as(f32, @floatFromInt(i)));
                var value: f32 = undefined;

                if (@typeInfo(T) == .float) {
                    value = self.data[@as(usize, @intCast(idx))];
                } else {
                    value = @as(f32, @floatFromInt(self.data[@as(usize, @intCast(idx))]));
                }

                sum += value * w;
                weight_sum += w;
            }
        }

        if (@typeInfo(T) == .float) {
            output_data[output_idx] = @floatCast(sum / weight_sum);
        } else {
            output_data[output_idx] = @intFromFloat(@round(sum / weight_sum));
        }
    }
}

fn cubic_weight(x: f32) f32 {
    const a = -0.75;
    const abs_x = @abs(x);
    if (abs_x <= 1) {
        return ((a + 2) * abs_x - (a + 3)) * abs_x * abs_x + 1;
    } else if (abs_x < 2) {
        return ((a * abs_x - 5 * a) * abs_x + 8 * a) * abs_x - 4 * a;
    }
    return 0;
}
