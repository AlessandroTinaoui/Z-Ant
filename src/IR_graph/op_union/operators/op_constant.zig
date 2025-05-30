const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const TensorCategory = tensorZant.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = @import("codegen").utils;

// https://onnx.ai/onnx/operators/onnx__Constant.html
// Outputs:
// - output (heterogeneous) - T: Output tensor containing the same value of the provided tensor.
// Attributes - only one of these should be specified:
// - value (TENSOR): The value for the elements of the output tensor.
// - sparse_value (SPARSE_TENSOR): The value for the elements of the output tensor in sparse format.
// - value_float (FLOAT): The value for the sole element for the scalar, float32, output tensor.
// - value_floats (FLOATS): The values for the elements for the 1D, float32, output tensor.
// - value_int (INT): The value for the sole element for the scalar, int64, output tensor.
// - value_ints (INTS): The values for the elements for the 1D, int64, output tensor.
// - value_string (STRING): The value for the sole element for the scalar, UTF-8 string, output tensor.
// - value_strings (STRINGS): The values for the elements for the 1D, UTF-8 string, output tensor.

pub const Constant = struct {
    output: *TensorZant,
    // attributes:
    value: ?*TensorZant,
    sparse_value: ?*TensorZant,
    value_float: ?f32,
    value_floats: ?[]f32,
    value_int: ?i64,
    value_ints: ?[]i64,
    value_string: ?[]const u8,
    value_strings: ?[][]const u8,

    pub fn init(nodeProto: *NodeProto) !Constant {
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;
        var value: ?*TensorZant = null;
        var sparse_value: ?*TensorZant = null;
        var value_float: ?f32 = null;
        var value_floats: ?[]f32 = null;
        var value_int: ?i64 = null;
        var value_ints: ?[]i64 = null;
        var value_string: ?[]const u8 = null;
        var value_strings: ?[][]const u8 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "value")) |_| {
                if (attr.type == onnx.AttributeType.TENSOR) value = attr.t;
            } else if (std.mem.indexOf(u8, attr.name, "sparse_value")) |_| {
                if (attr.type == onnx.AttributeType.SPARSE_TENSOR) sparse_value = attr.sparse_tensor;
            } else if (std.mem.indexOf(u8, attr.name, "value_float")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) value_float = attr.f;
            } else if (std.mem.indexOf(u8, attr.name, "value_floats")) |_| {
                if (attr.type == onnx.AttributeType.FLOATS) value_floats = attr.floats;
            } else if (std.mem.indexOf(u8, attr.name, "value_int")) |_| {
                if (attr.type == onnx.AttributeType.INT) value_int = attr.i;
            } else if (std.mem.indexOf(u8, attr.name, "value_ints")) |_| {
                if (attr.type == onnx.AttributeType.INTS) value_ints = attr.ints;
            } else if (std.mem.indexOf(u8, attr.name, "value_string")) |_| {
                if (attr.type == onnx.AttributeType.STRING) value_string = attr.s;
            } else if (std.mem.indexOf(u8, attr.name, "value_strings")) |_| {
                if (attr.type == onnx.AttributeType.STRINGS) value_strings = attr.strings;
            }
        }

        return Constant{
            .output = output,
            .value = value,
            .sparse_value = sparse_value,
            .value_float = value_float,
            .value_floats = value_floats,
            .value_int = value_int,
            .value_ints = value_ints,
            .value_string = value_string,
            .value_strings = value_strings,
        };
    }

    pub fn get_output_shape(self: Constant) []usize {
        return self.output.getShape();
    }

    pub fn get_output_tensor(self: Constant) *TensorZant {
        return self.output;
    }

    pub fn write_op(self: Constant, writer: std.fs.File.Writer) !void {
        const output_name = try utils.getSanitizedName(self.output.name);

        if (self.value != null) {
            try writer.print(
                \\
                \\    // Constant tensor_{s} already declared and initialized in predict.zig
            , .{output_name});
            return;
        } else if (self.value_float != null) {
            try writer.print(
                \\
                \\    // Initialize scalar float constant
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, {d}) catch return;
            , .{ output_name, self.value_float.? });
            return;
        } else if (self.value_floats != null) {
            try writer.print(
                \\
                \\    // Initialize 1D float array constant
                \\    const data_{s} = [_]T{{
            , .{output_name});

            for (self.value_floats.?, 0..) |val, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{val});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
            , .{ output_name, output_name, self.value_floats.?.len });
            return;
        } else if (self.value_int != null) {
            try writer.print(
                \\
                \\    // Initialize scalar int constant
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, @as(T, @floatFromInt({d}))) catch return;
            , .{ output_name, self.value_int.? });
            return;
        } else if (self.value_ints != null) {
            try writer.print(
                \\
                \\    // Initialize 1D int array constant
                \\    const data_{s} = [_]T{{
            , .{output_name});

            for (self.value_ints.?, 0..) |val, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("@as(T, @floatFromInt({d}))", .{val});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
            , .{ output_name, output_name, self.value_ints.?.len });
            return;
        } else if (self.value_string != null) {
            try writer.print(
                \\
                \\    // String constants are not directly supported in this numeric tensor library
                \\    // For now, we'll create a placeholder tensor with a single value
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, 0) catch return;
                \\    // The actual string value was: "{s}"
            , .{ output_name, self.value_string.? });
            return;
        } else if (self.value_strings != null) {
            try writer.print(
                \\
                \\    // String array constants are not directly supported in this numeric tensor library
                \\    // For now, we'll create a placeholder tensor with zeros
                \\    const data_{s} = [_]T{{
            , .{output_name});

            for (self.value_strings.?, 0..) |_, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("0", .{});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
                \\    // Note: This is a placeholder for string values that cannot be directly represented
            , .{ output_name, output_name, self.value_strings.?.len });
            return;
        } else if (self.sparse_value != null) {
            try writer.print(
                \\
                \\    // Sparse tensor constants are not yet fully supported
                \\    // Creating a placeholder tensor for sparse_value
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, 0) catch return;
                \\    mathHandler_log.warn("Warning: sparse_value attribute used but not fully supported\\n", .{{}});
            , .{output_name});
            return;
        }

        try writer.writeAll(
            \\
            \\    return error.ConstantValueNotFound;
        );
    }

    pub fn compute_output_shape(self: Constant) []usize {
        var output_shape: []usize = undefined;
        var lenght: usize = 0;
        if (self.value != null) {
            if (output_shape.len == 0) {
                output_shape = try allocator.dupe(usize, &[_]usize{1});
            } else {
                output_shape = self.value.?.getShape();
            }
        } else if (self.value_float != null or self.value_int != null or self.value_string != null) {
            output_shape = try allocator.dupe(usize, &[_]usize{1});
        } else if (self.value_floats != null) {
            lenght = self.value_floats.?.len;
            output_shape = try allocator.dupe(usize, &[_]usize{lenght});
        } else if (self.value_ints != null) {
            lenght = self.value_ints.?.len;
            output_shape = try allocator.dupe(usize, &[_]usize{lenght});
        } else if (self.value_strings != null) {
            lenght = self.value_strings.?.len;
            output_shape = try allocator.dupe(usize, &[_]usize{lenght});
        } else if (self.sparse_value != null) {
            output_shape = self.sparse_value.?.getShape();
        }
        return output_shape;
    }
};
