const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const tensorMath = zant.core.tensor.math_standard;

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

const utils = @import("codegen").utils;

// https://onnx.ai/onnx/operators/onnx__Concat.html
// INPUTS:
//      - inputs (variadic, heterogeneous) - T: List of tensors for concatenation
// OUTPUTS:
//      - concat_result (heterogeneous) - T: Concatenated tensor
// ATTRIBUTES:
//      - axis (int, required): Which axis to concat on
pub const Concat = struct {
    inputs: std.ArrayList(*TensorZant),
    concat_result: *TensorZant,
    //attributes:
    axis: i64, // default = 1,

    pub fn init(nodeProto: *NodeProto) !Concat {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        const concat_result = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.concat_result_notFound;

        for (nodeProto.input) |input| {
            const ptr = if (tensorZant.tensorMap.getPtr(input)) |ptr| ptr else return error.concat_result_notFound;
            try inputs.append(ptr);
        }
        var axis: i64 = 1.0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        return Concat{
            .inputs = inputs,
            .concat_result = concat_result,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Concat) []usize {
        return self.concat_result.getShape();
    }

    pub fn get_output_tensor(self: Concat) *TensorZant {
        return self.concat_result;
    }

    pub fn write_op(self: Concat, writer: std.fs.File.Writer) !void {

        // Special case for axis 0 with different ranks
        if (self.axis == 0) {
            // Find if there are tensors with different ranks
            var has_different_ranks = false;
            const first_rank = self.inputs.items[0].shape.len;

            for (self.inputs.items[1..]) |input| {
                if (input.shape.len != first_rank) {
                    has_different_ranks = true;
                    break;
                }
            }

            if (has_different_ranks) {
                _ = try writer.print(
                    \\
                    \\    // Special case for concatenation along axis 0 with different ranks
                    \\    // This requires custom handling as the standard concatenate function expects same rank
                    \\    mathHandler_log.warn("\\nWarning: Concatenating tensors with different ranks along axis 0\\n", .{{}});
                    \\
                    \\    // Create a list of tensors to concatenate
                    \\    var concat_tensor_list_{s} = [_]Tensor(T){{
                , .{try utils.getSanitizedName(self.concat_result.name)});

                for (self.inputs.items, 0..) |input, idx| {
                    if (idx > 0) {
                        _ = try writer.print(", ", .{});
                    }

                    var tensor_string: []u8 = undefined;
                    defer allocator.free(tensor_string);
                    if (input.tc == TensorCategory.INITIALIZER) {
                        tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                            "@constCast(&param_lib.tensor_",
                            try utils.getSanitizedName(input.name),
                            ")",
                        });
                    } else {
                        tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(input.name) });
                    }
                    _ = try writer.print("{s}", .{tensor_string});
                }

                _ = try writer.print(
                    \\}};
                    \\
                    \\    // Perform concatenation with special handling for different ranks
                    \\     try tensMath.concatenate_lean(T, &allocator, &concat_tensor_list_{s}, {},tensor_{s})
                , .{
                    try utils.getSanitizedName(self.concat_result.name),
                    self.axis,
                    try utils.getSanitizedName(self.concat_result.name),
                });

                return;
            }
        }

        // Standard case: all tensors have the same rank
        // Create a tensor list with all input tensors
        _ = try writer.print(
            \\
            \\    // Create a list of tensors to concatenate
            \\    var concat_tensor_list_{s} = [_]Tensor(T){{
        , .{try utils.getSanitizedName(self.concat_result.name)});

        for (self.inputs.items, 0..) |input, idx| {
            if (idx > 0) {
                _ = try writer.print(", ", .{});
            }

            if (input.tc == TensorCategory.INITIALIZER) {
                _ = try writer.print("param_lib.tensor_{s}", .{try utils.getSanitizedName(input.name)});
            } else {
                _ = try writer.print("tensor_{s}", .{try utils.getSanitizedName(input.name)});
            }
        }

        _ = try writer.print(
            \\}};
            \\
            \\    // Perform concatenation
            \\    tensMath.concatenate_lean(T, &allocator, &concat_tensor_list_{s}, {}, &tensor_{s} )
        , .{
            try utils.getSanitizedName(self.concat_result.name),
            self.axis,
            try utils.getSanitizedName(self.concat_result.name),
        });
    }

    pub fn compute_output_shape(self: Concat) []usize {
        var output_shape: []usize = undefined;
        var input_shapes = try allocator.alloc([]const usize, self.inputs.items.len);
        const axis = self.axis;

        for (self.inputs.items, 0..) |input, i| {
            var shape = try allocator.alloc(usize, input.get_shape().len);
            for (input.get_shape(), 0..) |dim, j| {
                shape[j] = if (dim < 0) 1 else @intCast(dim);
            }
            input_shapes[i] = shape;
        }
        output_shape = try tensorMath.get_concatenate_output_shape(input_shapes, axis);
        self.concat_result.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Concat) void { // TODO
        std.debug.print("\n Flatten:\n {any}", .{self});
    }
};
