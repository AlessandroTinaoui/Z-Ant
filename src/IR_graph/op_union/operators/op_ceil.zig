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

// https://onnx.ai/onnx/operators/onnx__Ceil.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor with ceiling of input elements
pub const Ceil = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Ceil {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Ceil{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Ceil) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: Ceil) *TensorZant {
        return self.output_Y;
    }

    pub fn write_op(self: Ceil, writer: std.fs.File.Writer) !void {
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.ceil_lean(T, {s}, &tensor_{s})
        , .{
            input_tensor_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn compute_output_shape(self: Ceil) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_ceil_output_shape(self.input_X.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Ceil) void { // TODO
        std.debug.print("\n Ceil:\n {any}", .{self});
    }
};
