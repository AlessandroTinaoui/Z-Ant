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

// https://onnx.ai/onnx/operators/onnx__Relu.html#l-onnx-doc-relu
// INPUTS:
//      - X (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - Y (heterogeneous) - T:  output tensor.

pub const Relu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Relu {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Relu{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Relu) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: Relu) *TensorZant {
        return self.output_Y;
    }

    pub fn write_op(self: Relu, writer: std.fs.File.Writer) !void {
        //----create tensor_A_string
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.ReLU_lean(T, {s}, &tensor_{s})
        , .{
            tensor_A_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn compute_output_shape(self: Relu) []usize {
        var output_shape: []usize = undefined;
        output_shape = self.input_X.shape;
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Relu) void {
        std.debug.print("\n Relu:\n {any}", .{self});
    }
};
