const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");

// --- onnx ---
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const tensorMath = zant.core.tensor.math_standard;
const TensorCategory = tensorZant.TensorCategory;
const utils = @import("codegen").utils;

//https://onnx.ai/onnx/operators/onnx__Softmax.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Softmax = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Softmax {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Softmax{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Softmax) []usize {
        return self.output_Y.shape;
    }

    pub fn compute_output_shape(self: Softmax) []usize {
        var output_shape: []usize = undefined;
        output_shape = self.input_X.shape;
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Softmax) void {
        std.debug.print("\n Softmax: {any}", .{self});
    }

    pub fn write_op(self: Softmax, writer: std.fs.File.Writer) !void {
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        _ = try writer.print(
            \\    tensMath.softmax_tensor_lean(
            \\        T,
            \\        {s}, // input tensor
            \\        &tensor_{s} // output tensor
            \\    );
            \\
        , .{
            tensor_input_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }
};
