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

//https://onnx.ai/onnx/operators/onnx__Tanh.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Tanh = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Tanh {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Tanh{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Tanh) []usize {
        return self.output_Y.shape;
    }

    pub fn get_output_tensor(self: Tanh) *TensorZant {
        return self.output_Y;
    }

    pub fn print(self: Tanh) void {
        std.debug.print("\n Tanh: {any}", .{self});
    }

    pub fn compute_output_shape(self: Tanh) []usize {
        const output_shape: []usize = undefined;
        const input_shape = self.input_X.shape;
        output_shape = try tensorMath.get_tanh_output_shape(input_shape);
        return output_shape;
    }

    pub fn write_op(self: Tanh, writer: std.fs.File.Writer) !void {
        // --- Input tensor string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        // --- Write the Tanh op
        _ = try writer.print(
            \\    tensMath.tanh_lean(
            \\        T,
            \\        {s}, // input tensor
            \\        &tensor_{s} // output tensor
            \\    );
        , .{
            tensor_X_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }
};
