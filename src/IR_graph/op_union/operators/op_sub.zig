const std = @import("std");
const zant = @import("zant");
const allocator = std.heap.page_allocator;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const utils = @import("../../utils.zig");
const TensorCategory = tensorZant.TensorCategory;

//https://onnx.ai/onnx/operators/onnx__Sub.html
// INPUTS:
//      - A (heterogeneous) - T: First input tensor
//      - B (heterogeneous) - T: Second input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Sub = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Sub {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Sub{
            .input_A = input_A,
            .input_B = input_B,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Sub) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: Sub) *TensorZant {
        return self.output_Y;
    }

    pub fn compute_output_shape(self: Sub) []usize {
        var output_shape: []usize = undefined;
        output_shape = try utils.broadcastShapes(allocator, self.input_A.shape, self.input_B.shape);
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Sub) void {
        std.debug.print("\n SUB: {any}", .{self});
    }

    pub fn write_op(self: Sub, writer: std.fs.File.Writer) !void {
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);

        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_A.name),
            });
        }

        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);

        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_B.name),
            });
        }

        _ = try writer.print(
            \\    tensMath.sub_tensors_lean(
            \\        T,
            \\        {s}, // input A
            \\        {s}, // input B
            \\        &tensor_{s} // output Y
            \\    );
        , .{
            tensor_A_string,
            tensor_B_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }
};
