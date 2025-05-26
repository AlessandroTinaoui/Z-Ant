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

//https://onnx.ai/onnx/operators/onnx__Split.html
// INPUTS:
//      - input (heterogeneous) - T: Input tensor
//      - split (optional, heterogeneous) - tensor(int64):
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - axis - INT (default is '0'): Indicate up to which input dimension should be split.
//      - num_outputs - INT: Number of outputs
pub const Split = struct {
    input: *TensorZant,
    split: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    axis: i64 = 0, // default = 0,

    pub fn init(nodeProto: *NodeProto) !Split {
        const input = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const splitTensor = if (nodeProto.input.len > 1) if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.axes_notFound else null;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = 0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i;
            }
        }

        return Split{
            .input = input,
            .split = splitTensor,
            .output_Y = output_Y,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Split) []usize {
        return self.output_Y.shape;
    }

    pub fn get_output_tensor(self: Split) *TensorZant {
        return self.output_Y;
    }

    pub fn compute_output_shape() ![]usize {} // TODO

    pub fn print(self: Split) void {
        std.debug.print("\n Split: {any}", .{self});
    }

    pub fn write_op(self: Split, writer: std.fs.File.Writer) !void {
        // --- Crea stringa per input
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);

        if (self.input.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input.name),
            });
        }

        // --- Crea stringa per split (se presente)
        var tensor_split_string: []u8 = undefined;
        defer allocator.free(tensor_split_string);

        if (self.split != null) {
            if (self.split.?.tc == TensorCategory.INITIALIZER) {
                tensor_split_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(self.split.?.name),
                    ")",
                });
            } else {
                tensor_split_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "&tensor_",
                    try utils.getSanitizedName(self.split.?.name),
                });
            }
        }

        // --- Scrivi chiamata a tensMath.split_tensors_lean
        _ = try writer.print(
            \\    tensMath.split_tensors_lean(
            \\        T,
            \\        {s}, // input tensor
            \\        {s}, // split tensor
            \\        &tensor_{s}, // output Y
            \\        {d}, // axis
            \\        {d}, // num_outputs (da determinare se presente in modo dinamico)
            \\    );
        , .{
            tensor_input_string,
            tensor_split_string,
            try utils.getSanitizedName(self.output_Y.name),
            self.axis,
                // Se num_outputs è determinabile, passalo, altrimenti utilizza un valore di default
                // Per ora supponiamo che self.output_Y abbia più tensori di output
                // Aggiungere logica per num_outputs se applicabile
            "num_outputs_value_here", // Placeholder, puoi modificare secondo la logica interna
        });
    }
};
