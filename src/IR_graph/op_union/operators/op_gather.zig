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
const tensorMath = zant.core.tensor.math_standard;
const utils = @import("codegen").utils;
const TensorCategory = tensorZant.TensorCategory;

// https://onnx.ai/onnx/operators/onnx__Gather.html
// INPUTS:
//      - A (heterogeneous) - T:  data tensor.
//      - B (heterogeneous) - T:  indices tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  result tensor.
// ATTRIBUTES:
//      - axis - INT (default is '0'): Indicate up to which input dimension should be gathered.

pub const Gather = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,
    //attributes:
    axis: i64 = 0, // default = 0,

    pub fn init(nodeProto: *NodeProto) !Gather {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        var axis: i64 = 0;
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i;
            }
        }

        return Gather{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Gather) []usize {
        return self.output_C.shape;
    }

    pub fn get_output_tensor(self: Gather) *TensorZant {
        return self.output_C;
    }

    // pub fn compute_output_shape(self: Gather) []usize {
    //     var output_shape: []usize = undefined;

    //     output_shape = try utils.usizeSliceToI64Slice(try tensorMath.get_gather_output_shape(
    //         try utils.i64SliceToUsizeSlice(data_shape),
    //         try utils.i64SliceToUsizeSlice(indices_shape),
    //         axis,
    //     ));
    // }

    pub fn write_op(self: Gather, writer: std.fs.File.Writer) !void {
        // Input A (data)
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);
        if (self.input_A.tc == tensorZant.TensorCategory.INITIALIZER) {
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

        // Input B (indices)
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        if (self.input_B.tc == tensorZant.TensorCategory.INITIALIZER) {
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

        // Output C
        const output_name = try utils.getSanitizedName(self.output_C.name);

        _ = try writer.print(
            \\
            \\
            \\    try tensMath.lean_gather(T, {s}, {s}, {}, &tensor_{s});
        , .{
            tensor_A_string,
            tensor_B_string,
            self.axis,
            output_name,
        });
    }

    pub fn print(self: Gather) void {
        std.debug.print("\n Gather:\n {any}", .{self});
    }
};
