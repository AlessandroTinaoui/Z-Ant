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
const IR_utils = @import("../../utils.zig"); //this is IR utils
const utils = @import("codegen").utils;

// https://onnx.ai/onnx/operators/onnx__OneHot.html
// INPUTS:
//      - indices (heterogeneous) - T1: Input tensor
//      - depth (heterogeneous) - T2: Scalar or Rank 1 tensor
//      - values (heterogeneous) - T3: Rank 1 tensor
// OUTPUTS:
//      - output (heterogeneous) - T3: Tensor of rank one greater than input tensor ‘indices’
// ATTRIBUTES:
//      - axis - INT (default is '-1'): (Optional) Axis

pub const OneHot = struct {
    indices: *TensorZant,
    depth: *TensorZant,
    values: *TensorZant,
    output: *TensorZant,
    axis: ?i64,

    pub fn init(nodeProto: *NodeProto) !OneHot {
        const indices = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const depth = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.depth_notFound;
        const values = if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.values_notFound;
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = -1;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        return OneHot{
            .indices = indices,
            .depth = depth,
            .values = values,
            .output = output,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: OneHot) []usize {
        return self.output.getShape();
    }

    pub fn get_output_tensor(self: OneHot) *TensorZant {
        return self.output;
    }

    pub fn compute_output_shape(self: OneHot) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_oneHot_output_shape(
            self.indices.getShape(),
            self.depth.ptr.?.i64.data[0],
            self.axis.?,
        );
        return output_shape;
    }

    pub fn write_op(self: OneHot, writer: std.fs.File.Writer) !void {
        //----create indices string
        var indices_string: []u8 = undefined;
        defer allocator.free(indices_string);
        if (self.indices.tc == TensorCategory.INITIALIZER) {
            indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.indices.name),
                ")",
            });
        } else {
            indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.indices.name),
                ")",
            });
        }

        //----create depth string
        var depth_string: []u8 = undefined;
        defer allocator.free(depth_string);
        if (self.depth.tc == TensorCategory.INITIALIZER) {
            depth_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.depth.name),
                ")",
            });
        } else {
            depth_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.depth.name),
                ")",
            });
        }

        //----create values string
        var values_string: []u8 = undefined;
        defer allocator.free(values_string);
        if (self.values.tc == TensorCategory.INITIALIZER) {
            values_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.values.name),
                ")",
            });
        } else {
            values_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.values.name),
                ")",
            });
        }

        _ = try writer.print(
            \\    
            \\
            \\    tensMath.oneHot_lean(
            \\        {s}, // T
            \\        {s}, // indices
            \\        {s}.data[0], // depth (scalare)
            \\        {s}, // values
            \\        {}, // axis
            \\        &tensor_{s}, // output
            \\    )
        , .{
            self.values.ty.toString(), // T
            indices_string, // indices
            depth_string, // depth
            values_string, // values
            self.axis, // axis
            try utils.getSanitizedName(self.output.name), // output
        });
    }

    pub fn print(self: OneHot) void { // TODO
        std.debug.print("\n OneHot:\n {any}", .{self});
    }
};
