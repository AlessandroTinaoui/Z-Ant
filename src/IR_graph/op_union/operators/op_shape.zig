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

// https://onnx.ai/onnx/operators/onnx__Shape.html
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
// OUTPUTS:
//      - shape (heterogeneous) - T1: Shape of the input tensor
// ATTRIBUTES:
//      - start - INT: First dimension to take
//      - end - INT: Last dimension to take
pub const Shape = struct {
    data: *TensorZant,
    shape: *TensorZant,
    //attributes:
    start: ?i64 = null,
    end: ?i64 = null,

    pub fn init(nodeProto: *NodeProto) !Shape {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.data_notFound;
        const shape = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.shape_notFound;

        var start: ?i64 = null;
        var end: ?i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "start")) {
                if (attr.type == onnx.AttributeType.INT) start = attr.i;
            } else if (std.mem.eql(u8, attr.name, "end")) {
                if (attr.type == onnx.AttributeType.INT) end = attr.i;
            }
        }

        return Shape{
            .data = data,
            .shape = shape,
            .start = start,
            .end = end,
        };
    }

    pub fn get_output_shape(self: Shape) []usize {
        return self.shape.getShape();
    }

    pub fn get_output_tensor(self: Shape) *TensorZant {
        return self.shape;
    }

    pub fn write_op(self: Shape, writer: std.fs.File.Writer) !void {
        //----create tensor_data_string
        var tensor_data_string: []u8 = undefined;
        defer allocator.free(tensor_data_string);
        if (self.data.tc == TensorCategory.INITIALIZER) {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.data.name),
                ")",
            });
        } else {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.data.name) });
        }

        //attributes
        _ = try writer.print(
            \\
            \\    tensMath.shape_onnx_lean(
            \\        T,
            \\        i64, //output type constraint
            \\        {s}, //input data tensor
            \\        {s}, //start
            \\        {s}, //end
            \\        &tensor_{s}, //output shape tensor
            \\    )
        , .{
            tensor_data_string,
            if (self.start) |s| try std.fmt.allocPrint(allocator, "{}", .{s}) else "null",
            if (self.end) |e| try std.fmt.allocPrint(allocator, "{}", .{e}) else "null",
            try utils.getSanitizedName(self.shape.name),
        });
    }

    pub fn compute_output_shape(self: Shape) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_shape_output_shape(
            self.data.shape,
            self.start,
            self.end,
        );
        self.shape.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Shape) void {
        std.debug.print("\n Shape:\n {any}", .{self});
    }
};
