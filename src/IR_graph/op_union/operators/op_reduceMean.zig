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

// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
//      - axes (optional, heterogeneous) - tensor(int64): A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'keepdims' is true.
// OUTPUTS:
//      - reduced (heterogeneous) - T: Reduced output tensor.
// ATTRIBUTES:
//      - keepdims (int, default is 1): Keep the reduced dimension or not, default 1 means keep the reduced dimension.
//      - noop_with_empty_axes (int, default is 0): Defines behavior if 'axes' is empty. Default behavior is to reduce all axes.
pub const ReduceMean = struct {
    data: *TensorZant,
    axes: ?*TensorZant,
    reduced: *TensorZant,
    //attributes:
    keepdims: bool, // default = true;
    noop_with_empty_axes: bool, // defualt = false;

    pub fn init(nodeProto: *NodeProto) !ReduceMean {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const axes = if (nodeProto.input.len > 1) if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.axes_notFound else null;
        const reduced = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var keepdims: bool = true;
        var noop_with_empty_axes: bool = false;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "keepdims")) {
                if (attr.type == onnx.AttributeType.INT) keepdims = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
                if (attr.type == onnx.AttributeType.INT) noop_with_empty_axes = attr.i != 0;
            }
        }

        return ReduceMean{
            .data = data,
            .axes = axes,
            .reduced = reduced,
            .keepdims = keepdims,
            .noop_with_empty_axes = noop_with_empty_axes,
        };
    }

    pub fn get_output_shape(self: ReduceMean) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: ReduceMean) *TensorZant {
        return self.output_Y;
    }

    pub fn write_op(self: ReduceMean, writer: std.fs.File.Writer) !void {

        // Create input tensor string
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);

        if (self.data.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.data.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.data.name) });
        }

        // Handle axes - either from attribute, input tensor, or as null
        var axes_str: []const u8 = "null";
        var needs_free = false;

        if (self.axes) |axes| {
            // Get axes from second input
            const axes_name = try utils.getSanitizedName(axes.name);

            if (axes.tc == TensorCategory.INITIALIZER) {
                // For initializer tensors, we need to extract the data directly
                axes_str = try std.fmt.allocPrint(allocator, "(@ptrCast([*]const i64, param_lib.tensor_{s}.data.ptr))[0..param_lib.tensor_{s}.size]", .{ axes_name, axes_name });
            } else {
                // For regular tensors
                axes_str = try std.fmt.allocPrint(allocator, "(@ptrCast([*]const i64, tensor_{s}.data.ptr))[0..tensor_{s}.size]", .{ axes_name, axes_name });
            }
            needs_free = true;
        }
        defer if (needs_free) allocator.free(axes_str);

        _ = try writer.print(
            \\
            \\    tensMath.reduce_mean_lean(
            \\        T, // type
            \\        {s}, // input tensor
            \\        &tensor_{s}, // output tensor
            \\        {s}, // axes
            \\        {s}, // keepdims
            \\        {s} // noop_with_empty_axes
            \\    )
        , .{
            input_tensor_string,
            try utils.getSanitizedName(self.reduced.name),
            axes_str,
            if (self.keepdims) "true" else "false",
            if (self.noop_with_empty_axes) "true" else "false",
        });
    }

    pub fn compute_output_shape(self: ReduceMean) []usize {
        var output_shape: []usize = undefined;
        const axes = self.axes.?.ptr.?.i64;
        const keepdims = self.keepdims;
        const noop_with_empty_axes = self.noop_with_empty_axes;
        output_shape = try tensorMath.get_reduce_mean_output_shape(
            self.data.shape,
            axes,
            keepdims,
            noop_with_empty_axes,
        );
        self.reduced.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: ReduceMean) void { // TODO
        std.debug.print("\n ReduceMean:\n {any}", .{self});
    }
};
