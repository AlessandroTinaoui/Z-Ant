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

// https://onnx.ai/onnx/operators/onnx__Slice.html
// INPUTS:
//      - input (heterogeneous) - T: Tensor of data to extract slices from.
//      - starts (heterogeneous) - T1: 1-D tensor of starting indices of corresponding axis in `axes`.
//      - ends (heterogeneous) - T1: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`.
//      - axes (heterogeneous) - T1: 1-D tensor of axes that `starts` and `ends` apply to.
//      - steps (heterogeneous) - T1: 1-D tensor of slice step of corresponding axis in `axes`.
// OUTPUTS:
//      - output (heterogeneous) - T: Sliced data tensor.

pub const Slice = struct {
    input: *TensorZant,
    starts: *TensorZant,
    ends: *TensorZant,
    axes: ?*TensorZant,
    steps: ?*TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Slice {
        const input = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const starts = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_X_notFound;
        const ends = if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_X_notFound;
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;
        // Optional inputs
        const axes: ?*TensorZant = if (nodeProto.input.len >= 4) if (tensorZant.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.axes_notFound else null;
        const steps: ?*TensorZant = if (nodeProto.input.len >= 4) if (tensorZant.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.steps_notFound else null;

        return Slice{
            .input = input,
            .starts = starts,
            .ends = ends,
            .axes = axes,
            .steps = steps,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Slice) []usize {
        return self.output.getShape();
    }

    pub fn get_output_tensor(self: Slice) *TensorZant {
        return self.output;
    }

    pub fn write_op() !void {} //TODO manuel

    pub fn compute_output_shape(self: Slice) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_slice_output_shape(
            self.input.shape,
            self.starts.ptr.?.i64.data,
            self.ends.ptr.?.i64.data,
            self.axes.ptr.?.i64.data,
            self.steps.ptr.?.i64.data,
        );
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Slice) void {
        std.debug.print("\n Slice: {any}", .{self});
    }
};
