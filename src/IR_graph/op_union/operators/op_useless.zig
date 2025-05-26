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

pub const Useless = struct {
    pub fn init(nodeProto: *NodeProto) !Useless {
        _ = nodeProto; //"details" will be a onnx struct
        return Useless{};
    }

    pub fn get_output_shape() []usize {
        const res: []usize = [_]usize{ 1, 1, 2, 2 };
        return res;
    }

    pub fn print(self: Useless) void {
        std.debug.print("\n Useless:\n {any}", .{self});
    }
};
