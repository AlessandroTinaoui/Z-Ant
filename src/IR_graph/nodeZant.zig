const std = @import("std");
const zant = @import("zant");

const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.tensor.Tensor;
const Op_union = @import("op_union/op_union.zig").Op_union;

//--- proto structure
const NodeProto = zant.onnx.NodeProto;
const TensorProto = zant.onnx.TensorProto;
const DataType = zant.onnx.DataType;

const utils = zant.utils;
pub const NodeZant = struct {
    name: ?[]const u8, //name of the node
    op_type: []const u8, //onnx name of the operation, see here: https://onnx.ai/onnx/operators/
    op: Op_union, // contains all the information of the operation
    next: std.ArrayList(*NodeZant), // points to the following nodes

    nodeProto: *NodeProto,
    ready: bool,

    /// Initializes a NodeZant instance starting from a NodeProto instance.
    pub fn init(nodeProto: *NodeProto) !NodeZant {
        return NodeZant{
            .name = nodeProto.name.?,
            .op_type = nodeProto.op_type,
            .op = try Op_union.init(nodeProto),
            .next = std.ArrayList(*NodeZant).init(allocator),
            .nodeProto = nodeProto,
            .ready = false,
        };
    }

    /// Deinitializes a NodeZant instance, freeing allocated resources.
    pub fn deinit(self: *NodeZant) void {
        self.next.deinit();
    }

    /// Adds a new node to the next list.
    pub fn add_next(self: *NodeZant, next_node: *NodeZant) !void {
        try self.next.append(next_node);
    }

    pub fn write_op(self: *NodeZant, writer: std.fs.File.Writer) !void {
        try self.op.write_op(writer);
    }
};
