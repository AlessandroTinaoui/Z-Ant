const std = @import("std");
const zant = @import("zant");

const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;

//--- protos ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;

//--- zant ---
pub const graphZant_lib = @import("graphZant.zig");
pub const GraphZant = graphZant_lib.GraphZant;

pub const NodeZant_lib = @import("nodeZant.zig");
pub const NodeZant = NodeZant_lib.NodeZant;

pub const tensorZant_lib = @import("tensorZant.zig");
pub const TensorZant = tensorZant_lib.TensorZant;
pub const TensorCategory = tensorZant_lib.TensorCategory;

pub const operators = @import("op_union/op_union.zig").operators;
pub const utils = @import("utils.zig");

pub fn init(modelProto: *ModelProto) !GraphZant {

    //initialize the tensor hash map
    try tensorZant_lib.initialize_tensorZantMap(modelProto);

    //check
    const graphProto = try if (modelProto.graph) |graph| graph else error.GraphNotAvailable;

    //initialize the graphZant
    var graphZant = try GraphZant.init(graphProto);

    //constructing the graph
    try graphZant.build_graph();

    return graphZant;
}
