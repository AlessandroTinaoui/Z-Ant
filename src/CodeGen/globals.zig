const std = @import("std");
const zant = @import("zant");
const Tensor = zant.tensor.Tensor;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
//--- proto
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeProto = onnx.AttributeProto;
const allocator = zant.utils.allocator.allocator;

//--- other
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const mathGen = codegen.math_handler;
const shapeGen = codegen.shape_handler;
const codegen_options = @import("codegen_options");
const globals_log = std.log.scoped(.globals);

pub var readyGraph: std.ArrayList(ReadyNode) = std.ArrayList(ReadyNode).init(allocator);
pub var tensorHashMap: std.StringHashMap(ReadyTensor) = std.StringHashMap(ReadyTensor).init(allocator); //key: TensorProto.name
// Map from tensor name to remaining use count in generated predict
pub var tensorUseCount: std.StringHashMap(usize) = std.StringHashMap(usize).init(allocator);

pub var onnxModel: ModelOnnx = undefined; //initialized in setGlobalAttributes(), it is mandatory

pub const io_struct = struct {
    name: []const u8,
    shape: []const i64,
};

pub var networkInput = io_struct{
    .name = "",
    .shape = &[_]i64{},
};

pub var networkOutput = io_struct{
    .name = "",
    .shape = &[_]i64{},
};
// DataType of the network input tensor (derived from ONNX graph)
// String form of the network input element type (e.g. "f32", "u8", etc.)
pub var networkInputTypeString: []const u8 = "";
// Add a global variable to store the actual DataType enum value
pub var networkInputDataType: DataType = .UNDEFINED;

pub var inputType: type = f32;

pub const TensorTag = enum {
    INITIALIZER,
    CONSTANT,
    INPUT,
    OUTPUT,
    LINK, //with "LINK" I mean a tensor that is used to link two nodes, it means that is is the output of a node and the input of another
};

// Struct to represent a tensor that is ready for computation
pub const ReadyTensor = struct {
    name: []const u8,
    ready: bool,
    shape: []const i64,
    dtype: DataType = .UNDEFINED,
    tensorProto: ?*TensorProto = null,
    tag: TensorTag = TensorTag.LINK,

    pub fn createInitializer(tensorProto: *TensorProto) !ReadyTensor {
        return ReadyTensor{
            .name = tensorProto.name.?,
            .ready = true,
            .shape = tensorProto.dims,
            .tensorProto = tensorProto,
            .tag = TensorTag.INITIALIZER,
        };
    }

    pub fn createInput(name: []const u8) !ReadyTensor {
        return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = networkInput.shape,
            .tensorProto = null,
            .tag = TensorTag.INPUT,
        };
    }

    pub fn createConstant(name: []const u8, tensorProto: *TensorProto) !ReadyTensor {
        return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = networkInput.shape,
            .tensorProto = tensorProto,
            .tag = TensorTag.CONSTANT,
        };
    }

    pub fn createLink(name: []const u8) !ReadyTensor {
        return ReadyTensor{ //default
            .name = name,
            .ready = false,
            .shape = networkInput.shape,
            .tensorProto = null,
            .tag = TensorTag.LINK,
        };
    }

    pub fn print(tensor: *ReadyTensor, detailed: bool) void {
        globals_log.info("\n      READY TENSOR : {s}", .{tensor.name});
        globals_log.info("\n         status:{s}ready", .{if (!tensor.ready) " not " else " "});
        globals_log.info("\n         tag: {any}", .{tensor.tag});
        globals_log.info("\n         shape: {any}", .{tensor.shape});
        if (detailed) if (tensor.tensorProto) |tp| tp.print("         ") else globals_log.info("\n         tensor.tensorProto :(null)", .{});
    }
};

// Struct representing a computational node in the ONNX model
pub const ReadyNode = struct {
    nodeProto: *NodeProto,
    inputs: std.ArrayList(?*ReadyTensor),
    outputs: std.ArrayList(*ReadyTensor),
    ready: bool,

    // Creates a ReadyNode by preparing its input and output tensors
    pub fn create(nodeProto: *NodeProto) !ReadyNode {
        // globals_log.info("\n\nReadyNode.create() --> {s}", .{nodeProto.name.?});
        var newReadyNode = ReadyNode{
            .nodeProto = nodeProto,
            .inputs = std.ArrayList(?*ReadyTensor).init(allocator),
            .outputs = std.ArrayList(*ReadyTensor).init(allocator),
            .ready = false,
        };

        for (nodeProto.input) |input_name| { //for each input tensor in NodeProto

            //adding the readyTensor to the model
            if (std.mem.eql(u8, input_name, "")) {
                try newReadyNode.inputs.append(null);
            } else {
                try newReadyNode.inputs.append(if (tensorHashMap.getPtr(input_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            }
        }
        for (nodeProto.output) |output_name| { //for each output tensor

            //adding the readyTensor to the model
            try newReadyNode.outputs.append(if (tensorHashMap.getPtr(output_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // globals_log.info("\n   added output {s} to node {s} ", .{ output_name, nodeProto.name.? });
        }

        // -- COMPUTING THE OUTPUT SHAPE --
        try shapeGen.compute_output_shape(&newReadyNode);

        return newReadyNode;
    }

    pub fn print(node: *ReadyNode, detailed: bool) void {
        globals_log.info("\n ------ READY NODE : ", .{});
        if (detailed) node.nodeProto.print("  ") else globals_log.info("\n {s} ", .{node.nodeProto.name.?});
        globals_log.info("\n  ---inputs : ", .{});
        for (node.inputs.items) |in| if (in) |i| i.print(detailed) else globals_log.info("\n      NULL INPUT", .{});
        globals_log.info("\n  ---outputs : ", .{});
        for (node.outputs.items) |out| out.print(detailed);
    }
};

pub fn setGlobalAttributes(model: ModelOnnx) !void {
    //initializing global attributes
    onnxModel = model;

    //ready graph
    readyGraph.deinit();
    readyGraph = std.ArrayList(ReadyNode).init(allocator);

    //hash map
    tensorHashMap.deinit();
    tensorHashMap = std.StringHashMap(ReadyTensor).init(allocator);

    //First convert the optional String of numbers divided by a comma into an array
    const parsedInputshape: []const i64 = try utils.parseNumbers(codegen_options.shape);

    //setting the input
    const inputs = model.graph.?.inputs;
    networkInput.name = inputs[0].name.?;
    // record input shape
    networkInput.shape = inputs[0].type.?.tensor_type.?.shape.?.shape;
    // Derive and store the input element type string (e.g., "f32", "u8")
    const raw_et: u32 = inputs[0].type.?.tensor_type.?.elem_type;
    const int_val = @as(i32, @intCast(raw_et));
    const input_dt = @as(DataType, @enumFromInt(int_val));
    // Store the calculated DataType globally
    networkInputDataType = input_dt;
    networkInputTypeString = try utils.getTypeString(input_dt);

    //setting the output
    const outputs = model.graph.?.outputs;
    globals_log.info("\n SETTING networkOutput \n name = {s} \n shape={any}", .{ outputs[0].name.?, outputs[0].type.?.tensor_type.?.shape.?.shape });
    networkOutput.name = outputs[0].name.?;
    networkOutput.shape = outputs[0].type.?.tensor_type.?.shape.?.shape;

    // Use -Dshape if provided, otherwise keep the ONNX model's shape
    if (parsedInputshape.len > 0) {
        networkInput.shape = parsedInputshape;
    } else if (networkInput.shape.len == 0) {
        globals_log.warn("\n\n ERROR: \n     Input shape is necessary to proceed! \n     Ensure that the onnx model has one or compile with -Dshape=''<your_shape>''", .{});
        return error.NoInputShape;
    }

    // Print the final input details AFTER potentially overriding shape
    globals_log.info("\n FINAL networkInput \n name = {s} \n shape={any}", .{ networkInput.name, networkInput.shape });

    //create the hashMap
    try populateReadyTensorHashMap(model);

    //create the ReadyGraph
    try populateReadyGraph(model);
    // Initialize the tensor use counts (number of times each tensor is consumed)
    tensorUseCount.deinit();
    tensorUseCount = std.StringHashMap(usize).init(allocator);
    for (readyGraph.items) |*node| {
        for (node.inputs.items) |input_opt| {
            if (input_opt) |input| {
                const name = input.name;
                if (name.len > 0) {
                    const old_count = if (tensorUseCount.getPtr(name)) |ptr| ptr.* else 0;
                    try tensorUseCount.put(name, old_count + 1);
                }
            }
        }
    }

    globals_log.info("\n NODE: {s}", .{model.graph.?.nodes[0].output[0]});
}

// ----------------------- HASH MAP -----------------------
// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor
fn populateReadyTensorHashMap(model: ModelOnnx) !void {
    const protoGraph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    for (protoGraph.initializers) |init_ptr| {
        //create the readyTensor
        var readyTensor: ReadyTensor = try ReadyTensor.createInitializer(init_ptr);
        readyTensor.dtype = init_ptr.data_type;
        //add the readyTensor to the HashMap
        try tensorHashMap.put(readyTensor.name, readyTensor);
    }

    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto
        for (node.input) |input_name| {
            try addToTensorHashMap(input_name, node, protoGraph);
        }
        for (node.output) |output_name| {
            try addToTensorHashMap(output_name, node, protoGraph);
        }
    }
}

pub fn addToTensorHashMap(name: []const u8, nodeProto: *NodeProto, graph: *GraphProto) !void {
    if (tensorHashMap.get(name) != null or std.mem.eql(u8, name, "")) {
        return;
    } else {
        var readyTensor: ReadyTensor = undefined;
        var tensor_dtype: DataType = .UNDEFINED;

        //if input
        if (utils.isInput(name)) {
            readyTensor = try ReadyTensor.createInput(name);
            // Find dtype from graph inputs
            // Attempt to read the data type from graph inputs
            for (graph.inputs) |graph_input| {
                if (std.mem.eql(u8, graph_input.name.?, name)) {
                    const raw_et: u32 = graph_input.type.?.tensor_type.?.elem_type;
                    const int_val_in = @as(i32, @intCast(raw_et));
                    tensor_dtype = @as(DataType, @enumFromInt(int_val_in));
                    break;
                }
            }
        }
        //if constant, pay attention, we add the Constatant only if it is a TENSOR (aka AttributeProto.t)
        else if (std.mem.eql(u8, nodeProto.op_type, "Constant")) {
            //add the readyTensor to the HashMap
            if (nodeProto.attribute.len > 0 and nodeProto.attribute[0].type == onnx.AttributeType.TENSOR) {
                const const_tensor_proto = nodeProto.attribute[0].t.?;
                readyTensor = try ReadyTensor.createConstant(name, const_tensor_proto);
                tensor_dtype = const_tensor_proto.data_type;
            } else {
                // Handle non-tensor constants if necessary, or assume LINK for now
                readyTensor = try ReadyTensor.createLink(name);
                // Try to find dtype from value_info for non-tensor constants if needed
                // Try to infer dtype from value_info
                for (graph.value_info) |vi| {
                    if (vi.name) |vi_name| {
                        if (std.mem.eql(u8, vi_name, name)) {
                            if (vi.type) |t| {
                                if (t.tensor_type) |tt| {
                                    const raw_et_vi_link = tt.elem_type;
                                    const int_val_vi_link = @as(i32, @intCast(raw_et_vi_link));
                                    tensor_dtype = @as(DataType, @enumFromInt(int_val_vi_link));
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        //else default (LINK)
        else {
            readyTensor = try ReadyTensor.createLink(name);
            // Find dtype from value_info for LINK tensors
            var found_in_value_info = false;
            // Also check value_info for LINK tensors
            for (graph.value_info) |vi| {
                // Check if vi.name matches and is not null
                if (vi.name) |vi_name| {
                    if (std.mem.eql(u8, vi_name, name)) {
                        // Safely access type and tensor_type
                        if (vi.type) |t| {
                            if (t.tensor_type) |tt| {
                                const raw_et_vi_link = tt.elem_type;
                                const int_val_vi_link = @as(i32, @intCast(raw_et_vi_link));
                                tensor_dtype = @as(DataType, @enumFromInt(int_val_vi_link));
                                found_in_value_info = true;
                                // Found the type, exit the loop
                                break;
                            }
                        }
                        // Break if name matches, even if type info wasn't found/complete
                        break;
                    }
                }
            }
            // Also check graph outputs if not found in value_info
            if (!found_in_value_info) {
                // Finally check graph outputs
                for (graph.outputs) |graph_output| {
                    // Check if graph_output.name matches and is not null
                    if (graph_output.name) |output_name| {
                        if (std.mem.eql(u8, output_name, name)) {
                            // Safely access type and tensor_type
                            if (graph_output.type) |t| {
                                if (t.tensor_type) |tt| {
                                    const raw_et_out = tt.elem_type;
                                    const int_val_out = @as(i32, @intCast(raw_et_out));
                                    tensor_dtype = @as(DataType, @enumFromInt(int_val_out));
                                    // Found the type, exit the loop
                                    break;
                                }
                            }
                            // Break if name matches, even if type info wasn't found/complete
                            break;
                        }
                    }
                }
            }

            // --- START HEURISTIC FALLBACK FOR SHAPE TENSORS ---
            // If type is still undefined, check common shape tensor naming patterns
            if (tensor_dtype == .UNDEFINED) {
                if (std.mem.endsWith(u8, name, "_shape")) {
                    globals_log.info("\nINFO: Tensor '{s}' type is UNDEFINED. Defaulting to INT64 based on name pattern (likely a shape tensor).", .{name});
                    tensor_dtype = .INT64; // Default to INT64 for likely shape tensors
                }
            }
            // --- END HEURISTIC FALLBACK ---
        }

        // --- START TYPE OVERRIDE FOR SPECIFIC OPS ---
        if (std.mem.eql(u8, nodeProto.op_type, "DynamicQuantizeLinear")) {
            if (nodeProto.output.len >= 3) { // Check if node has expected outputs
                if (std.mem.eql(u8, name, nodeProto.output[0])) {
                    globals_log.info("\nINFO: Overriding dtype for DynamicQuantizeLinear output y '{s}' to UINT8.", .{name});
                    tensor_dtype = .UINT8; // y output is u8
                } else if (std.mem.eql(u8, name, nodeProto.output[1])) {
                    globals_log.info("\nINFO: Overriding dtype for DynamicQuantizeLinear output y_scale '{s}' to FLOAT.", .{name});
                    tensor_dtype = .FLOAT; // y_scale output is f32
                } else if (std.mem.eql(u8, name, nodeProto.output[2])) {
                    globals_log.info("\nINFO: Overriding dtype for DynamicQuantizeLinear output y_zero_point '{s}' to UINT8.", .{name});
                    tensor_dtype = .UINT8; // y_zero_point output is u8
                }
            }
        }
        // Add specific override for ConvInteger output type
        else if (std.mem.eql(u8, nodeProto.op_type, "ConvInteger")) {
            if (nodeProto.output.len > 0 and std.mem.eql(u8, name, nodeProto.output[0])) {
                globals_log.info("\nINFO: Overriding dtype for ConvInteger output '{s}' to INT32.", .{name});
                tensor_dtype = .INT32; // ConvInteger output is always i32
            }
        }
        // Add overrides for other ops if needed (e.g., Cast might benefit too)
        // --- END TYPE OVERRIDE FOR SPECIFIC OPS ---

        if (tensor_dtype == .UNDEFINED) {
            globals_log.warn("\nWARNING: Could not determine dtype for tensor '{s}' (Node: {s}). Defaulting to FLOAT.", .{ name, nodeProto.name orelse "unnamed" });
            // Assign a default type instead of leaving it undefined
            tensor_dtype = .FLOAT;
            // Optionally return an error here if type is mandatory
            // return error.DataTypeNotFoundForTensor;
        }

        readyTensor.dtype = tensor_dtype;
        //add the readyTensor to the HashMap
        try tensorHashMap.put(name, readyTensor);
    }
}

// ----------------------- READY GRAPH -----------------------
// Creates a graph representation with all nodes in a ready-to-compute state
fn populateReadyGraph(model: ModelOnnx) !void {
    const graph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    for (graph.nodes) |node_ptr| { //for each NodeProto in the GraphProto

        try readyGraph.append(try ReadyNode.create(node_ptr));
    }
}
// Decrements the remaining use count for a tensor. Returns updated count (0 if none or unknown).
pub fn decrementUseCount(name: []const u8) usize {
    if (tensorUseCount.getPtr(name)) |ptr| {
        ptr.* -= 1;
        return ptr.*;
    } else {
        return 0;
    }
}
