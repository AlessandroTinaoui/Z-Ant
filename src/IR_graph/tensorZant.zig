const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
pub const AnyTensor = zant.core.tensor.AnyTensor;

const utils = @import("utils.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const ValueInfoProto = onnx.ValueInfoProto;

const allocator = std.heap.page_allocator;

pub const TensorType = enum {
    // Floating point types
    f16,
    f32,
    f64,

    // Signed integer types
    i8,
    i16,
    i32,
    i64,

    // Unsigned integer types
    u8,
    u16,
    u32,
    u64,

    // Boolean (often used for masks)
    bool,

    undefined,

    pub fn toString(self: TensorType) []const u8 {
        return switch (self) {
            .f16 => "f16",
            .f32 => "f32",
            .f64 => "f64",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .bool => "bool",
            .undefined => "undefined",
        };
    }
};

pub const TensorCategory = enum {
    INPUT,
    OUTPUT,
    INITIALIZER,
    LINK,
    CONSTANT,

    pub fn toString(self: TensorCategory) []const u8 {
        return switch (self) {
            .INPUT => ".INPUT",
            .OUTPUT => ".OUTPUT",
            .INITIALIZER => ".INITIALIZER",
            .LINK => ".LINK",
            .CONSTANT => ".CONSTANT",
        };
    }
};

pub const TensorZant = struct {
    name: []const u8,
    ty: TensorType,
    tc: TensorCategory,
    ptr: ?*AnyTensor,
    shape: []usize,
    stride: []usize,

    pub fn init(name: []const u8, tensorProto: ?*TensorProto, value_info: ?*ValueInfoProto, shape: ?[]usize, tensorCategory: TensorCategory) !TensorZant {
        std.debug.print("\n ----------- init({s}, {s}, {s}, {s}) ", .{ name, if (tensorProto) |_| "tp" else "null", if (value_info) |_| "vi" else "null", tensorCategory.toString() });

        var tensor: ?*AnyTensor = null;
        var shape_i64: []i64 = undefined;
        var shape_usize: []usize = undefined;

        if (tensorProto) |tp| { //if the tensorProto is given it means that the tensor we are initializing is
            tensor = try allocator.create(AnyTensor);
            tensor.?.* = try utils.protoTensor2AnyTensor(tp); //create the ptr to AnyTensor
            shape_usize = tensor.?.get_shape(); //saves the shape
        } else if (value_info) |vi| {
            shape_i64 = if (utils.getTensorShapeFromValueInfo(vi)) |s| s else { //search for the shape
                std.debug.print("\n ERROR: {s} value_info shape not found ", .{name});
                return error.shapeNotfound;
            };
            shape_usize = try utils.i64SliceToUsizeSlice(shape_i64); //saves the shape
        } else if (shape) |s| {
            shape_usize = s;
        } else {
            std.debug.print("\n ERROR: {s} not found ", .{name});
            return error.shapeNotfound;
        }

        std.debug.print("\n                shape:{any} ", .{shape_usize});

        return TensorZant{
            .name = name,
            .ty = if (tensor) |t| utils.getAnyTensorType(t.*) else TensorType.undefined, //if .ty is set to undefined it means that it is a "link" tensor between 2 nodes, the .ty must be set when the nodes are created
            .tc = tensorCategory,
            .ptr = tensor,
            .shape = shape_usize,
            .stride = try TensorZant.computeStride(shape_usize),
        };
    }

    pub fn deint(self: *TensorZant) void {
        if (self.ptr) |at| {
            at.deint();
        }

        allocator.free(self.ptr);
        allocator.free(self.name);
        allocator.free(self.shape);
        allocator.free(self.stride);
    }

    pub fn getShape(self: *TensorZant) []usize {
        return self.shape;
    }

    pub fn getStride(self: *TensorZant) []usize {
        return self.stride;
    }

    pub fn computeStride(shape: []usize) ![]usize {
        const num_dims = shape.len;
        var strides = try allocator.alloc(usize, num_dims);

        var i: usize = num_dims - 1;
        if (i >= 0) strides[i] = 1 else return strides;

        while (i > 0) {
            strides[i - 1] = strides[i] * shape[i];
            i -= 1;
        }
        return strides;
    }

    pub fn set_tensorType(self: *TensorZant, ty: TensorType) void {
        if (ty == TensorType.undefined) {
            std.debug.print("\n ERROR: illegal behavior! you cannot set a tensor type to undefined! ", .{});
            return error.illegalBehavior;
        }
        self.ty = ty;
    }

    // Returns the id of a tensorZant from the hashMap
    pub fn get_tensorZantID(self: *TensorZant) usize {
        var hasher = std.hash.Wyhash.init(0);
        std.hash.autoHashStrat(&hasher, self.name, .Deep);
        return @as(usize, @intCast(hasher.final()));
    }
};

// ----------------------- HASH MAP -----------------------
pub var tensorMap: std.StringHashMap(TensorZant) = std.StringHashMap(TensorZant).init(allocator);

// Returns the id of a tensorZant from the hashMap
pub fn get_tensorZantID(self: TensorZant) u64 {
    var hasher = std.hash.Wyhash.init(0);
    std.hash.hash(&hasher, self.name, .Deep);
    return hasher.final();
}

// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor
pub fn initialize_tensorZantMap(modelProto: *ModelProto) !void {
    //free and reinit the tensorMap
    tensorMap.deinit();
    tensorMap = std.StringHashMap(TensorZant).init(allocator);

    std.debug.print("\n ---- initialize_tensorZantMap ---- ", .{});

    const protoGraph = try if (modelProto.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    std.debug.print("\n -------- initializers ", .{});

    //initializer : *TensorProto,
    for (protoGraph.initializers) |init_ptr| {
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            init_ptr.name.?,
            init_ptr,
            null,
            null,
            TensorCategory.INITIALIZER,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding inputs to the hash map
    std.debug.print("\n -------- inputs ", .{});

    for (protoGraph.inputs) |inputs_ptr| { //inputs : *ValueInfoProto,
        if (tensorMap.getPtr(inputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            inputs_ptr.name.?,
            null,
            inputs_ptr,
            null,
            TensorCategory.INPUT,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding outputs to the hash map
    std.debug.print("\n -------- outputs ", .{});

    for (protoGraph.outputs) |outputs_ptr| { //outputs : *ValueInfoProto,
        if (tensorMap.getPtr(outputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            outputs_ptr.name.?,
            null,
            outputs_ptr,
            null,
            TensorCategory.OUTPUT,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    std.debug.print("\n -------- nodes ", .{});
    //adding all the nodes inputs and outputs
    for (protoGraph.nodes, 0..) |node, i| { //for each NodeProto in the GraphProto
        std.debug.print("\n --- {} ", .{i});
        // node.print(null); //DEBUG

        //WHy CONSTANT nodes need a different initialization?
        if (std.mem.eql(u8, node.op_type, "Constant")) {
            const tensorZant: TensorZant = try TensorZant.init(
                node.output[0],
                node.attribute[0].t.?,
                null,
                null,
                TensorCategory.CONSTANT,
            );
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        } else {
            for (node.input) |input_name| {
                //if the tensor is null is represented by an empty string in the onnx, so It must not be initialized in the hashMap
                if (std.mem.eql(u8, input_name, "")) continue;
                //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
                if (tensorMap.getPtr(input_name) != null) continue;

                //create the readyTensor
                const tensorZant: TensorZant = try TensorZant.init(
                    input_name,
                    null,
                    utils.getValueInfoTensorFromGraphInfo(input_name, protoGraph),
                    null,
                    TensorCategory.LINK,
                );
                //add the readyTensor to the HashMap
                try tensorMap.put(tensorZant.name, tensorZant);
            }
            for (node.output) |output_name| {

                // //WHy RESHAPE nodes need a different initialization?
                // if (std.mem.eql(u8, node.op_type, "Reshape")) {
                //     var shape = try allocator.alloc(usize, node.attribute[0].ints.len);
                //     for (node.attribute[0].ints, 0..) |dim, j| {
                //         shape[j] = @as(usize, @intCast(dim));
                //     }
                //     const tensorZant: TensorZant = try TensorZant.init(
                //         node.output[0],
                //         null,
                //         null,
                //         shape,
                //         TensorCategory.CONSTANT,
                //     );
                //     //add the readyTensor to the HashMap
                //     try tensorMap.put(tensorZant.name, tensorZant);
                // }
                //if the tensor is null is represented by an empty string in the onnx, so It must not be initialized in the hashMap
                if (std.mem.eql(u8, output_name, "")) continue;
                //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
                if (tensorMap.getPtr(output_name) != null) continue;
                //create the readyTensor
                const tensorZant: TensorZant = try TensorZant.init(
                    output_name,
                    null,
                    utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph),
                    null,
                    TensorCategory.LINK,
                );
                //add the readyTensor to the HashMap
                try tensorMap.put(tensorZant.name, tensorZant);
            }
        }
    }
}
