const std = @import("std");
const os = std.os;

const zant = @import("zant");

const Codegen_log = std.log.scoped(.shape_handler);

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const allocator = zant.utils.allocator.allocator;

// --- proto libs
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeType = onnx.AttributeType;

// --- codeGen libs
const ReadyNode = @import("globals.zig").ReadyNode;
const ReadyTensor = @import("globals.zig").ReadyTensor;
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const codegen_options = @import("codegen_options");
const globals = @import("globals.zig");

// ----------------------------------- SHAPE inference -----------------------------------

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    // Ensure the node has a name for debugging purposes
    if (readyNode.nodeProto.name == null) {
        // Generate a name like "OpType_OutputName"
        const op_type = readyNode.nodeProto.op_type;
        const output_name = readyNode.outputs.items[0].name; // Directly assign since it's not optional
        _ = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ op_type, output_name }); // Keep allocation for potential local use or logging if needed, but don't assign. Free later if stored.
        // Note: This allocated name needs to be managed if the NodeProto lifetime extends beyond this scope.
        // Assuming the global allocator lives long enough or NodeProto is processed quickly.
    }

    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        //https://onnx.ai/onnx/operators/onnx__Add.html
        try compute_Add_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "AveragePool")) {
        try compute_averagePool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "BatchNormalization")) {
        //https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
        try compute_batchNormalization_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Cast")) {
        // https://onnx.ai/onnx/operators/onnx__Cast.html
        try compute_cast_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Ceil")) {
        //https://onnx.ai/onnx/operators/onnx__Ceil.html
        try compute_ceil_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Clip")) {
        //https://onnx.ai/onnx/operators/onnx__Clip.html
        try compute_clip_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        //https://onnx.ai/onnx/operators/onnx__Concat.html
        try compute_concat_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Constant")) {
        //https://onnx.ai/onnx/operators/onnx__Constant.html
        try compute_constant_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Conv")) {
        //https://onnx.ai/onnx/operators/onnx__Conv.html
        try compute_conv_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "ConvInteger")) {
        //https://onnx.ai/onnx/operators/onnx__ConvInteger.html
        try compute_convInteger_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Div")) {
        //https://onnx.ai/onnx/operators/onnx__Div.html
        try compute_Div_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "DynamicQuantizeLinear")) {
        // https://onnx.ai/onnx/operators/onnx_aionnx_preview_training__DynamicQuantizeLinear.html
        try compute_dynamicQuantizeLinear_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Elu")) {
        //https://onnx.ai/onnx/operators/onnx__Elu.html
        try compute_elu_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Flatten")) {
        //https://onnx.ai/onnx/operators/onnx__Flatten.html
        try compute_flatten_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Floor")) {
        //https://onnx.ai/onnx/operators/onnx__Floor.html
        try compute_floor_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Squeeze")) {
        //https://onnx.ai/onnx/operators/onnx__Squeeze.html
        try compute_squeeze_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        try compute_gather_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        //https://onnx.ai/onnx/operators/onnx__Gemm.html
        try compute_gemm_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gelu")) {
        //https://onnx.ai/onnx/operators/onnx__Gelu.html
        try compute_gelu_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        try compute_leaky_relu_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        try compute_longsoftmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        try compute_matmul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        try compute_maxPool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        //https://onnx.ai/onnx/operators/onnx__Mul.html
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Neg")) {
        //https://onnx.ai/onnx/operators/onnx__Neg.html
        try compute_neg_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        //https://onnx.ai/onnx/operators/onnx__OneHot.html
        try compute_oneHot_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Pad")) {
        //https://onnx.ai/onnx/operators/onnx__Pad.html
        try compute_pads_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "ReduceMean")) {
        try compute_reducemean_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        //https://onnx.ai/onnx/operators/onnx__Relu.html
        try compute_ReLU_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // https://onnx.ai/onnx/operators/onnx__Reshape.html
        try compute_reshape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        try compute_resize_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        try compute_shape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        try compute_sigmoid_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        //https://onnx.ai/onnx/operators/onnx__Softmax.html
        try compute_softmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        try compute_slice_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        //https://onnx.ai/onnx/operators/onnx__Split.html
        try compute_split_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sqrt")) {
        try compute_sqrt_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        //https://onnx.ai/onnx/operators/onnx__Sub.html
        try compute_Sub_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Tanh")) {
        try compute_tanh_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        try compute_transpose_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Unsqueeze")) {
        try compute_unsqueeze_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Identity")) {
        //https://onnx.ai/onnx/operators/onnx__Identity.html
        try compute_identity_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mean")) {
        // https://onnx.ai/onnx/operators/onnx__Mean.html
        try compute_mean_output_shape(readyNode);
    } else {
        Codegen_log.warn("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------
inline fn compute_Add_output_shape(readyNode: *ReadyNode) !void {
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = readyNode.inputs.items[0].?.shape;
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_batchNormalization_output_shape(readyNode: *ReadyNode) !void {
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_batchNormalization_output_shape(try utils.i64SliceToUsizeSlice(readyNode.inputs.items[0].?.shape)));
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_cast_output_shape(readyNode: *ReadyNode) !void {
    // Cast is an element-wise operation, output shape is identical to input shape
    Codegen_log.info("\n====== compute_cast_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        // Cast operation preserves the input shape
        shape = try allocator.dupe(i64, input_shape);
        Codegen_log.info("\n output_shape: []i64 = {any}", .{shape});
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_Sub_output_shape(readyNode: *ReadyNode) !void {
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = readyNode.inputs.items[0].?.shape;
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_constant_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_constant_output_shape node: {s}======", .{readyNode.nodeProto.name.?});

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        readyNode.outputs.items[0].shape = tensorShape;
        return;
    } else {
        // Check each possible attribute type for the Constant node
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "value")) {
                // Handle tensor value - use existing utility
                const shape = try utils.getConstantTensorDims(readyNode.nodeProto);

                // If the shape is empty (scalar in ONNX), use [1] instead
                if (shape.len == 0) {
                    readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
                } else {
                    readyNode.outputs.items[0].shape = shape;
                }

                Codegen_log.info("\n output_shape from tensor: []i64 = {any}", .{readyNode.outputs.items[0].shape});
                return;
            } else if (std.mem.eql(u8, attr.name, "value_float") or std.mem.eql(u8, attr.name, "value_int") or
                std.mem.eql(u8, attr.name, "value_string"))
            {
                // These are scalar values - output shape is [1]
                readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
                Codegen_log.info("\n output_shape scalar: []i64 = {any}", .{readyNode.outputs.items[0].shape});
                return;
            } else if (std.mem.eql(u8, attr.name, "value_floats") or std.mem.eql(u8, attr.name, "value_ints")) {
                // These are 1D arrays - shape is [length]
                var length: i64 = 0;
                if (attr.type == AttributeType.FLOATS) {
                    length = @intCast(attr.floats.len);
                } else if (attr.type == AttributeType.INTS) {
                    length = @intCast(attr.ints.len);
                }
                readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{length});
                Codegen_log.info("\n output_shape 1D array: []i64 = {any}", .{readyNode.outputs.items[0].shape});
                return;
            } else if (std.mem.eql(u8, attr.name, "value_strings")) {
                // 1D array of strings - shape is [length]
                const length: i64 = @intCast(attr.strings.len);
                readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{length});
                Codegen_log.info("\n output_shape string array: []i64 = {any}", .{readyNode.outputs.items[0].shape});
                return;
            } else if (std.mem.eql(u8, attr.name, "sparse_value")) {
                // For sparse tensor, we need to handle it differently
                Codegen_log.warn("\n Warning: Sparse tensor support is limited", .{});

                // Use a placeholder shape for sparse tensors - assuming scalar for now
                readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
                Codegen_log.info("\n output_shape from sparse tensor (placeholder): []i64 = {any}", .{readyNode.outputs.items[0].shape});
                return;
            }
        }
    }

    return error.ConstantValueNotFound;
}

inline fn compute_ReLU_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_ReLU_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    Codegen_log.info("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].?.shape});

    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = readyNode.inputs.items[0].?.shape;
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_reshape_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_reshape_output_shape node: {s}======", .{readyNode.nodeProto.name orelse "(unnamed)"});
    const input_rt: *globals.ReadyTensor = readyNode.inputs.items[0].?;
    const input_shape_i64 = input_rt.shape;
    Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape_i64});

    var new_shape_spec: []const isize = undefined; // Use []const isize as required by get_reshape_output_shape
    var shape_spec_found: bool = false;
    var shape_input_needs_free = false; // Flag to track if we allocated new_shape_spec
    var allow_zero: bool = false;

    // 1. Get allowzero attribute (default 0 -> false)
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "allowzero")) {
            if (attr.type == AttributeType.INT and attr.i != 0) {
                allow_zero = true;
            }
            break; // Found allowzero, no need to check other attributes for this
        }
    }
    Codegen_log.debug("\n allowzero: {}", .{allow_zero});

    // 2. Get the target shape spec (new_shape_spec)
    // Try getting shape from the second input tensor first
    if (readyNode.inputs.items.len > 1) {
        const shape_input = readyNode.inputs.items[1].?;
        if (shape_input.tensorProto != null and shape_input.tensorProto.?.int64_data != null) {
            // Shape is in the tensorProto data (preferred)
            new_shape_spec = shape_input.tensorProto.?.int64_data.?;
            shape_spec_found = true;
            Codegen_log.debug("\n new shape spec from input tensorProto: []i64 = {any}", .{new_shape_spec});
        } else if (shape_input.tensorProto != null and shape_input.tensorProto.?.int64_data == null) {
            const proto = shape_input.tensorProto.?;
            // Check data type - Reshape requires INT64 shape
            if (proto.data_type != .INT64) {
                Codegen_log.warn("ERROR: Reshape shape input tensorProto has incorrect data type: {any}. Expected INT64.", .{proto.data_type});
                return error.InvalidShapeDataType;
            }

            // Try reading from raw_data if int64_data is null
            if (proto.raw_data) |raw| {
                Codegen_log.debug("\n Shape input tensorProto has raw_data ({} bytes), attempting to parse as i64...", .{raw.len});
                // Call a new utility function to parse raw_data
                const parsed_shape = utils.parseI64RawData(raw) catch |err| {
                    Codegen_log.warn("\n ERROR: Failed to parse raw_data for shape tensor: {any}", .{err});
                    return error.RawDataParseFailed; // Or specific error from parsing
                };
                // Important: parsed_shape is allocated by the util func and needs freeing later.
                // Convert []i64 to []const isize for new_shape_spec
                var temp_shape_spec = try allocator.alloc(isize, parsed_shape.len);
                for (parsed_shape, 0..) |dim, i| {
                    temp_shape_spec[i] = dim;
                }
                new_shape_spec = temp_shape_spec; // Assign the parsed shape
                shape_spec_found = true;
                shape_input_needs_free = true; // Mark that we allocated this spec
                Codegen_log.debug("\n new shape spec parsed from raw_data: []isize = {any}", .{new_shape_spec});
                // We also need to free the intermediate parsed_shape ([]i64)
                defer allocator.free(parsed_shape);
            } else {
                // Data type is INT64, but int64_data is null and raw_data is null/empty.
                Codegen_log.warn("ERROR: Reshape shape input tensorProto is INT64 but contains no int64_data or raw_data.", .{});
                return error.ShapeDataMissing;
            }
        } else {
            // If tensorProto is null, this input doesn't directly provide the shape data.
            // shape_spec_found remains false, attributes will be checked.
            Codegen_log.debug("\n Shape input tensorProto is null, will check attributes.", .{});
        }
    } else {
        // If no second input, try getting shape from the 'shape' attribute
        // shape_spec_found remains false, attributes will be checked.
        Codegen_log.debug("\n No second input for shape, will check attributes.", .{});
    }

    // If new_shape_spec is still null after checking input, check attributes
    if (!shape_spec_found) {
        var shape_attr: ?[]const i64 = null;
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "shape")) {
                if (attr.type == AttributeType.INTS) {
                    shape_attr = attr.ints;
                    break;
                } else {
                    Codegen_log.warn("ERROR: Reshape 'shape' attribute has unexpected type {}", .{attr.type});
                    return error.InvalidAttributeType;
                }
            }
        }

        if (shape_attr) |sa| {
            var temp_shape_spec = try allocator.alloc(isize, sa.len);
            for (sa, 0..) |dim, i| {
                temp_shape_spec[i] = dim;
            }
            new_shape_spec = temp_shape_spec;
            shape_input_needs_free = true; // Mark that we allocated this
            shape_spec_found = true;
            Codegen_log.debug("\n new shape spec from attribute: []isize = {any}", .{new_shape_spec});
        } else {
            // Default behavior: use the inverse of the input shape
            Codegen_log.debug("No shape tensor or attribute found, using inverse of input shape as default.", .{});

            // Allocate space for the inverse shape
            var temp_shape_spec = try allocator.alloc(isize, input_shape_i64.len);

            // Create inverse shape by reversing the dimensions
            for (input_shape_i64, 0..) |dim, i| {
                const inverse_index = input_shape_i64.len - 1 - i;
                temp_shape_spec[inverse_index] = dim;
            }

            new_shape_spec = temp_shape_spec;
            shape_input_needs_free = true; // Mark that we allocated this
            shape_spec_found = true;
            Codegen_log.debug("\n using inverse shape as default: []isize = {any}", .{new_shape_spec});
        }
    } else {
        // Ensure cleanup if we allocated the shape spec FROM THE INPUT PATH
        // The defer covers the attribute path if allocation happens there.
        // NOTE: This defer placement might be tricky. Consider simplifying allocation management.
        defer if (shape_input_needs_free) allocator.free(new_shape_spec);
    }

    // If after all checks, shape_spec is still not found, something went wrong.
    if (!shape_spec_found) {
        Codegen_log.debug("Critical Error: Shape spec was not found after checking inputs and attributes.", .{});
        return error.ShapeNotFound;
    }

    // 3. Convert input shape to usize
    const input_shape_usize = try utils.i64SliceToUsizeSlice(input_shape_i64);
    defer allocator.free(input_shape_usize);
    Codegen_log.info("\n input_shape_usize: []usize = {any}", .{input_shape_usize});

    // 4. Call the new shape calculation function
    const output_shape_usize = try tensorMath.get_reshape_output_shape(input_shape_usize, new_shape_spec, allow_zero);
    defer allocator.free(output_shape_usize); // Free the result from get_reshape_output_shape
    Codegen_log.info("\n calculated output_shape_usize: []usize = {any}", .{output_shape_usize});

    // 5. Convert result back to i64
    Codegen_log.debug("\n >>> DEBUG: output_shape_usize before conversion: {any}\n", .{output_shape_usize});
    const output_shape_i64 = try utils.usizeSliceToI64Slice(output_shape_usize);
    // NOTE: utils.usizeSliceToI64Slice allocates, so the caller (or ReadyNode deinit) should free it.

    // 6. Assign the final shape
    readyNode.outputs.items[0].shape = output_shape_i64;
    Codegen_log.info("\n final output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_softmax_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_softmax_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    Codegen_log.info("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].?.shape});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = readyNode.inputs.items[0].?.shape;
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_gemm_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_gemm_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    Codegen_log.info("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].?.shape});
    Codegen_log.debug("\n weight_shape: []i64 = {any}", .{readyNode.inputs.items[1].?.shape});
    Codegen_log.debug("\n bias_shape: []i64 = {any}", .{readyNode.inputs.items[2].?.shape});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = readyNode.inputs.items[2].?.shape;
    }

    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_oneHot_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_oneHot_output_shape node: {s} ======\n", .{readyNode.nodeProto.name orelse "(unnamed)"});

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        // Verifica che ci siano esattamente 3 input: indices, depth, values
        if (readyNode.inputs.items.len != 3) {
            std.debug.print("\n ERROR: OneHot expects exactly 3 inputs, got {d}\n", .{readyNode.inputs.items.len});
            return error.InvalidNumberOfInputs;
        }

        const indices = readyNode.inputs.items[0].?;
        const depth_tensor = readyNode.inputs.items[1].?;
        const values = readyNode.inputs.items[2].?;

        std.debug.print("\n indices_shape: []i64 = {any}", .{indices.shape});
        std.debug.print("\n depth_shape: []i64 = {any}", .{depth_tensor.shape});
        std.debug.print("\n values_shape: []i64 = {any}", .{values.shape});

        // Verifica che depth sia uno scalare (forma [] o [1])
        const depth_shape_i64 = depth_tensor.shape;
        const effective_depth_shape_i64 = if (depth_shape_i64.len == 0) &[_]i64{1} else depth_shape_i64;
        if (effective_depth_shape_i64.len > 1 or effective_depth_shape_i64[0] != 1) {
            std.debug.print("\n ERROR: depth must be a scalar, got shape {any}\n", .{effective_depth_shape_i64});
            return error.InvalidDepthShape;
        }

        // Verifica che values abbia forma [2]
        const values_shape_i64 = values.shape;
        const effective_values_shape_i64 = if (values_shape_i64.len == 0) &[_]i64{1} else values_shape_i64;
        if (effective_values_shape_i64.len != 1 or effective_values_shape_i64[0] != 2) {
            std.debug.print("\n ERROR: values must have shape [2], got shape {any}\n", .{effective_values_shape_i64});
            return error.InvalidValuesShape;
        }

        // Estrai il valore di depth
        var depth: i64 = undefined;
        if (depth_tensor.tensorProto != null and depth_tensor.tensorProto.?.int64_data != null) {
            depth = depth_tensor.tensorProto.?.int64_data.?[0];
        } else if (depth_tensor.tensorProto != null and depth_tensor.tensorProto.?.raw_data != null) {
            const raw = depth_tensor.tensorProto.?.raw_data.?;
            if (raw.len < @sizeOf(i64)) {
                std.debug.print("\n ERROR: depth raw_data is too small to contain an i64\n", .{});
                return error.InvalidDepthData;
            }
            depth = std.mem.readInt(i64, raw[0..@sizeOf(i64)], .little);
        } else {
            std.debug.print("\n ERROR: depth tensorProto is missing valid data\n", .{});
            return error.DepthDataMissing;
        }

        // Verifica che depth sia positivo
        if (depth <= 0) {
            std.debug.print("\n ERROR: depth must be positive, got {d}\n", .{depth});
            return error.InvalidDepthValue;
        }

        // Estrai l'attributo axis (default: -1)
        var axis: i64 = -1;
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != AttributeType.INT) {
                    std.debug.print("\n ERROR: axis attribute must be INT, got type {any}\n", .{attr.type});
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
                break;
            }
        }

        const indices_shape_i64 = indices.shape;
        const indices_shape_usize = try utils.i64SliceToUsizeSlice(indices_shape_i64);
        defer allocator.free(indices_shape_usize);

        const output_shape_usize = try tensorMath.get_oneHot_output_shape(indices_shape_usize, depth, axis);
        defer allocator.free(output_shape_usize);

        shape = try utils.usizeSliceToI64Slice(output_shape_usize);
    }

    readyNode.outputs.items[0].shape = shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_mul_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_mul_output_shape node: {s} ======\n", .{readyNode.nodeProto.name.?});

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_a = readyNode.inputs.items[0];
        const input_b = readyNode.inputs.items[1];

        Codegen_log.info("\n input_a_shape: []i64 = {any}", .{input_a.?.shape});
        Codegen_log.info("\n input_b_shape: []i64 = {any}", .{input_b.?.shape});

        const shape_a_i64 = input_a.?.shape;
        const shape_b_i64 = input_b.?.shape;

        // Handle empty shapes by treating them as {1} for broadcasting calculation
        const effective_shape_a_i64 = if (shape_a_i64.len == 0) &[_]i64{1} else shape_a_i64;
        const effective_shape_b_i64 = if (shape_b_i64.len == 0) &[_]i64{1} else shape_b_i64;

        // Convert effective shapes to usize
        const shape_a_usize = try utils.i64SliceToUsizeSlice(effective_shape_a_i64);
        const shape_b_usize = try utils.i64SliceToUsizeSlice(effective_shape_b_i64);

        // Use TensorMath to compute the output shape using effective shapes
        const output_shape_usize = try tensorMath.get_mul_output_shape(shape_a_usize, shape_b_usize);

        // Defer freeing the intermediate usize slices *after* they've been used
        if (shape_a_i64.len != 0) {
            defer allocator.free(shape_a_usize);
        }
        if (shape_b_i64.len != 0) {
            defer allocator.free(shape_b_usize);
        }
        // Defer freeing the result from get_mul_output_shape *after* it has been used for conversion
        defer allocator.free(output_shape_usize);

        // Convert the result back to i64
        shape = try utils.usizeSliceToI64Slice(output_shape_usize);
        // Note: The memory for 'shape' is now owned by the caller/ReadyNode management
    }

    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_conv_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_conv_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape: []const i64 = readyNode.inputs.items[0].?.shape;
        const kernel_shape: []const i64 = readyNode.inputs.items[1].?.shape;

        var stride: ?[]i64 = null;
        var dilation: ?[]i64 = null;
        var auto_pad: []const u8 = "NOTSET";
        var pads: ?[]i64 = null;
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "strides")) {
                if (attr.type == AttributeType.INTS) stride = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "dilations")) {
                if (attr.type == AttributeType.INTS) dilation = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "auto_pad")) {
                if (attr.type == AttributeType.STRING) auto_pad = attr.s;
            }
            if (std.mem.eql(u8, attr.name, "pads")) {
                if (attr.type == AttributeType.INTS) pads = attr.ints;
            }
        }

        if (stride == null) return error.StridesNotFound;
        if (dilation == null) return error.DilationsNotFound;

        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        Codegen_log.debug("\n kernel_shape: []i64 = {any}", .{kernel_shape});
        Codegen_log.debug("\n stride: []i64 = {any}", .{stride.?});
        //Codegen_log.debug("\n pads: []i64 = {any}", .{pads.?});
        shape = try utils.usizeSliceToI64Slice(
            @constCast(
                &try tensorMath.get_convolution_output_shape(
                    try utils.i64SliceToUsizeSlice(input_shape),
                    try utils.i64SliceToUsizeSlice(kernel_shape),
                    try utils.i64SliceToUsizeSlice(stride.?),
                    if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
                    try utils.i64SliceToUsizeSlice(dilation.?),
                    auto_pad,
                ),
            ),
        );
    }
    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_maxPool_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_maxPool_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].?.shape;
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        var kernel_shape: ?[]i64 = null;
        var stride: ?[]i64 = null;

        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "kernel_shape")) {
                if (attr.type == AttributeType.INTS) kernel_shape = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "strides")) {
                if (attr.type == AttributeType.INTS) stride = attr.ints;
            }
        }

        if (kernel_shape == null) return error.KernelShapeNotFound;
        if (stride == null) return error.StridesNotFound;

        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        Codegen_log.debug("\n kernel_shape: []i64 = {any}", .{kernel_shape.?});
        Codegen_log.debug("\n stride: []i64 = {any}", .{stride.?});

        const kernel_2d = [2]usize{ @intCast(kernel_shape.?[0]), @intCast(kernel_shape.?[1]) };
        const stride_2d = [2]usize{ @intCast(stride.?[0]), @intCast(stride.?[1]) };

        shape = try utils.usizeSliceToI64Slice(
            @constCast(
                &try tensorMath.get_pooling_output_shape(
                    try utils.i64SliceToUsizeSlice(input_shape),
                    kernel_2d,
                    stride_2d,
                ),
            ),
        );
    }
    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_averagePool_output_shape(readyNode: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__AveragePool.html
    // Computes the output shape for an AveragePool node based on input shape and attributes.
    const input_shape: []const i64 = readyNode.inputs.items[0].?.shape;
    var output_shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        output_shape = tensorShape;
    } else {
        var kernel_shape: ?[]i64 = null;
        var stride: ?[]i64 = null;
        var dilation: ?[]i64 = null;
        var auto_pad: []const u8 = "NOTSET";
        var pads: ?[]i64 = null;
        var ceil_mode: bool = false;
        var count_include_pad: bool = false;

        // Extract attributes from node
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "kernel_shape")) {
                if (attr.type == AttributeType.INTS) kernel_shape = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "strides")) {
                if (attr.type == AttributeType.INTS) stride = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "dilations")) {
                if (attr.type == AttributeType.INTS) dilation = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "auto_pad")) {
                if (attr.type == AttributeType.STRING) auto_pad = attr.s;
            } else if (std.mem.eql(u8, attr.name, "pads")) {
                if (attr.type == AttributeType.INTS) pads = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "ceil_mode")) {
                if (attr.type == AttributeType.INT) ceil_mode = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "count_include_pad")) {
                if (attr.type == AttributeType.INT) count_include_pad = attr.i != 0;
            }
        }

        // Check mandatory attributes
        if (kernel_shape == null) return error.KernelShapeNotFound;
        if (stride == null) return error.StridesNotFound;

        // Create proper allocated slices for default values
        var default_stride: []i64 = undefined;
        var default_dilation: []i64 = undefined;
        var default_pads: []i64 = undefined;
        var should_free_stride = false;
        var should_free_dilation = false;
        var should_free_pads = false;

        if (stride == null) {
            default_stride = try allocator.alloc(i64, 2);
            default_stride[0] = 1;
            default_stride[1] = 1;
            stride = default_stride;
            should_free_stride = true;
        }

        if (dilation == null) {
            default_dilation = try allocator.alloc(i64, 2);
            default_dilation[0] = 1;
            default_dilation[1] = 1;
            dilation = default_dilation;
            should_free_dilation = true;
        }

        if (pads == null) {
            default_pads = try allocator.alloc(i64, 4);
            @memset(default_pads, 0);
            pads = default_pads;
            should_free_pads = true;
        }

        defer {
            if (should_free_stride) allocator.free(default_stride);
            if (should_free_dilation) allocator.free(default_dilation);
            if (should_free_pads) allocator.free(default_pads);
        }

        // Convert AutoPadType from string
        var auto_pad_type: tensorMath.AutoPadType = .NOTSET;
        if (std.mem.eql(u8, auto_pad, "VALID")) {
            auto_pad_type = .VALID;
        } else if (std.mem.eql(u8, auto_pad, "SAME_UPPER")) {
            auto_pad_type = .SAME_UPPER;
        } else if (std.mem.eql(u8, auto_pad, "SAME_LOWER")) {
            auto_pad_type = .SAME_LOWER;
        }

        // Convert parameters to usize
        const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
        defer allocator.free(usize_input_shape);

        const usize_kernel_shape = try utils.i64SliceToUsizeSlice(kernel_shape.?);
        defer allocator.free(usize_kernel_shape);

        const usize_stride = try utils.i64SliceToUsizeSlice(stride.?);
        defer allocator.free(usize_stride);

        const usize_dilation = try utils.i64SliceToUsizeSlice(dilation.?);
        defer allocator.free(usize_dilation);

        const usize_pads = try utils.i64SliceToUsizeSlice(pads.?);
        defer allocator.free(usize_pads);

        // Call the AveragePool shape function
        output_shape = try utils.usizeSliceToI64Slice(@constCast(try tensorMath.get_onnx_averagepool_output_shape(usize_input_shape, usize_kernel_shape, usize_stride, usize_dilation, usize_pads, auto_pad_type, ceil_mode)));
    }
    // Assign the output shape to the node
    readyNode.outputs.items[0].shape = output_shape;
    // Codegen_log.debug("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_reducemean_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_reducemean_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = try utils.i64SliceToUsizeSlice(readyNode.inputs.items[0].?.shape);
        defer allocator.free(input_shape);

        // Get attributes
        var keepdims: bool = true;
        var noop_with_empty_axes: bool = false;

        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "keepdims")) {
                if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
                if (attr.type == AttributeType.INT) noop_with_empty_axes = attr.i != 0;
            }
        }

        // Get axes from second input if it exists
        var axes: ?[]const i64 = null;
        if (readyNode.inputs.items.len > 1 and
            readyNode.inputs.items[1].?.tensorProto != null and
            readyNode.inputs.items[1].?.tensorProto.?.int64_data != null)
        {
            axes = readyNode.inputs.items[1].?.tensorProto.?.int64_data.?;
        }

        Codegen_log.info("\n input_shape: []usize = {any}", .{input_shape});
        Codegen_log.debug("\n axes: ?[]i64 = {any}", .{axes});
        Codegen_log.debug("\n keepdims: {}", .{keepdims});
        Codegen_log.debug("\n noop_with_empty_axes: {}", .{noop_with_empty_axes});

        const output_shape = try tensorMath.get_reduce_mean_output_shape(input_shape, axes, keepdims, noop_with_empty_axes);
        defer allocator.free(output_shape);

        shape = try utils.usizeSliceToI64Slice(output_shape);
        Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
    }
    readyNode.outputs.items[0].shape = shape;
}
inline fn compute_slice_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_slice_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        const starts = readyNode.inputs.items[1].?.tensorProto.?.int64_data.?;
        const ends = readyNode.inputs.items[2].?.tensorProto.?.int64_data.?;

        var axes: ?[]i64 = null;
        var steps: ?[]i64 = null;

        // Get axes if provided (input 3)
        if (readyNode.inputs.items.len > 3) {
            axes = readyNode.inputs.items[3].?.tensorProto.?.int64_data.?;
        }

        // Get steps if provided (input 4)
        if (readyNode.inputs.items.len > 4) {
            steps = readyNode.inputs.items[4].?.tensorProto.?.int64_data.?;
        }

        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        Codegen_log.debug("\n starts: []i64 = {any}", .{starts});
        Codegen_log.debug("\n ends: []i64 = {any}", .{ends});
        Codegen_log.debug("\n axes: []i64 = {any}", .{axes});
        Codegen_log.debug("\n steps: []i64 = {any}", .{steps});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_slice_output_shape(
            try utils.i64SliceToUsizeSlice(input_shape),
            starts,
            ends,
            axes,
            steps,
        ));
    }

    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_shape_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_shape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;

        // Get start and end attributes if they exist
        var start: ?i64 = null;
        var end: ?i64 = null;

        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "start")) {
                if (attr.type == AttributeType.INT) start = attr.i;
            } else if (std.mem.eql(u8, attr.name, "end")) {
                if (attr.type == AttributeType.INT) end = attr.i;
            }
        }
        // Calculate output size
        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_shape_output_shape(try utils.i64SliceToUsizeSlice(input_shape), start, end));
    }

    // Shape operator always outputs a 1D tensor }
    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_gather_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_gather_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const data_shape = readyNode.inputs.items[0].?.shape;
        const indices_shape = readyNode.inputs.items[1].?.shape;

        // Get axis attribute, default is 0
        var axis: i64 = 0;
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == AttributeType.INT) axis = attr.i;
            }
        }

        // Handle negative axis
        if (axis < 0) {
            axis += @as(i64, @intCast(data_shape.len));
        }

        Codegen_log.debug("\n data_shape: []i64 = {any}", .{data_shape});
        Codegen_log.debug("\n indices_shape: []i64 = {any}", .{indices_shape});
        Codegen_log.debug("\n axis: {}", .{axis});

        // Calculate output shape:
        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_gather_output_shape(
            try utils.i64SliceToUsizeSlice(data_shape),
            try utils.i64SliceToUsizeSlice(indices_shape),
            axis,
        ));
    }

    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_sigmoid_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_sigmoid_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_sigmoid_output_shape(try utils.i64SliceToUsizeSlice(input_shape)));
    }
    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_transpose_output_shape(readyNode: *ReadyNode) !void {
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        //get perm
        var perm: ?[]i64 = null;
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "perm")) {
                if (attr.type == AttributeType.INTS) {
                    perm = attr.ints;
                }
            }
        }
        const input_shape = try utils.i64SliceToUsizeSlice(readyNode.inputs.items[0].?.shape);

        if (perm) |p| {
            shape = try utils.usizeSliceToI64Slice(try tensorMath.get_transpose_output_shape(input_shape, try utils.i64SliceToUsizeSlice(p)));
        } else {
            const perm_usize: ?[]const usize = null;
            shape = try utils.usizeSliceToI64Slice(try tensorMath.get_transpose_output_shape(input_shape, perm_usize));
        }
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_unsqueeze_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_unsqueeze_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        // Get axes from attributes or from the second input tensor
        var axes: ?[]const i64 = null;

        // First check if axes is provided as an input tensor (ONNX opset 13+)
        if (readyNode.inputs.items.len > 1 and readyNode.inputs.items[1].?.tensorProto != null) {
            axes = readyNode.inputs.items[1].?.tensorProto.?.int64_data.?;
            Codegen_log.debug("\n axes from input tensor: []i64 = {any}", .{axes.?});
        } else {
            // Otherwise, check for axes attribute (ONNX opset < 13)
            for (readyNode.nodeProto.attribute) |attr| {
                if (std.mem.eql(u8, attr.name, "axes")) {
                    if (attr.type == AttributeType.INTS) {
                        axes = attr.ints;
                        Codegen_log.debug("\n axes from attribute: []i64 = {any}", .{axes.?});
                        break;
                    }
                }
            }
        }

        if (axes == null) return error.UnsqueezeAxesNotFound;

        // Calculate output shape
        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_unsqueeze_output_shape(
            try utils.i64SliceToUsizeSlice(input_shape),
            axes.?,
        ));
    }

    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

pub fn compute_concat_output_shape(readyNode: *ReadyNode) !void {
    // Codegen_log.debug("\n compute_concat_output_shape for node: {s}", .{readyNode.nodeProto.name.?});

    // Get the axis attribute (required)
    var axis: i64 = 0;
    var axis_found = false;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) {
                axis = attr.i;
                axis_found = true;
            } else {
                return error.ConcatAxisNotINT;
            }
        }
    }

    if (!axis_found) {
        return error.ConcatAxisNotFound;
    }

    // Codegen_log.debug("\n   axis: {}", .{axis});
    // Codegen_log.debug("\n   number of inputs: {}", .{readyNode.inputs.items.len});

    // Ensure there's at least one input tensor
    if (readyNode.inputs.items.len == 0) {
        return error.ConcatNoInputs;
    }

    // Print input shapes
    for (readyNode.inputs.items, 0..) |input, i| {
        if (input) |in| Codegen_log.debug("\n   input[{}] shape: []i64 = {any}", .{ i, in.shape }) else Codegen_log.debug("\n   input[{}] is null", .{i});
    }

    // Convert input shapes to usize for get_concatenate_output_shape
    Codegen_log.debug("\n   Converting input shapes to usize...", .{});
    var input_shapes = try allocator.alloc([]const usize, readyNode.inputs.items.len);
    errdefer {
        Codegen_log.warn("\n   Error occurred, cleaning up input_shapes...", .{});
        for (input_shapes) |shape| {
            allocator.free(shape);
        }
        allocator.free(input_shapes);
    }

    for (readyNode.inputs.items, 0..) |input, i| {
        Codegen_log.debug("\n   Converting input[{}] shape to usize...", .{i});
        // Handle negative values by using 1 as a placeholder
        var shape = try allocator.alloc(usize, input.?.shape.len);
        for (input.?.shape, 0..) |dim, j| {
            shape[j] = if (dim < 0) 1 else @intCast(dim);
        }
        input_shapes[i] = shape;
        Codegen_log.debug("\n   Converted shape: []usize = {any}", .{input_shapes[i]});
    }

    // Get output shape using the existing function
    Codegen_log.debug("\n   Calling get_concatenate_output_shape...", .{});
    const output_shape = try tensorMath.get_concatenate_output_shape(input_shapes, axis);
    errdefer {
        Codegen_log.warn("\n   Error occurred, cleaning up output_shape...", .{});
        allocator.free(output_shape);
    }
    Codegen_log.debug("\n   Got output shape: []usize = {any}", .{output_shape});

    // Convert back to i64 for storing in readyNode
    Codegen_log.debug("\n   Converting output shape back to i64...", .{});
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    Codegen_log.debug("\n   Final output shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});

    // Clean up
    Codegen_log.debug("\n   Cleaning up temporary allocations...", .{});
    for (input_shapes) |shape| {
        allocator.free(shape);
    }
    allocator.free(input_shapes);
    allocator.free(output_shape);
    // Codegen_log.debug("\n   Cleanup complete", .{});
}

inline fn compute_sqrt_output_shape(readyNode: *ReadyNode) !void {
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_sqrt_output_shape(try utils.i64SliceToUsizeSlice(input_shape)));
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_tanh_output_shape(readyNode: *ReadyNode) !void {
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_tanh_output_shape(try utils.i64SliceToUsizeSlice(input_shape)));
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_gelu_output_shape(readyNode: *ReadyNode) !void {
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_gelu_output_shape(try utils.i64SliceToUsizeSlice(input_shape)));
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_floor_output_shape(readyNode: *ReadyNode) !void {
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_floor_output_shape(try utils.i64SliceToUsizeSlice(input_shape)));
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_elu_output_shape(readyNode: *ReadyNode) !void {
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        shape = try utils.usizeSliceToI64Slice(try tensorMath.get_elu_output_shape(try utils.i64SliceToUsizeSlice(input_shape)));
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_ceil_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_ceil_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        const output_shape = try tensorMath.get_ceil_output_shape(try utils.i64SliceToUsizeSlice(input_shape));
        shape = try utils.usizeSliceToI64Slice(output_shape);
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_clip_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_ceil_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input = readyNode.inputs.items[0] orelse {
        return error.InputTensorIsNull;
    };

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape = input.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

        const output_shape = try tensorMath.get_ceil_output_shape(try utils.i64SliceToUsizeSlice(input_shape));
        shape = try utils.usizeSliceToI64Slice(output_shape);
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_identity_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_identity_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
        return;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        const output_shape = try tensorMath.get_identity_output_shape(try utils.i64SliceToUsizeSlice(input_shape));
        // Identity operation preserves the input shape
        shape = try utils.usizeSliceToI64Slice(output_shape);
        Codegen_log.info("\n output_shape: []i64 = {any}", .{shape});
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_leaky_relu_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_leaky_relu_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
        return;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        const output_shape = try tensorMath.get_leaky_relu_output_shape(try utils.i64SliceToUsizeSlice(input_shape));
        // LeakyReLU is an element-wise operation, output shape is identical to input shape
        shape = try utils.usizeSliceToI64Slice(output_shape);
        Codegen_log.info("\n output_shape: []i64 = {any}", .{shape});
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_longsoftmax_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_longsoftmax_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
        return;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        const output_shape = try tensorMath.get_longsoftmax_output_shape(try utils.i64SliceToUsizeSlice(input_shape));
        // LongSoftmax is an element-wise operation, output shape is identical to input shape
        shape = try utils.usizeSliceToI64Slice(output_shape);
        Codegen_log.info("\n output_shape: []i64 = {any}", .{shape});
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_matmul_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_matmul_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        const input_shape_a = readyNode.inputs.items[0].?.shape;
        const input_shape_b = readyNode.inputs.items[1].?.shape;
        Codegen_log.info("\n input_shape_a: []i64 = {any}", .{input_shape_a});
        Codegen_log.info("\n input_shape_b: []i64 = {any}", .{input_shape_b});

        const output_shape = try tensorMath.get_mat_mul_output_shape(try utils.i64SliceToUsizeSlice(input_shape_a), try utils.i64SliceToUsizeSlice(input_shape_b));
        // MatMul is an element-wise operation, output shape is identical to input shape
        shape = try utils.usizeSliceToI64Slice(output_shape);
        Codegen_log.info("\n output_shape: []i64 = {any}", .{shape});
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_split_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_split_output_shape node: {s}======", .{readyNode.nodeProto.name.?});

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        readyNode.outputs.items[0].shape = tensorShape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;

        // Get axis attribute (default is 0)
        var axis: i64 = 0;
        var split_sizes: ?[]i64 = null;

        // Extract attributes
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == AttributeType.INT) axis = attr.i;
            } else if (std.mem.eql(u8, attr.name, "split")) {
                if (attr.type == AttributeType.INTS) split_sizes = attr.ints;
            }
        }

        // Check if split_sizes is provided as an input (ONNX opset 13+)
        if (readyNode.inputs.items.len > 1 and
            readyNode.inputs.items[1].?.tensorProto != null and
            readyNode.inputs.items[1].?.tensorProto.?.int64_data != null)
        {
            split_sizes = readyNode.inputs.items[1].?.tensorProto.?.int64_data.?;
        }

        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        Codegen_log.debug("\n axis: {}", .{axis});
        Codegen_log.debug("\n split_sizes: {any}", .{split_sizes});
        Codegen_log.debug("\n num_outputs: {}", .{readyNode.outputs.items.len});

        // Convert i64 split_sizes to usize if provided
        var usize_split_sizes: ?[]usize = null;
        defer if (usize_split_sizes != null) allocator.free(usize_split_sizes.?);

        if (split_sizes) |sizes| {
            usize_split_sizes = try allocator.alloc(usize, sizes.len);
            for (sizes, 0..) |size, i| {
                usize_split_sizes.?[i] = @intCast(size);
            }
        }

        // Convert input_shape to usize
        const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
        defer allocator.free(usize_input_shape);

        // Get output shapes using the utility function
        const output_shapes = try tensorMath.get_split_output_shapes(usize_input_shape, axis, usize_split_sizes, readyNode.outputs.items.len // Pass the number of outputs
        );
        defer {
            for (output_shapes) |shape| {
                allocator.free(shape);
            }
            allocator.free(output_shapes);
        }

        // Ensure we have enough output tensors
        if (readyNode.outputs.items.len != output_shapes.len) {
            return error.MismatchedOutputCount;
        }

        // Set the output shapes
        for (output_shapes, 0..) |shape, i| {
            readyNode.outputs.items[i].shape = try utils.usizeSliceToI64Slice(shape);
            Codegen_log.info("\n output[{}] shape: []i64 = {any}", .{ i, readyNode.outputs.items[i].shape });
        }
    }
}

pub fn compute_resize_output_shape(readyNode: *ReadyNode) !void {
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |Shape| {
        shape = Shape;
    } else {
        const input_shape = readyNode.inputs.items[0].?.shape;
        var scales: ?[]const f32 = null;
        var sizes: ?[]const i64 = null;

        const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
        defer allocator.free(usize_input_shape);

        if (readyNode.inputs.items.len > 2 and readyNode.inputs.items[2].?.tensorProto != null) {
            if (readyNode.inputs.items[2].?.tensorProto.?.float_data != null) {
                scales = readyNode.inputs.items[2].?.tensorProto.?.float_data.?;
            }
        }

        if (readyNode.inputs.items.len > 3 and readyNode.inputs.items[3].?.tensorProto != null) {
            if (readyNode.inputs.items[3].?.tensorProto.?.int64_data != null) {
                sizes = readyNode.inputs.items[3].?.tensorProto.?.int64_data.?;
            }
        }

        const usize_sizes = try utils.i64SliceToUsizeSlice(sizes.?);
        defer allocator.free(usize_sizes);

        const output_shape = try tensorMath.get_resize_output_shape(usize_input_shape, scales, usize_sizes);

        shape = try utils.usizeSliceToI64Slice(output_shape);
    }
    readyNode.outputs.items[0].shape = shape;
}

pub fn compute_resize_output_shape_generic(comptime T: type, input_shape: []const T, scales: ?[]const f32, sizes: ?[]const T) ![]T {
    // Make sure we support all parameter types
    if (scales != null) {
        // Calculate output shape based on scales
        var output_shape = try allocator.alloc(T, input_shape.len);

        for (0..input_shape.len) |i| {
            output_shape[i] = @intFromFloat(@as(f32, @floatFromInt(input_shape[i])) * scales.?[i]);
        }

        return output_shape;
    }

    if (sizes != null) {
        // Use sizes directly
        return sizes.?;
    }

    // If neither scales nor sizes is provided, return the input shape
    return input_shape;
}

inline fn compute_neg_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_neg_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        Codegen_log.info("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].?.shape});
        const input_shape = readyNode.inputs.items[0].?.shape;

        const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
        defer allocator.free(usize_input_shape);

        const output_shape = try tensorMath.get_neg_output_shape(usize_input_shape);
        defer allocator.free(output_shape);

        shape = try utils.usizeSliceToI64Slice(output_shape);
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_Div_output_shape(readyNode: *ReadyNode) !void {
    // Codegen_log.info("\n====== compute_Div_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    // Codegen_log.info("\n input[0] shape: []i64 = {any}", .{readyNode.inputs.items[0].?.shape});
    // Codegen_log.info("\n input[1] shape: []i64 = {any}", .{readyNode.inputs.items[1].?.shape});

    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        Codegen_log.debug("\n Using shape from tensor: []i64 = {any}", .{tensorShape});
        // Use the shape with more dimensions between tensor shape and input shapes
        const max_dim = @max(tensorShape.len, readyNode.inputs.items[0].?.shape.len, readyNode.inputs.items[1].?.shape.len);
        if (tensorShape.len < max_dim) {
            Codegen_log.debug("\n Tensor shape has fewer dimensions, using shape with {} dimensions", .{max_dim});
            // For element-wise operations, use input[0] shape and add first dimension from input[1]
            if (readyNode.inputs.items[1].?.shape.len > readyNode.inputs.items[0].?.shape.len) {
                var new_shape = try allocator.alloc(i64, readyNode.inputs.items[1].?.shape.len);
                new_shape[0] = readyNode.inputs.items[1].?.shape[0]; // Use first dimension from input[1]
                for (readyNode.inputs.items[0].?.shape, 0..) |dim, i| {
                    new_shape[i + 1] = dim; // Copy remaining dimensions from input[0]
                }
                shape = new_shape;
            } else {
                shape = readyNode.inputs.items[0].?.shape;
            }
        } else {
            shape = tensorShape;
        }
    } else {
        Codegen_log.debug("\n Using shape with more dimensions", .{});
        // For element-wise operations, use input[0] shape and add first dimension from input[1]
        if (readyNode.inputs.items[1].?.shape.len > readyNode.inputs.items[0].?.shape.len) {
            var new_shape = try allocator.alloc(i64, readyNode.inputs.items[1].?.shape.len);
            new_shape[0] = readyNode.inputs.items[1].?.shape[0]; // Use first dimension from input[1]
            for (readyNode.inputs.items[0].?.shape, 0..) |dim, i| {
                new_shape[i + 1] = dim; // Copy remaining dimensions from input[0]
            }
            shape = new_shape;
        } else {
            shape = readyNode.inputs.items[0].?.shape;
        }
    }
    readyNode.outputs.items[0].shape = shape;
    // Codegen_log.info("\n Final output shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_pads_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_pads_output_shape node: {s}=====", .{readyNode.nodeProto.name.?});

    // Input 0: data
    if (readyNode.inputs.items[0] == null) return error.InputTensorNotFound;
    const data_shape_i64 = readyNode.inputs.items[0].?.shape;
    Codegen_log.debug("\n data_shape: {any}", .{data_shape_i64});

    // Input 1: pads (required, must be int64)
    if (readyNode.inputs.items.len < 2 or readyNode.inputs.items[1] == null or readyNode.inputs.items[1].?.tensorProto == null or readyNode.inputs.items[1].?.tensorProto.?.int64_data == null) {
        Codegen_log.warn("\nERROR: Pads input (index 1) is missing or not a constant int64 tensor.", .{});
        return error.PadsInputInvalid;
    }
    const pads_values_i64 = readyNode.inputs.items[1].?.tensorProto.?.int64_data.?;
    Codegen_log.debug("\n pads_values: {any}", .{pads_values_i64});

    // Input 2: constant_value (optional, shape not needed for output shape calculation)

    // Input 3: axes (optional, must be int64 or int32)
    var axes_values_isize: ?[]const isize = null;
    var axes_buffer: []isize = undefined; // Buffer for conversion
    defer if (axes_values_isize != null) allocator.free(axes_buffer);

    if (readyNode.inputs.items.len > 3 and readyNode.inputs.items[3] != null and readyNode.inputs.items[3].?.tensorProto != null) {
        const axes_proto = readyNode.inputs.items[3].?.tensorProto.?;
        if (axes_proto.int64_data != null) {
            const axes_i64 = axes_proto.int64_data.?;
            axes_buffer = try allocator.alloc(isize, axes_i64.len);
            for (axes_i64, 0..) |val, i| {
                axes_buffer[i] = @intCast(val);
            }
            axes_values_isize = axes_buffer;
            Codegen_log.debug("\n axes (from i64): {any}", .{axes_values_isize});
        } else if (axes_proto.int32_data != null) {
            const axes_i32 = axes_proto.int32_data.?;
            axes_buffer = try allocator.alloc(isize, axes_i32.len);
            for (axes_i32, 0..) |val, i| {
                axes_buffer[i] = @intCast(val);
            }
            axes_values_isize = axes_buffer;
            Codegen_log.debug("\n axes (from i32): {any}", .{axes_values_isize});
        } else {
            Codegen_log.warn("\nWARNING: Axes input (index 3) provided but is not int64 or int32 data.", .{});
            // Proceed without axes if the type is wrong
        }
    } else {
        Codegen_log.debug("\n axes: not provided", .{});
    }

    // Convert data shape to usize
    const data_shape_usize = try utils.i64SliceToUsizeSlice(data_shape_i64);
    defer allocator.free(data_shape_usize);

    // Call the shape calculation function
    const output_shape_usize = try tensorMath.get_pads_output_shape(allocator, data_shape_usize, pads_values_i64, axes_values_isize);
    defer allocator.free(output_shape_usize);

    // Convert result back to i64 for storing in readyNode
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape_usize);
    Codegen_log.info("\n final output_shape: {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_mean_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_mean_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;
    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        if (readyNode.inputs.items.len == 0) {
            return error.EmptyInputList;
        }

        var input_shapes = try allocator.alloc([]usize, readyNode.inputs.items.len);
        defer allocator.free(input_shapes);
        for (readyNode.inputs.items, 0..) |input, i| {
            Codegen_log.info("\n input_{}_shape: []i64 = {any}", .{ i, input.?.shape });
            input_shapes[i] = try utils.i64SliceToUsizeSlice(input.?.shape);
        }

        const output_shape_usize = try tensorMath.get_mean_output_shape(input_shapes);
        shape = try utils.usizeSliceToI64Slice(@constCast(output_shape_usize));
    }
    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_flatten_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_flatten_output_shape node: {s}======", .{readyNode.nodeProto.name orelse "(unnamed)"});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        if (readyNode.inputs.items.len == 0) {
            return error.EmptyInputList;
        }
        const input_shape_i64 = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape_i64});

        var axis: i64 = 1; // Default ONNX
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != AttributeType.INT) {
                    Codegen_log.warn("\n ERROR: Flatten 'axis' attribute has unexpected type {}", .{attr.type});
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
                break;
            }
        }
        Codegen_log.debug("\n axis: {}", .{axis});

        const input_shape_usize = try utils.i64SliceToUsizeSlice(input_shape_i64);
        defer allocator.free(input_shape_usize);
        Codegen_log.debug("\n input_shape_usize: []usize = {any}", .{input_shape_usize});

        const output_shape_usize = try tensorMath.get_flatten_output_shape(input_shape_usize, @intCast(axis));
        //defer allocator.free(output_shape_usize); // Libera il risultato di get_flatten_output_shape
        Codegen_log.debug("\n output_shape_usize: []usize = {any}", .{output_shape_usize});

        shape = try utils.usizeSliceToI64Slice(@constCast(output_shape_usize));
    }

    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_squeeze_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_squeeze_output_shape node: {s}======", .{readyNode.nodeProto.name orelse "(unnamed)"});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        if (readyNode.inputs.items.len == 0) {
            return error.EmptyInputList;
        }

        const input_shape_i64 = readyNode.inputs.items[0].?.shape;
        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape_i64});

        const input_shape_usize = try utils.i64SliceToUsizeSlice(input_shape_i64);
        defer allocator.free(input_shape_usize);
        Codegen_log.debug("\n input_shape_usize: []usize = {any}", .{input_shape_usize});

        var axes_values_isize: ?[]const isize = null;
        var axes_buffer: []isize = undefined; // Buffer for conversion
        defer if (axes_values_isize != null) allocator.free(axes_buffer);

        if (readyNode.inputs.items.len > 2 and readyNode.inputs.items[2] != null and readyNode.inputs.items[2].?.tensorProto != null) {
            const axes_proto = readyNode.inputs.items[2].?.tensorProto.?;
            if (axes_proto.int64_data != null) {
                const axes_i64 = axes_proto.int64_data.?;
                axes_buffer = try allocator.alloc(isize, axes_i64.len);
                for (axes_i64, 0..) |val, i| {
                    axes_buffer[i] = @intCast(val);
                }
                axes_values_isize = axes_buffer;
                Codegen_log.debug("\n axes (from i64): {any}", .{axes_values_isize});
            } else if (axes_proto.int32_data != null) {
                const axes_i32 = axes_proto.int32_data.?;
                axes_buffer = try allocator.alloc(isize, axes_i32.len);
                for (axes_i32, 0..) |val, i| {
                    axes_buffer[i] = @intCast(val);
                }
                axes_values_isize = axes_buffer;
                Codegen_log.debug("\n axes (from i32): {any}", .{axes_values_isize});
            } else {
                Codegen_log.warn("\nWARNING: Axes input (index 3) provided but is not int64 or int32 data.", .{});
                // Proceed without axes if the type is wrong
            }
        } else {
            Codegen_log.debug("\n axes: not provided", .{});
        }

        const output_shape_usize = try tensorMath.get_squeeze_output_shape(input_shape_usize, axes_values_isize);
        Codegen_log.debug("\n output_shape_usize: []usize = {any}", .{output_shape_usize});

        shape = try utils.usizeSliceToI64Slice(@constCast(output_shape_usize));
    }

    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_dynamicQuantizeLinear_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_dynamicQuantizeLinear_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].?.shape;
    Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});

    // Ensure the correct number of outputs
    if (readyNode.outputs.items.len != 3) {
        Codegen_log.debug("ERROR: DynamicQuantizeLinear expects 3 outputs, but got {}.", .{readyNode.outputs.items.len});
        return error.MismatchedOutputCount;
    }

    // Convert input shape to usize
    const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
    defer allocator.free(usize_input_shape);

    // Get output shapes using the utility function
    const output_shapes = try tensorMath.get_dynamicQuantizeLinear_output_shape(usize_input_shape);
    defer {
        for (output_shapes) |shape| {
            allocator.free(shape);
        }
        allocator.free(output_shapes);
    }

    // Assign shapes to output tensors
    // Output 0: y (quantized data) - shape is same as input
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shapes[0]);
    Codegen_log.debug("\n output[0] (y) shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});

    // Output 1: y_scale (scalar) - shape is {1}
    readyNode.outputs.items[1].shape = try utils.usizeSliceToI64Slice(output_shapes[1]);
    Codegen_log.debug("\n output[1] (y_scale) shape: []i64 = {any}", .{readyNode.outputs.items[1].shape});

    // Output 2: y_zero_point (scalar) - shape is {1}
    readyNode.outputs.items[2].shape = try utils.usizeSliceToI64Slice(output_shapes[2]);
    Codegen_log.debug("\n output[2] (y_zero_point) shape: []i64 = {any}", .{readyNode.outputs.items[2].shape});
}

inline fn compute_convInteger_output_shape(readyNode: *ReadyNode) !void {
    Codegen_log.info("\n====== compute_convInteger_output_shape node: {s}=====", .{readyNode.nodeProto.name.?});
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        // ConvInteger shape calculation is the same as Conv
        const input_shape: []const i64 = readyNode.inputs.items[0].?.shape;
        const kernel_shape: []const i64 = readyNode.inputs.items[1].?.shape;

        // Extract attributes similar to compute_conv_output_shape
        var stride: ?[]i64 = null;
        var dilation: ?[]i64 = null;
        var auto_pad: []const u8 = "NOTSET";
        var pads: ?[]i64 = null;
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "strides")) {
                if (attr.type == AttributeType.INTS) stride = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "dilations")) {
                if (attr.type == AttributeType.INTS) dilation = attr.ints;
            } else if (std.mem.eql(u8, attr.name, "auto_pad")) {
                if (attr.type == AttributeType.STRING) auto_pad = attr.s;
            } else if (std.mem.eql(u8, attr.name, "pads")) {
                if (attr.type == AttributeType.INTS) pads = attr.ints;
            }
        }

        // Defaults if not found (as per ONNX spec)
        const default_stride = [_]i64{ 1, 1 }; // Assuming 2D for now
        const default_dilation = [_]i64{ 1, 1 };

        const stride_ref = stride orelse &default_stride;
        const dilation_ref = dilation orelse &default_dilation;

        Codegen_log.info("\n input_shape: []i64 = {any}", .{input_shape});
        Codegen_log.debug("\n kernel_shape: []i64 = {any}", .{kernel_shape});
        Codegen_log.debug("\n stride: []i64 = {any}", .{stride_ref});
        Codegen_log.debug("\n dilation: []i64 = {any}", .{dilation_ref});
        Codegen_log.debug("\n pads: ?[]i64 = {any}", .{pads});
        Codegen_log.debug("\n auto_pad: {s}", .{auto_pad});

        // Convert shapes and attributes to usize slices for the math function
        const input_shape_usize = try utils.i64SliceToUsizeSlice(input_shape);
        defer allocator.free(input_shape_usize);
        const kernel_shape_usize = try utils.i64SliceToUsizeSlice(kernel_shape);
        defer allocator.free(kernel_shape_usize);
        const stride_usize = try utils.i64SliceToUsizeSlice(stride_ref);
        defer allocator.free(stride_usize);
        const dilation_usize = try utils.i64SliceToUsizeSlice(dilation_ref);
        defer allocator.free(dilation_usize);

        var pads_usize: ?[]usize = null;
        var pads_alloc: []usize = undefined; // Keep track of allocation
        if (pads) |p| {
            pads_alloc = try utils.i64SliceToUsizeSlice(p);
            pads_usize = pads_alloc;
        }
        defer if (pads_usize != null) allocator.free(pads_alloc);

        // Call the existing convolution shape calculation function
        const output_shape_usize_array = try tensorMath.get_convolution_output_shape(
            input_shape_usize,
            kernel_shape_usize,
            stride_usize,
            pads_usize,
            dilation_usize,
            auto_pad,
        );

        // Convert the [4]usize array back to []const i64 slice
        // Pass a slice directly from the const array. usizeSliceToI64Slice takes []const usize.
        shape = try utils.usizeSliceToI64Slice(@constCast(&output_shape_usize_array));
    }
    readyNode.outputs.items[0].shape = shape;
    Codegen_log.info("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}
