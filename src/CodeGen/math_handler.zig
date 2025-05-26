const std = @import("std");
const os = std.os;

const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const allocator = zant.utils.allocator.allocator;

const mathHandler_log = std.log.scoped(.mathHandler);

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
const parameters = codegen.parameters;
const codegen_options = @import("codegen_options");
const globals = @import("globals.zig");

// ----------------------------------- MATH -----------------------------------

/// This method map and write the ONNX operations with the Zant LeanTensorMath mathods
/// Follow the link for details: https://onnx.ai/onnx/operators/?utm_source=chatgpt.com
pub fn write_math_op(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // Dynamic allocation of intermediate output tensors if requested
    if (codegen_options.dynamic) {
        for (node.outputs.items) |output| {
            const san_name = try utils.getSanitizedName(output.name);
            const type_str = try utils.getTypeString(output.dtype);
            const dims = output.shape;
            // Emit shape constant for this output
            _ = try writer.print("    var shape_{s}  = [_]usize{{", .{san_name});
            for (dims, 0..) |dim, i| {
                if (i != 0) _ = try writer.print(", ", .{});
                _ = try writer.print("{d}", .{dim});
            }
            _ = try writer.print("}};", .{});
            // Allocate tensor on heap
            _ = try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_{s}) catch return;", .{ san_name, type_str, san_name });
            // Defer deinitialization ONLY if it's not the final network output
            if (!std.mem.eql(u8, output.name, globals.networkOutput.name)) {
                _ = try writer.print("    defer tensor_{s}.deinit();\n", .{san_name});
            }
        }
    }
    if (codegen_options.comm) {
        try write_op_info(writer, node);
    }
    if (codegen_options.log) {
        try writer.print(
            \\ 
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running {s} operation...\n")));
            \\    }}
        , .{node.*.nodeProto.*.op_type});
    }

    if (std.mem.eql(u8, node.nodeProto.op_type, "Add")) {
        try write_add(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "AveragePool")) {
        try write_averagePool(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "BatchNormalization")) {
        try write_BatchNormalization(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Ceil")) {
        try write_ceil(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Clip")) {
        try write_clip(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Concat")) {
        try write_concat(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Constant")) {
        try write_constant(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Conv")) {
        try write_conv(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "ConvInteger")) {
        try write_convInteger(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Div")) {
        try write_div(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "DynamicQuantizeLinear")) {
        try write_dynamicQuantizeLinear(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Elu")) {
        try write_elu(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Flatten")) {
        try write_flatten(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Floor")) {
        try write_floor(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Squeeze")) {
        try write_squeeze(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gather")) {
        try write_gather(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gemm")) {
        try write_gemm(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gelu")) {
        try write_gelu(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Identity")) {
        try write_identity(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LeakyRelu")) {
        try write_leaky_relu(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LogSoftmax")) {
        try writer.writeAll("// Handle LogSoftmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MatMul")) {
        try write_matmul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try write_maxPool(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try write_mul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Neg")) {
        try write_neg(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try write_oneHot(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Pad")) {
        try write_pads(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "ReduceMean")) {
        try write_reduceMean(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try write_ReLU(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try write_reshape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try write_resize(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sigmoid")) {
        try write_sigmoid(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Softmax")) {
        try write_softmax(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Slice")) {
        try write_slice(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Split")) {
        try write_split(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sqrt")) {
        try write_sqrt(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sub")) {
        try write_sub(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sum")) {
        try write_sum(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Tanh")) {
        try write_tanh(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Transpose")) {
        try write_transpose(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Shape")) {
        try write_shape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Unsqueeze")) {
        try write_unsqueeze(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mean")) {
        try write_mean(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Cast")) {
        try write_cast(writer, node);
    } else {
        // Stub for unsupported operations: generate unreachable at runtime
        _ = try writer.print(
            \\
            \\    // Operation {s} not supported, inserting stub
            \\    unreachable("Unsupported op: {s}");
        , .{ node.nodeProto.op_type, node.nodeProto.op_type });
        return;
    }

    try writer.writeAll(" catch return;");
}

fn write_op_info(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    try writer.print(
        \\
        \\
        \\   //forwarding operation : {s}
        \\   //parameters:
        \\   //   inputs: 
    , .{node.*.nodeProto.*.op_type});

    //write the inputs
    for (node.inputs.items) |input| {
        try writer.print(
            \\
            \\   //      -> {s} 
        , .{input.name});
    }
    try writer.print(
        \\
        \\   //    outputs: 
    , .{});

    //write the outputs
    for (node.outputs.items) |output| {
        try writer.print(
            \\
            \\   //      <- {s} 
        , .{output.name});
    }
}

inline fn write_add(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Add.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensors_lean(T, T, {s}, {s}, &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_BatchNormalization(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1
    //      - scale (heterogeneous) - T1: Scale tensor of shape ©.
    //      - B (heterogeneous) - T1: Bias tensor of shape ©.
    //      - input_mean (heterogeneous) - T2: running (training) or estimated (testing) mean tensor of shape ©.
    //      - input_var (heterogeneous) - T2: running (training) or estimated (testing) variance tensor of shape ©.
    // OUTPUT:
    //      - Y (heterogeneous) - T: The output tensor of the same shape as X
    // ATTRIBUTES:
    //      - epsilon - FLOAT (default is '1e-05'): The epsilon value to use to avoid division by zero.
    //      - momentum - FLOAT (default is '0.9'): Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum).
    //      - training_mode - INT (default is '0'): If set to true, it indicates BatchNormalization is being used for training, and outputs 1 and 2 are to be computed.

    var epsilon: f32 = 1e-05;
    var momentum: f32 = 0.9;
    // var training_mode: bool = false; -> NOT USED, ALWAYS FALSE for Zant

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "epsilon")) |_| {
            if (attr.type == AttributeType.FLOAT) epsilon = attr.f else return error.BatchNorm_epsilon_NotFloat;
        } else if (std.mem.indexOf(u8, attr.name, "momentum")) |_| {
            if (attr.type == AttributeType.FLOAT) momentum = attr.f else return error.BatchNorm_momentum_NotFloat;
        } else if (std.mem.indexOf(u8, attr.name, "training_mode")) |_| {
            if (attr.type == AttributeType.INT) if (attr.i != 0) return error.BatchNorm_training_NotAvailable;
        }
    }

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_scale_string
    var tensor_scale_string: []u8 = undefined;
    defer allocator.free(tensor_scale_string);

    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);

    if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[2].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[2].?.name), ")" });
    }

    //----create tensor_input_mean_string
    var tensor_input_mean_string: []u8 = undefined;
    defer allocator.free(tensor_input_mean_string);

    if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_input_mean_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[3].?.name),
            ")",
        });
    } else {
        tensor_input_mean_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[3].?.name), ")" });
    }

    //----create tensor_input_var_string
    var tensor_input_var_string: []u8 = undefined;
    defer allocator.free(tensor_input_var_string);

    if (node.inputs.items[4].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_input_var_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[4].?.name),
            ")",
        });
    } else {
        tensor_input_var_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[4].?.name), ")" });
    }

    // pub inline fn batchNormalization_lean( comptime T: anytype, comptime T1: anytype, comptime T2: anytype, input: *Tensor(T), scales: *Tensor(T1), B: *Tensor(T1), input_mean: Tensor(T2), input_var: Tensor(T2), epsilon: f32, momentum: f32, training_mode: bool, output: *Tensor(T))
    _ = try writer.print(
        \\    
        \\
        \\    tensMath.batchNormalization_lean(
        \\        {s}, //type 0
        \\        {s}, //type 1
        \\        {s}, //type 2
        \\        {s}, //input
        \\        {s}, //scales
        \\        {s}, //B
        \\        {s}, //input_mean
        \\        {s}, //input_var
        \\        {}, //epsilon
        \\        {}, //momentum
        \\        false, //training_mode
        \\        &tensor_{s}, //output
        \\    )
    , .{
        try getSafeTensorTypeString(node.inputs.items[0].?, node.nodeProto.name orelse "UnnamedBatchNormInput0"), // MODIFIED: Use helper for input X type
        try getSafeTensorTypeString(node.inputs.items[1].?, node.nodeProto.name orelse "UnnamedBatchNormInput1"), // MODIFIED: Use helper for input scale type
        try getSafeTensorTypeString(node.inputs.items[3].?, node.nodeProto.name orelse "UnnamedBatchNormInput3"), // MODIFIED: Use helper for input mean/var type (check ONNX spec for correct index if this is not mean's type)
        tensor_X_string,
        tensor_scale_string,
        tensor_B_string,
        tensor_input_mean_string,
        tensor_input_var_string,
        epsilon,
        momentum,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_oneHot(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__OneHot.html
    // INPUTS:
    //      - indices (heterogeneous) - T1: Tensor of indices.
    //      - depth (heterogeneous) - T2: Scalar tensor for depth.
    //      - values (heterogeneous) - T3: Tensor of shape [off_value, on_value].
    // OUTPUT:
    //      - output (heterogeneous) - T3: Output tensor with one-hot encoding.
    // ATTRIBUTES:
    //      - axis - INT (default is -1): Axis along which to add the one-hot dimension.

    var axis: i64 = -1; // Default axis per ONNX
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type != AttributeType.INT) return error.InvalidAxisType;
            axis = attr.i;
        }
    }

    //----create indices string
    var indices_string: []u8 = undefined;
    defer allocator.free(indices_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    }

    //----create depth string
    var depth_string: []u8 = undefined;
    defer allocator.free(depth_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        depth_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        depth_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    }

    //----create values string
    var values_string: []u8 = undefined;
    defer allocator.free(values_string);
    if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
        values_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[2].?.name),
            ")",
        });
    } else {
        values_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(node.inputs.items[2].?.name),
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
        try utils.getTypeString(globals.tensorHashMap.getPtr(node.inputs.items[2].?.name).?.tensorProto.?.data_type), // T
        indices_string, // indices
        depth_string, // depth
        values_string, // values
        axis, // axis
        try utils.getSanitizedName(node.outputs.items[0].name), // output
    });
}

inline fn write_sub(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Sub.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\    tensMath.sub_tensors_lean(T, T, {s}, ({s}), &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_conv(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Conv.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    //      - W (heterogeneous) - T: The weight tensor
    //      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
    //      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
    //      - group - INT (default is '1'): number of groups input channels and output channels are divided into
    //      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";
    var dilations: ?[]i64 = null;
    var group: i64 = 1;
    var kernel_shape: ?[]i64 = null;
    var pads: ?[]i64 = null;
    var strides: ?[]i64 = null; //mandatory

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
            if (attr.type == AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
        }
    }

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_W_string
    var tensor_W_string: []u8 = undefined;
    defer allocator.free(tensor_W_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    //----create ?bias string
    var bias_string: []u8 = undefined;
    // Bias Tensor B is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const B_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
    } else {
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
    }

    //----create stride string (mandatory)
    // TODO: implement default stride, see docs above
    if (strides == null) return error.StrideNotFound;
    const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(strides.?);

    //----create ?pads string
    var pads_string: []const u8 = "null";
    if (pads != null) {
        if (pads.?.len > 0) { // Check if the slice is actually non-empty
            pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
            // Assuming no allocation needed to be freed, following write_conv
        } else {
            pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
        }
    } // else pads_string remains "null"

    //----create ?dilatations string
    var dilat_string: []const u8 = "null";
    if (dilations != null) {
        if (dilations.?.len > 0) {
            dilat_string = try utils.i64SliceToUsizeArrayString(dilations.?);
        } else {
            dilat_string = "&[_]usize{}";
        }
    } // else dilat_string remains "null"

    // pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void
    _ = try writer.print(
        \\    
        \\
        \\    tensMath.conv_lean(
        \\        T, //type
        \\        {s}, //input
        \\        {s}, //kernel
        \\        &tensor_{s}, //output
        \\        {s}, //bias
        \\        {s}, //stride
        \\        {s}, //pads
        \\        {s}, //dilatations
        \\        {}, //group
        \\        "{s}", //auto_pad
        \\    )
    , .{
        tensor_X_string, //Input
        tensor_W_string, //Kernel
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        bias_string, //Bias
        stride_string, //Strides
        pads_string, //Pads
        dilat_string, //Dilatations
        group, //Group
        auto_pad, //auto_pad
    });
}

inline fn write_concat(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Concat.html
    // INPUTS:
    //      - inputs (variadic, heterogeneous) - T: List of tensors for concatenation
    // OUTPUTS:
    //      - concat_result (heterogeneous) - T: Concatenated tensor
    // ATTRIBUTES:
    //      - axis (int, required): Which axis to concat on

    // Get the axis attribute
    var axis: i64 = 0;
    var axis_found = false;

    for (node.nodeProto.attribute) |attr| {
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

    // Special case for axis 0 with different ranks
    if (axis == 0) {
        // Find if there are tensors with different ranks
        var has_different_ranks = false;
        const first_rank = node.inputs.items[0].?.shape.len;

        for (node.inputs.items[1..]) |input| {
            if (input.?.shape.len != first_rank) {
                has_different_ranks = true;
                break;
            }
        }

        if (has_different_ranks) {
            _ = try writer.print(
                \\
                \\    // Special case for concatenation along axis 0 with different ranks
                \\    // This requires custom handling as the standard concatenate function expects same rank
                \\    mathHandler_log.warn("\\nWarning: Concatenating tensors with different ranks along axis 0\\n", .{{}});
                \\
                \\    // Create a list of tensors to concatenate
                \\    var concat_tensor_list_{s} = [_]Tensor(T){{
            , .{try utils.getSanitizedName(node.outputs.items[0].name)});

            for (node.inputs.items, 0..) |input, idx| {
                if (idx > 0) {
                    _ = try writer.print(", ", .{});
                }

                var tensor_string: []u8 = undefined;
                defer allocator.free(tensor_string);
                if (input.?.tag == globals.TensorTag.INITIALIZER) {
                    tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                        "@constCast(&param_lib.tensor_",
                        try utils.getSanitizedName(input.?.name),
                        ")",
                    });
                } else {
                    tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(input.?.name) });
                }
                _ = try writer.print("{s}", .{tensor_string});
            }

            _ = try writer.print(
                \\}};
                \\
                \\    // Perform concatenation with special handling for different ranks
                \\     try tensMath.concatenate_lean(T, &allocator, &concat_tensor_list_{s}, {},tensor_{s})
            , .{
                try utils.getSanitizedName(node.outputs.items[0].name),
                axis,
                try utils.getSanitizedName(node.outputs.items[0].name),
            });

            return;
        }
    }

    // Standard case: all tensors have the same rank
    // Create a tensor list with all input tensors
    _ = try writer.print(
        \\
        \\    // Create a list of tensors to concatenate
        \\    var concat_tensor_list_{s} = [_]Tensor(T){{
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});

    for (node.inputs.items, 0..) |input, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }

        if (input.?.tag == globals.TensorTag.INITIALIZER) {
            _ = try writer.print("param_lib.tensor_{s}", .{try utils.getSanitizedName(input.?.name)});
        } else {
            _ = try writer.print("tensor_{s}", .{try utils.getSanitizedName(input.?.name)});
        }
    }

    _ = try writer.print(
        \\}};
        \\
        \\    // Perform concatenation
        \\    tensMath.concatenate_lean(T, &allocator, &concat_tensor_list_{s}, {}, &tensor_{s} )
    , .{
        try utils.getSanitizedName(node.outputs.items[0].name),
        axis,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_constant(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Constant.html
    // Outputs:
    // - output (heterogeneous) - T: Output tensor containing the same value of the provided tensor.
    // Attributes - only one of these should be specified:
    // - value (TENSOR): The value for the elements of the output tensor.
    // - sparse_value (SPARSE_TENSOR): The value for the elements of the output tensor in sparse format.
    // - value_float (FLOAT): The value for the sole element for the scalar, float32, output tensor.
    // - value_floats (FLOATS): The values for the elements for the 1D, float32, output tensor.
    // - value_int (INT): The value for the sole element for the scalar, int64, output tensor.
    // - value_ints (INTS): The values for the elements for the 1D, int64, output tensor.
    // - value_string (STRING): The value for the sole element for the scalar, UTF-8 string, output tensor.
    // - value_strings (STRINGS): The values for the elements for the 1D, UTF-8 string, output tensor.

    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    for (node.nodeProto.attribute) |attr| {
        if (attr.type == onnx.AttributeType.TENSOR) {
            try writer.print(
                \\
                \\    // Constant tensor_{s} already declared and inizialized in predict.zig write_constantTensor()
            , .{output_name});

            return;
        } else if (std.mem.eql(u8, attr.name, "value_float")) {
            if (attr.type != AttributeType.FLOAT) return error.ConstantAttributeTypeMismatch;

            // Create a scalar tensor with a float value
            try writer.print(
                \\
                \\    // Initialize scalar float constant
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, {d}) catch return;
            , .{ output_name, attr.f });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_floats")) {
            if (attr.type != AttributeType.FLOATS) return error.ConstantAttributeTypeMismatch;

            // Create 1D tensor with float values
            try writer.print(
                \\
                \\    // Initialize 1D float array constant
                \\    const data_{s} = [_]T{{
            , .{output_name});

            // Write array elements
            for (attr.floats, 0..) |val, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{val});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
            , .{ output_name, output_name, attr.floats.len });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_int")) {
            if (attr.type != AttributeType.INT) return error.ConstantAttributeTypeMismatch;

            // Create a scalar tensor with an int value
            try writer.print(
                \\
                \\    // Initialize scalar int constant
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, @as(T, @floatFromInt({d}))) catch return;
            , .{ output_name, attr.i });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_ints")) {
            if (attr.type != AttributeType.INTS) return error.ConstantAttributeTypeMismatch;

            // Create 1D tensor with int values
            try writer.print(
                \\
                \\    // Initialize 1D int array constant
                \\    const data_{s} = [_]T{{
            , .{output_name});

            // Write array elements
            for (attr.ints, 0..) |val, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("@as(T, @floatFromInt({d}))", .{val});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
            , .{ output_name, output_name, attr.ints.len });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_string")) {
            if (attr.type != AttributeType.STRING) return error.ConstantAttributeTypeMismatch;

            // String constants are not directly supported in this numeric tensor library
            try writer.print(
                \\
                \\    // String constants are not directly supported in this numeric tensor library
                \\    // For now, we'll create a placeholder tensor with a single value
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, 0) catch return;
                \\    // The actual string value was: "{s}"
            , .{ output_name, attr.s });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_strings")) {
            if (attr.type != AttributeType.STRINGS) return error.ConstantAttributeTypeMismatch;

            // String array constants are not directly supported in this numeric tensor library
            try writer.print(
                \\
                \\    // String array constants are not directly supported in this numeric tensor library
                \\    // For now, we'll create a placeholder tensor with zeros
                \\    const data_{s} = [_]T{{
            , .{output_name});

            // Create a placeholder array of zeros with the same length
            for (attr.strings, 0..) |_, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("0", .{});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
                \\    // Note: This is a placeholder for string values that cannot be directly represented
            , .{ output_name, output_name, attr.strings.len });
            return;
        } else if (std.mem.eql(u8, attr.name, "sparse_value")) {
            // Sparse tensor constants require special handling
            try writer.print(
                \\
                \\    // Sparse tensor constants are not yet fully supported
                \\    // Creating a placeholder tensor for sparse_value
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, 0) catch return;
                \\    mathHandler_log.warn("Warning: sparse_value attribute used but not fully supported\\n", .{{}});
            , .{output_name});
            return;
        }
    }

    // If we get here, no valid constant value was found
    try writer.writeAll(
        \\
        \\    return error.ConstantValueNotFound;
    );
}

inline fn write_div(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Div.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",

            try utils.getSanitizedName(node.inputs.items[0].?.name),

            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",

            try utils.getSanitizedName(node.inputs.items[1].?.name),

            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\    tensMath.div_lean(T, {s}, ({s}), &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_gather(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gather.html
    // INPUTS:
    //      - data (heterogeneous) - T: Tensor of rank r >= 1.
    //      - indices (heterogeneous) - tensor(int64): Tensor of int64 indices, of any rank q.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Tensor of rank q + r - 1.
    // ATTRIBUTES:
    //      - axis (int, default is 0): Which axis to gather on. Negative value means counting dimensions from the back.

    var axis: i64 = 0;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        }
    }

    // Create data tensor string
    var data_tensor_string: []u8 = undefined;
    defer allocator.free(data_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        data_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        data_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // Create indices tensor string
    const indices_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
    var indices_tensor_string: []u8 = undefined;
    defer allocator.free(indices_tensor_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        indices_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "param_lib.tensor_",
            indices_name,
        });
    } else {
        indices_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "tensor_",
            indices_name,
        });
    }

    _ = try writer.print(
        \\    
        \\
        \\    //creating the indices Tensor(usize)
        \\    
        \\    const usize_slice_{s} = utils.sliceToUsizeSlice({s}.data);
        \\    var usize_tensor_{s} = Tensor(usize).fromConstBuffer(&allocator, usize_slice_{s}, {s}.shape);
        \\    defer allocator.free(usize_slice_{s});
        \\    
    , .{
        indices_name, //usize_slice_
        indices_tensor_string, //tensor_
        indices_name, //usize_tensor_
        indices_name, //usize_slice_
        indices_tensor_string, //tensor_.shape
        indices_name, //usize_slice_ for free
    });

    _ = try writer.print(
        \\
        \\
        \\    tensMath.gather_lean(
        \\        T, //type
        \\        {s}, //data tensor
        \\        &usize_tensor_{s}, //indices tensor
        \\        {}, //axis
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        data_tensor_string, // Input data tensor
        indices_name, // Input indices tensor
        axis, // Selected axis
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_gemm(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gemm.html
    // INPUTS:
    //      - Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
    //      - Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
    //      - Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
    //OUTPUTS:
    //      - Output tensor of shape (M, N).
    // ATTRIBUTES:
    //      - alpha. FLOAT (default is '1.0'): Scalar multiplier for the product of input tensors A * B.
    //      - beta - FLOAT (default is '1.0'): Scalar multiplier for input tensor C.
    //      - transA - INT (default is '0'): Whether A should be transposed
    //      - transB - INT (default is '0'): Whether B should be transposed

    var alpha: f32 = 1.0;
    var beta: f32 = 1.0;
    var transA: bool = false;
    var transB: bool = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
            if (attr.type == AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
            if (attr.type == AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
            if (attr.type == AttributeType.INT) transA = if (attr.i != 0) true else false else return error.GemmTransANotINT;
        } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
            if (attr.type == AttributeType.INT) transB = if (attr.i != 0) true else false else return error.GemmTransBNotINT;
        }
    }

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
    }

    // Input Tensor C is optional! verify the presence
    var tensor_C_string: []u8 = undefined;
    if (node.inputs.items.len == 3) {
        const sanitized_tensor_C = try utils.getSanitizedName(node.inputs.items[2].?.name);
        tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&",
            if (globals.tensorHashMap.getPtr(node.inputs.items[2].?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_tensor_C,
            ")",
        });
    } else {
        tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.gemm_lean(T, {s}, {s}, {s}, {}, {}, {s}, {s}, &tensor_{s} )
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        tensor_C_string,
        alpha,
        beta,
        if (transA) "true" else "false",
        if (transB) "true" else "false",
        try utils.getSanitizedName(node.outputs.items[0].name), // Output
    });
}

inline fn write_matmul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__MatMul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
    }

    // Calculate b_width_bytes safely, handling potential null tensorProto
    // Get type information for tensor B to estimate element size
    const input_B_name = node.inputs.items[1].?.name;
    const ready_tensor_B = globals.tensorHashMap.getPtr(input_B_name) orelse {
        mathHandler_log.warn("Error: Tensor '{s}' not found in globals.tensorHashMap for MatMul.\n", .{input_B_name});
        return error.TensorNotFound;
    };

    var element_size_bytes: usize = 4; // Default to f32 size as fallback
    if (ready_tensor_B.tensorProto) |tp| {
        const data_type = tp.data_type;
        // Determine size from DataType enum
        element_size_bytes = switch (data_type) {
            .FLOAT => @sizeOf(f32),
            .FLOAT16 => @sizeOf(f16),
            .INT64 => @sizeOf(i64),
            .INT32 => @sizeOf(i32),
            .INT8 => @sizeOf(i8),
            .UINT8 => @sizeOf(u8),
            // Add other supported types as needed
            else => blk: {
                mathHandler_log.warn("Warning: Unsupported DataType '{any}' for MatMul input B '{s}'. Assuming f32 size.\n", .{ data_type, input_B_name });
                break :blk 4;
            },
        };
    } else {
        // Fallback if tensorProto is null - log a warning
        mathHandler_log.warn("Warning: TensorProto for MatMul input B '{s}' is null. Assuming f32 size for width calculation.\n", .{input_B_name});
    }

    const b_dims = node.inputs.items[1].?.shape.len;
    if (b_dims == 0) {
        mathHandler_log.warn("Error: MatMul input B '{s}' has zero dimensions.\n", .{input_B_name});
        return error.InvalidShape; // Avoid panic on empty shape
    }

    const b_width_elements: usize = @intCast(node.inputs.items[1].?.shape[b_dims - 1]);
    const b_width_bytes: usize = b_width_elements * element_size_bytes;

    if (b_width_bytes >= std.atomic.cache_line) { //B is large enough for the new mat mul to work;
        _ = try writer.print(
            \\
            \\    tensMath.blocked_mat_mul_lean(T, {s}, {s}, &tensor_{s})
        , .{
            tensor_A_string, // Input tensor A
            tensor_B_string, // Input tensor B
            try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
        });
    } else { //B is not large enough, so we keep the old but improved mat_mul
        _ = try writer.print(
            \\
            \\    tensMath.mat_mul_lean(T, {s}, {s}, &tensor_{s})
        , .{
            tensor_A_string, // Input tensor A
            tensor_B_string, // Input tensor B
            try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
        });
    }
}

inline fn write_maxPool(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //https://onnx.ai/onnx/operators/onnx__MaxPool.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output data tensor from average or max pooling across the input tensor.
    //      - (NOT IMPLEMENTED) Indices (optional, heterogeneous) - I: Indices tensor from max pooling across the input tensor.
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
    //      - ceil_mode - INT (default is '0'): Whether to use ceil or floor (default) to compute the output shape
    //      - dilations - INTS : Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis
    //      - kernel_shape - INTS (required) : The size of the kernel along each axis.
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - storage_order - INT (default is '0'): The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";

    var ceil_mode: i64 = 0;

    var dilations: ?[]i64 = null;

    var kernel_shape: ?[]i64 = null; //mandatory

    var pads: ?[]i64 = null;

    var storage_order: i64 = 0;

    var strides: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.MaxPoolAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
            if (attr.type == AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.MaxPoolDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.MaxPoolKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.MaxPoolPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "storage_order")) |_| {
            if (attr.type == AttributeType.INT) storage_order = attr.i else return error.MaxPoolStorage_orderNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.MaxPoolStridesNotINTS;
        }
    }

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create kernel_shape string
    var kernel_shape_string: []const u8 = undefined;
    if (kernel_shape != null) {
        kernel_shape_string = try utils.i64SliceToUsizeArrayString(kernel_shape.?);
    } else {
        return error.Kernel_shapeNotFound;
    }

    //----create strides string
    var strides_string: []const u8 = undefined;
    if (strides != null) {
        strides_string = try utils.i64SliceToUsizeArrayString(strides.?);
    } else {
        return error.StridesNotFound;
    }

    //----create dilations string
    var dilations_string: []const u8 = undefined;
    if (dilations != null) {
        dilations_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilations_string = try utils.i64SliceToUsizeArrayString(&[_]i64{ 1, 1, 1, 1 }); // TODO: It is hardcoded in 4D, not the most elegant solution
    }

    //----create pads string
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        return error.PadsNotFound;
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.onnx_maxpool_lean(
        \\        T,
        \\        {s}, //Input
        \\        &tensor_{s}, //Output
        \\        {s}, //kernel_shape
        \\        {s}, //strides
        \\        {s}, //dilations
        \\        {s}, //pads
        \\        tensMath.AutoPadType.{s}, //auto_pad
        \\    )
    , .{
        tensor_X_string, //Input
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        kernel_shape_string, //kernel_shape
        strides_string, //strides
        dilations_string, //dilatations
        pads_string, //pads
        auto_pad, //auto_pad
    });
}

inline fn write_averagePool(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__AveragePool.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output data tensor from average pooling
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): NOTSET, SAME_UPPER, SAME_LOWER, VALID
    //      - ceil_mode - INT (default is '0'): Whether to use ceil or floor
    //      - count_include_pad - INT (default is '0'): Whether to include padding in averaging
    //      - dilations - INTS: Dilation value along each spatial axis (default 1)
    //      - kernel_shape - INTS (required): Kernel size along each axis
    //      - pads - INTS: Padding for each spatial axis
    //      - strides - INTS: Stride along each spatial axis (default 1)

    mathHandler_log.debug("DEBUG: write_averagePool called for node: {s}\n", .{node.nodeProto.name orelse "unnamed"});

    var auto_pad: []const u8 = "NOTSET";
    var ceil_mode: i64 = 0;
    var count_include_pad: i64 = 0;
    var dilations: ?[]i64 = null;
    var kernel_shape: ?[]i64 = null; // Obbligatorio
    var pads: ?[]i64 = null;
    var strides: ?[]i64 = null;

    // Leggi gli attributi
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.AveragePoolAutoPadNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
            if (attr.type == AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "count_include_pad")) |_| {
            if (attr.type == AttributeType.INT) count_include_pad = attr.i else return error.AveragePoolCountIncludePadNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.AveragePoolDilationsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.AveragePoolKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.AveragePoolPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.AveragePoolStridesNotINTS;
        }
    }

    // Crea tensor_X_string per l'input
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "&tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
        });
    }

    // Crea stringa per kernel_shape
    var kernel_shape_string: []const u8 = undefined;
    if (kernel_shape != null) {
        kernel_shape_string = try utils.i64SliceToUsizeArrayString(kernel_shape.?);
    } else {
        return error.Kernel_shapeNotFound;
    }

    // Crea stringa per strides
    var strides_string: []const u8 = undefined;
    if (strides != null) {
        strides_string = try utils.i64SliceToUsizeArrayString(strides.?);
    } else {
        return error.StridesNotFound;
    }

    // Crea stringa per dilations
    var dilations_string: []const u8 = undefined;
    if (dilations != null) {
        dilations_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilations_string = try utils.i64SliceToUsizeArrayString(&[_]i64{ 1, 1, 1, 1 }); // TODO: Hardcoded in 4D, not the most elegant solution
    }

    // Crea stringa per pads
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        return error.PadsNotFound;
    }

    // Scrivi la chiamata a onnx_averagepool_lean
    _ = try writer.print(
        \\
        \\
        \\    tensMath.onnx_averagepool_lean(
        \\        T,
        \\        {s}, // Input
        \\        &tensor_{s}, // Output
        \\        {s}, // kernel_shape
        \\        {s}, // strides
        \\        {s}, // dilations
        \\        {s}, // pads
        \\        tensMath.AutoPadType.{s}, // auto_pad
        \\        {s}, // count_include_pad
        \\    )
    , .{
        tensor_X_string, // Input
        try utils.getSanitizedName(node.outputs.items[0].name), // Output
        kernel_shape_string, // kernel_shape
        strides_string, // strides
        dilations_string, // dilations
        pads_string, // pads
        auto_pad, // auto_pad
        if (count_include_pad == 1) "true" else "false", // count_include_pad
    });
}

inline fn write_mul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Mul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.mul_lean(T, {s}, ({s}), &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_reduceMean(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__ReduceMean.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    //      - axes (optional, heterogeneous) - tensor(int64): A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'keepdims' is true.
    // OUTPUTS:
    //      - reduced (heterogeneous) - T: Reduced output tensor.
    // ATTRIBUTES:
    //      - keepdims (int, default is 1): Keep the reduced dimension or not, default 1 means keep the reduced dimension.
    //      - noop_with_empty_axes (int, default is 0): Defines behavior if 'axes' is empty. Default behavior is to reduce all axes.

    // Get attributes
    var keepdims: bool = true;
    var noop_with_empty_axes: bool = false;
    var axes_attr: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
            if (attr.type == AttributeType.INT) noop_with_empty_axes = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "axes")) {
            if (attr.type == AttributeType.INTS) axes_attr = attr.ints;
        }
    }

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // Handle axes - either from attribute, input tensor, or as null
    var axes_str: []const u8 = "null";
    var needs_free = false;

    // First check if axes is defined as an attribute
    if (axes_attr != null) {
        // Create a static array from the axes attribute
        const axes_array_name = try std.fmt.allocPrint(allocator, "axes_{s}", .{try utils.getSanitizedName(node.outputs.items[0].name)});
        defer allocator.free(axes_array_name);

        try writer.print(
            \\
            \\    // Define axes array from attribute
            \\    const {s} = [_]i64{{
        , .{axes_array_name});

        for (axes_attr.?, 0..) |axis, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{axis});
        }

        try writer.print(
            \\}};
            \\
        , .{});

        axes_str = try std.fmt.allocPrint(allocator, "&{s}", .{axes_array_name});
        needs_free = true;
    }
    // If not found in attributes, check if provided as an input tensor
    else if (node.inputs.items.len > 1) {
        // Get axes from second input
        const axes_name = try utils.getSanitizedName(node.inputs.items[1].?.name);

        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
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
        try utils.getSanitizedName(node.outputs.items[0].name),
        axes_str,
        if (keepdims) "true" else "false",
        if (noop_with_empty_axes) "true" else "false",
    });
}

inline fn write_ReLU(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0].? -> input
    //node.outputs.items[0] -> output

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.ReLU_lean(T, {s}, &tensor_{s})
    , .{
        tensor_A_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_elu(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Elu.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor
    // ATTRIBUTES:
    //      - alpha - FLOAT (default is '1.0'): Coefficient of ELU operator

    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    }

    const tensor_type = try utils.getTypeString(globals.tensorHashMap.getPtr(node.inputs.items[0].?.name).?.tensorProto.?.data_type);

    // alpha attribute
    var alpha: f32 = 1.0;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "alpha")) {
            if (attr.type != AttributeType.FLOAT) {
                return error.InvalidAttributeType;
            }
            alpha = attr.f;
        }
    }

    _ = try writer.print(
        \\
        \\    tensMath.elu_lean(
        \\        {s}, // type
        \\        {s}, // input
        \\        &tensor_{s}, // output
        \\        {d} // alpha
        \\    )
    , .{
        tensor_type,
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
        alpha,
    });
}

inline fn write_flatten(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Flatten.html
    // INPUTS:
    //      - data (heterogeneous) - T: Input tensor of any shape.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Output tensor with shape [outer_dim, inner_dim].
    // ATTRIBUTES:
    //      - axis - INT (default is '1'): Indicate up to which input dimension should be flattened.

    //----create tensor_input_string
    var tensor_input_string: []u8 = undefined;
    defer allocator.free(tensor_input_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    }

    const tensor_type = try utils.getTypeString(globals.tensorHashMap.getPtr(node.inputs.items[0].?.name).?.tensorProto.?.data_type);

    _ = try writer.print(
        \\
        \\    tensMath.flatten_lean(
        \\        {s}, // type
        \\        {s}, // input
        \\        &tensor_{s}, // output
        \\    )
    , .{
        tensor_type,
        tensor_input_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_squeeze(writer: std.fs.File.Writer, node: *ReadyNode) !void {

    // Squeeze - 23 : https://onnx.ai/onnx/operators/onnx__Squeeze.html
    // Inputs:
    //  - data (heterogeneous) - T: Tensors with at least max(dims) dimensions
    //  - axes (optional, heterogeneous) - tensor(int64): List of integers indicating the dimensions to squeeze
    //      Negative value means counting dimensions from the back
    //      Accepted range is [-r, r-1] where r = rank(data)
    // Outputs:
    //  - squeezed (heterogeneous) - T: Reshaped tensor with same data as input

    var tensor_input_string: []u8 = undefined;
    defer allocator.free(tensor_input_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    }

    var tensor_type: []const u8 = undefined;
    const input_ready_tensor = globals.tensorHashMap.getPtr(node.inputs.items[0].?.name);

    if (input_ready_tensor) |rt| {
        if (rt.tensorProto) |tp| {
            tensor_type = try utils.getTypeString(tp.data_type);
        } else {
            // Fallback if tensorProto is null
            mathHandler_log.warn("Warning: tensorProto is null for Squeeze input '{s}'. Falling back to f32 type.\n", .{node.inputs.items[0].?.name});
            tensor_type = "f32";
        }
    } else {
        // Fallback if ReadyTensor is not found in the map
        mathHandler_log.warn("Warning: ReadyTensor not found for Squeeze input '{s}'. Falling back to f32 type.\n", .{node.inputs.items[0].?.name});
        tensor_type = "f32";
    }

    _ = try writer.print(
        \\
        \\    tensMath.squeeze_lean(
        \\        {s}, // type
        \\        {s}, // input
        \\        &tensor_{s}, // output
        \\    )
    , .{
        tensor_type,
        tensor_input_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_reshape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    // Inputs:
    //      - data (T): An input tensor.
    //      - shape (tensor(int64)): Specifies the output shape.
    // Attributes:
    //      - allowzero (int, default 0): DEPRECATED. If true (non-zero), the output shape can contain 0.
    //      - shape (ints): Alternative way to provide shape (used if input 'shape' is not provided).
    // REMOVED: const T = node.outputs.items[0].tensorProto.?.dataType; // T is not needed by reshape_lean

    // Find allowzero attribute (deprecated but might exist)
    var allowzer0: bool = false;
    var shape_attribute: ?[]const i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "allowzero")) {
            if (attr.type == AttributeType.INT) allowzer0 = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "shape")) {
            if (attr.type == AttributeType.INTS) shape_attribute = attr.ints;
        }
    }

    // Input tensor string creation
    const sanitized_input_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    const input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[0].?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_input_name,
    });
    defer allocator.free(input_string);

    // Shape slice generation logic
    var shape_slice_code = std.ArrayList(u8).init(allocator);
    defer shape_slice_code.deinit();
    const output_sanitized_name = try utils.getSanitizedName(node.outputs.items[0].name);
    var shape_from_attr = false; // Track source of shape

    if (shape_attribute) |attr_shape| {
        shape_from_attr = true;
        // Shape from attribute
        // Generate code like: const shape_slice_<output_name> = [_]isize{ val1, val2, ... };
        try shape_slice_code.writer().print("const shape_slice_{s} = [_]isize{{", .{output_sanitized_name});
        for (attr_shape, 0..) |val, i| {
            try shape_slice_code.writer().print("{s}{}", .{ if (i > 0) ", " else "", val });
        }
        try shape_slice_code.writer().print("}};", .{});
    } else {
        // Shape from input tensor
        if (node.inputs.items.len < 2) {
            mathHandler_log.warn("ERROR: Reshape node '{s}' requires a 'shape' attribute or a second input tensor, but neither was found during code generation.", .{node.nodeProto.name orelse "-"});
            return error.ShapeNotFound;
        }
        const shape_input_tensor = node.inputs.items[1].?;
        const sanitized_shape_name = try utils.getSanitizedName(shape_input_tensor.name);
        const shape_tensor_name = try std.mem.concat(allocator, u8, &[_][]const u8{
            if (globals.tensorHashMap.getPtr(shape_input_tensor.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_shape_name,
        });
        defer allocator.free(shape_tensor_name);

        // Generate code to convert tensor data to isize slice
        try shape_slice_code.writer().print(
            \\    // Convert shape tensor data to isize slice
            \\    // Pass the local allocator to the utils function
            \\    const shape_slice_{s} = utils.sliceToIsizeSlice(allocator, {s}.data); // Removed catch return
            \\    defer allocator.free(shape_slice_{s}); // Free the runtime allocated slice
        , .{
            output_sanitized_name, // Use output name for uniqueness
            shape_tensor_name,
            output_sanitized_name,
        });
    }

    const input_ready_tensor = globals.tensorHashMap.getPtr(node.inputs.items[0].?.name) orelse return error.TensorNotFound;
    const input_type_string = try utils.getTypeString(input_ready_tensor.dtype);

    // Pre-build complex arguments for the format string
    const shape_slice_var_name = try std.fmt.allocPrint(allocator, "shape_slice_{s}", .{output_sanitized_name});
    defer allocator.free(shape_slice_var_name);
    const shape_slice_arg = try std.fmt.allocPrint(allocator, "{s}{s}", .{ if (shape_from_attr) "&" else "", shape_slice_var_name });
    defer allocator.free(shape_slice_arg);

    const output_tensor_arg = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{output_sanitized_name});
    defer allocator.free(output_tensor_arg);

    // Generate the final call using pre-built arguments
    _ = try writer.print(
        \\
        \\
        \\    // Reshape Operation for {s}
        \\    {s} // Generated shape slice code
        \\
        \\    tensMath.reshape_lean(
        \\        {s}, // Use actual input tensor type
        \\        @constCast(&{s}),
        \\        {s}, // Pre-built shape slice argument
        \\        {s}, // Format boolean correctly
        \\        {s}  // Pre-built output tensor argument
        \\    )
    , .{
        node.nodeProto.name orelse "-", // Arg 1 for op name
        shape_slice_code.items, // Arg 2 for shape code
        input_type_string, // Arg 3 for input type
        input_string, // Arg 4 for input tensor
        shape_slice_arg, // Arg 5 for shape slice
        if (allowzer0) "true" else "false", // Arg 6 for allowzero
        output_tensor_arg, // Arg 7 for output tensor
    });
}

inline fn write_sigmoid(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0].? -> input
    //node.outputs.items[0] -> output

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.sigmoid_lean(T, {s}, &tensor_{s})
    , .{
        tensor_A_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

const Converter = zant.utils.type_converter;

/// Helper function to cast tensor data to i64 array
fn castTensorDataToI64Array(tensor_string: []const u8) ![]const u8 {
    // Create a temporary array and initialize with the tensor data
    return try std.fmt.allocPrint(allocator,
        \\blk: {{
        \\    const data_slice = {s};
        \\    // Define the result array in one go, directly applying the conversion
        \\    var temp_i64_arr =  allocator.alloc(i64, data_slice.len) catch return;
        \\    for (data_slice, 0..) |val, i| {{
        \\        temp_i64_arr[i] = if (@typeInfo(@TypeOf(val)) == .int) val else @intFromFloat(val);
        \\    }}
        \\    // The result array will be managed by the caller
        \\    break :blk temp_i64_arr;
        \\}}
    , .{tensor_string});
}

inline fn write_slice(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Slice.html
    // INPUTS:
    //      - input (heterogeneous) - T: Tensor of data to extract slices from.
    //      - starts (heterogeneous) - T1: 1-D tensor of starting indices of corresponding axis in `axes`.
    //      - ends (heterogeneous) - T1: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`.
    //      - axes (heterogeneous) - T1: 1-D tensor of axes that `starts` and `ends` apply to.
    //      - steps (heterogeneous) - T1: 1-D tensor of slice step of corresponding axis in `axes`.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Sliced data tensor.

    // First, get the sanitized names for all tensors
    const input_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    const starts_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
    const ends_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", input_name, ")" });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", input_name, ")" });
    }

    // Create starts tensor string
    var starts_tensor_string: []u8 = undefined;
    defer allocator.free(starts_tensor_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        starts_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "param_lib.tensor_", starts_name, ".data" });
    } else {
        starts_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", starts_name, ".data" });
    }

    // Create cast code for starts
    const starts_i64_code = try castTensorDataToI64Array(starts_tensor_string);
    defer allocator.free(starts_i64_code);

    // Create ends tensor string
    var ends_tensor_string: []u8 = undefined;
    defer allocator.free(ends_tensor_string);
    if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
        ends_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "param_lib.tensor_", ends_name, ".data" });
    } else {
        ends_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", ends_name, ".data" });
    }

    // Create cast code for ends
    const ends_i64_code = try castTensorDataToI64Array(ends_tensor_string);
    defer allocator.free(ends_i64_code);

    // Handle optional axes and steps inputs
    var axes_str: []const u8 = "null";
    var axes_i64_code: []const u8 = "null";
    var steps_str: []const u8 = "null";
    var steps_i64_code: []const u8 = "null";

    if (node.inputs.items.len > 3) {
        const axes_name = try utils.getSanitizedName(node.inputs.items[3].?.name);
        if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
            axes_str = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data", .{axes_name});
        } else {
            axes_str = try std.fmt.allocPrint(allocator, "tensor_{s}.data", .{axes_name});
        }
        axes_i64_code = try castTensorDataToI64Array(axes_str);
        defer if (axes_str.len > 4) allocator.free(axes_str);
    }

    if (node.inputs.items.len > 4) {
        const steps_name = try utils.getSanitizedName(node.inputs.items[4].?.name);
        if (node.inputs.items[4].?.tag == globals.TensorTag.INITIALIZER) {
            steps_str = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data", .{steps_name});
        } else {
            steps_str = try std.fmt.allocPrint(allocator, "tensor_{s}.data", .{steps_name});
        }
        steps_i64_code = try castTensorDataToI64Array(steps_str);
        defer if (steps_str.len > 4) allocator.free(steps_str);
    }

    // Generate defer code for axes and steps
    var axes_defer_code: []const u8 = "";
    var steps_defer_code: []const u8 = "";
    var axes_var_code: []const u8 = "null";
    var steps_var_code: []const u8 = "null";
    var axes_decl_code: []const u8 = "";
    var steps_decl_code: []const u8 = "";

    if (axes_str.len > 4) {
        axes_defer_code = try std.fmt.allocPrint(allocator, "defer allocator.free(axes_arr_{s});", .{output_name});
        axes_var_code = try std.fmt.allocPrint(allocator, "axes_arr_{s}", .{output_name});
        axes_decl_code = try std.fmt.allocPrint(allocator, "const axes_arr_{s} = {s};", .{ output_name, axes_i64_code });
    }

    if (steps_str.len > 4) {
        steps_defer_code = try std.fmt.allocPrint(allocator, "defer allocator.free(steps_arr_{s});", .{output_name});
        steps_var_code = try std.fmt.allocPrint(allocator, "steps_arr_{s}", .{output_name});
        steps_decl_code = try std.fmt.allocPrint(allocator, "const steps_arr_{s} = {s};", .{ output_name, steps_i64_code });
    }

    defer {
        if (axes_defer_code.len > 0) allocator.free(axes_defer_code);
        if (steps_defer_code.len > 0) allocator.free(steps_defer_code);
        if (axes_var_code.len > 4) allocator.free(axes_var_code);
        if (steps_var_code.len > 4) allocator.free(steps_var_code);
        if (axes_decl_code.len > 0) allocator.free(axes_decl_code);
        if (steps_decl_code.len > 0) allocator.free(steps_decl_code);
    }

    _ = try writer.print(
        \\
        \\
        \\    // Allocate arrays for slice operation
        \\    const starts_arr_{s} = {s};
        \\    const ends_arr_{s} = {s};
        \\    {s}
        \\    {s}
        \\    defer allocator.free(starts_arr_{s});
        \\    defer allocator.free(ends_arr_{s});
        \\    {s}
        \\    {s}
        \\
        \\    tensMath.slice_onnx_lean(
        \\        T, //type
        \\        {s}, //input tensor
        \\        starts_arr_{s}, //starts (casted to i64)
        \\        ends_arr_{s}, //ends (casted to i64)
        \\        {s}, //axes (casted to i64 if not null)
        \\        {s}, //steps (casted to i64 if not null)
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        // Variable names with unique suffixes
        output_name,
        starts_i64_code,
        output_name,
        ends_i64_code,
        // Only declare axes and steps if they're used
        if (axes_decl_code.len > 0) axes_decl_code else "// no axes needed",
        if (steps_decl_code.len > 0) steps_decl_code else "// no steps needed",
        output_name,
        output_name,
        if (axes_defer_code.len > 0) axes_defer_code else "// no axes to free",
        if (steps_defer_code.len > 0) steps_defer_code else "// no steps to free",
        input_tensor_string,
        output_name,
        output_name,
        axes_var_code,
        steps_var_code,
        output_name,
    });

    // Free any allocated memory for the i64 code strings
    if (axes_i64_code.len > 4 and axes_str.len > 4) allocator.free(axes_i64_code);
    if (steps_i64_code.len > 4 and steps_str.len > 4) allocator.free(steps_i64_code);
}

inline fn write_softmax(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0].? -> input
    //node.outputs.items[0] -> output

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.softmax_lean(T, {s}, &tensor_{s})
    , .{
        tensor_A_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_sum(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Sum.html
    // INPUTS:
    //      - list of tensors
    // OUTPUTS:
    //      - sum (heterogeneous) - T: Output tensor.

    //Writing the tensor list with all the inputs
    _ = try writer.print(
        \\
        \\
        \\    const my_tensor_list = [_]*const Tensor(T){{
    , .{});

    for (node.inputs.items, 0..) |tens, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }

        var new_tensor_string: []u8 = undefined;
        const sanitized_tensor_name = try utils.getSanitizedName(tens.?.name);

        new_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            if (globals.tensorHashMap.getPtr(tens.?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_tensor_name,
        });

        _ = try writer.print(
            \\{s}
        , .{try utils.getSanitizedName(new_tensor_string)});
    }

    _ = try writer.print("}}", .{});

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensor_list_lean(T, T, &my_tensor_list, &tensor_{s})
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});
}

inline fn write_shape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Shape.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    // OUTPUTS:
    //      - shape (heterogeneous) - T1: Shape of the input tensor
    // ATTRIBUTES:
    //      - start - INT: First dimension to take
    //      - end - INT: Last dimension to take

    var start: ?i64 = null;
    var end: ?i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "start")) {
            if (attr.type == AttributeType.INT) start = attr.i;
        } else if (std.mem.eql(u8, attr.name, "end")) {
            if (attr.type == AttributeType.INT) end = attr.i;
        }
    }

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&param_lib.tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.shape_onnx_lean(
        \\        T,
        \\        T, //type
        \\        @constCast({s}), //input tensor
        \\        {s}, //start
        \\        {s}, //end
        \\        &tensor_{s}, //output tensor,
        \\    )
    , .{
        tensor_A_string,
        if (start) |s| try std.fmt.allocPrint(allocator, "{}", .{s}) else "null",
        if (end) |e| try std.fmt.allocPrint(allocator, "{}", .{e}) else "null",
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_unsqueeze(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    // INPUTS:
    //      - data (heterogeneous) - T: Original tensor
    //      - axes (optional) - tensor(int64): List of integers indicating the dimensions to be inserted.
    //        Negative value means counting dimensions from the back.
    // OUTPUTS:
    //      - expanded (heterogeneous) - T: Reshaped tensor with same data as input.
    // ATTRIBUTES (deprecated in opset 13):
    //      - axes - INTS: List of integers indicating the dimensions to be inserted.

    if (node.inputs.items[0]) |input_tensor| {
        const input_name = try utils.getSanitizedName(input_tensor.name);
        const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

        // Create input tensor string
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);
        if (input_tensor.tag == globals.TensorTag.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", input_name, ")" });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", input_name, ")" });
        }

        // Determine if axes is provided as an input tensor or as an attribute
        var axes_str: []const u8 = "null";
        var needs_free = false;

        if (node.inputs.items.len > 1) {
            // Axes is provided as an input tensor (opset 13+)
            const axes_tensor_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
            if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
                axes_str = try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{axes_tensor_name});
            } else {
                axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{axes_tensor_name});
            }
            needs_free = true;
        } else {
            // Axes is provided as an attribute (opset < 13)
            for (node.nodeProto.attribute) |attr| {
                if (std.mem.eql(u8, attr.name, "axes")) {
                    if (attr.type == AttributeType.INTS) {
                        axes_str = try utils.i64ToI64ArrayString(attr.ints);
                        needs_free = true;
                        break;
                    }
                }
            }
        }

        defer if (needs_free) allocator.free(axes_str);

        // Generate code for the unsqueeze operation
        try writer.print(
            \\     
            \\    tensMath.unsqueeze_lean(
            \\        T, //type
            \\        {s}, //input tensor
            \\        {s}, //axes tensor
            \\        &tensor_{s}, //output tensor
            \\    )
        , .{
            input_tensor_string, //input tensor
            axes_str, //axes tensor
            output_name, //output tensor
        });
    } else {
        return error.InvalidInput;
    }
}

inline fn write_transpose(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Transpose.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    // OUTPUTS:
    //      - transposed (heterogeneous) - T: Transposed output.
    // ATTRIBUTES:
    //      - perm - INTS: A list of integers. By default, reverse the dimensions,
    //        otherwise permute the axes according to the values given.

    // Get the perm attribute if it exists
    var perm_str: []const u8 = "null";
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "perm")) {
            if (attr.type == AttributeType.INTS) {
                perm_str = try utils.i64SliceToUsizeArrayString(attr.ints);
            }
        }
    }

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&param_lib.tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.transpose_onnx_lean(
        \\        T, //type
        \\        @constCast({s}), //input tensor
        \\        {s}, //perm
        \\        &tensor_{s}, //output tensor
        \\        allocator, // pass the local allocator instance
        \\    )
    , .{
        tensor_A_string, // Input tensor
        perm_str, // Permutation array
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_floor(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Floor.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor with floor of input elements (If x is integral, +0, -0, NaN, or infinite, x itself is returned)

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.floor_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_gelu(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    var approximate: []const u8 = "none";
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "approximate")) {
            if (attr.type == AttributeType.STRING) approximate = attr.s;
        }
    }

    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.gelu_lean(T, {s}, "{s}", &tensor_{s})
    , .{
        input_tensor_string,
        approximate,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_tanh(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Tanh.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor with hyperbolic tangent of input elements

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.tanh_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_sqrt(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.sqrt_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_ceil(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Ceil.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor with ceiling of input elements

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.ceil_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_identity(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Identity.html
    // INPUTS:
    //      - input (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - output (heterogeneous) - T: Tensor with same shape and contents as input

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.identity_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_leaky_relu(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor
    // ATTRIBUTES:
    //      - alpha (float, default is 0.01): Coefficient of leakage

    // Get alpha attribute, default to 0.01 if not specified
    var alpha: f32 = 0.01;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "alpha")) {
            if (attr.type == AttributeType.FLOAT) alpha = attr.f;
        }
    }

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.leakyReLU_lean(T, {s}, {d}, &tensor_{s})
    , .{
        input_tensor_string,
        alpha,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_split(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Split.html
    // INPUTS:
    //      - input (heterogeneous) - T: The tensor to split
    //      - split (optional, heterogeneous) - tensor(int64): Optional tensor specifying the size of each split
    // OUTPUTS:
    //      - outputs (variadic, heterogeneous) - T: One or more outputs forming splits of the input
    // ATTRIBUTES:
    //      - axis (int, default is 0): Which axis to split on
    //      - split (list of ints, deprecated): Length of each output. This attribute is deprecated in favor of the 'split' input

    // Get axis attribute (default is 0)
    var axis: i64 = 0;
    var split_sizes_attr: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        } else if (std.mem.eql(u8, attr.name, "split")) {
            if (attr.type == AttributeType.INTS) split_sizes_attr = attr.ints;
        }
    }

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // Handle split sizes - either from input tensor or attribute
    var split_sizes_str: []const u8 = "null";
    var needs_free = false;

    if (node.inputs.items.len > 1 and node.inputs.items[1].?.tensorProto != null) {
        // Split sizes from input tensor (opset 13+)
        const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

        // Extract split sizes from the input tensor
        try writer.print(
            \\
            \\    // Extract split sizes from the input tensor
        , .{});

        // For initializers, access directly from the parameter library
        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
            try writer.print(
                \\
                \\    const split_sizes_tensor_{0s} = param_lib.tensor_{1s};
                \\    var split_sizes_{0s} = allocator.alloc(usize, split_sizes_tensor_{0s}.size) catch @panic("Out of memory");
                \\    defer allocator.free(split_sizes_{0s});
                \\    
                \\    // Convert int64 data to usize
                \\    for (split_sizes_tensor_{0s}.data, 0..) |val, i| {{
                \\        split_sizes_{0s}[i] = @as(usize, @intFromFloat(val));
                \\    }}
            , .{ output_name, try utils.getSanitizedName(node.inputs.items[1].?.name) });
        } else {
            try writer.print(
                \\
                \\    const split_sizes_tensor_{0s} = tensor_{1s};
                \\    var split_sizes_{0s} = allocator.alloc(usize, split_sizes_tensor_{0s}.size) catch @panic("Out of memory");
                \\    defer allocator.free(split_sizes_{0s});
                \\    
                \\    // Convert int64 data to usize
                \\    for (split_sizes_tensor_{0s}.data, 0..) |val, i| {{
                \\        split_sizes_{0s}[i] = @as(usize, @intFromFloat(val));
                \\    }}
            , .{ output_name, try utils.getSanitizedName(node.inputs.items[1].?.name) });
        }

        split_sizes_str = try std.fmt.allocPrint(allocator, "split_sizes_{s}", .{output_name});
        needs_free = true;
    } else if (split_sizes_attr != null) {
        // Split sizes from attribute (deprecated but still supported)
        const split_array_name = try std.fmt.allocPrint(allocator, "split_sizes_{s}", .{try utils.getSanitizedName(node.outputs.items[0].name)});
        defer allocator.free(split_array_name);

        try writer.print(
            \\
            \\    // Define split sizes array
            \\    const {s} = [_]i64{{
        , .{split_array_name});

        for (split_sizes_attr.?, 0..) |size, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{size});
        }

        try writer.print(
            \\}};
            \\
        , .{});

        split_sizes_str = try std.fmt.allocPrint(allocator, "&{s}", .{split_array_name});
        needs_free = true;
    }
    defer if (needs_free) allocator.free(split_sizes_str);

    // Create a different approach that works with the expected types
    try writer.print(
        \\
        \\    // Create array for output tensor pointers to store final results
        \\    var output_ptrs_{s} = [_]*Tensor(T){{
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});

    for (node.outputs.items, 0..) |output, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("&tensor_{s}", .{try utils.getSanitizedName(output.name)});
    }

    try writer.print(
        \\}};
        \\
        \\    // Create temporary tensors that split_lean can operate on
        \\    var temp_tensors_{0s} = allocator.alloc(Tensor(T), {1d}) catch @panic("Out of memory");
        \\    defer {{
        \\        for (temp_tensors_{0s}) |*t| t.deinit();
        \\        allocator.free(temp_tensors_{0s});
        \\    }}
        \\
        \\    // Initialize the temporary tensors
        \\    for (temp_tensors_{0s}) |*t| {{
        \\        t.* = Tensor(T).init(&allocator) catch @panic("Failed to initialize tensor");
        \\    }}
    , .{ try utils.getSanitizedName(node.outputs.items[0].name), node.outputs.items.len });

    // Convert split sizes to usize if provided
    if (!std.mem.eql(u8, split_sizes_str, "null")) {
        try writer.print(
            \\
            \\    // Call split_lean with the extracted split sizes
            \\    tensMath.split_lean(T, {2s}, {3d}, {1s}, &temp_tensors_{0s}) catch unreachable;
        , .{ try utils.getSanitizedName(node.outputs.items[0].name), split_sizes_str, input_tensor_string, axis });
    } else {
        // Get the proper axis value string
        const axis_str = if (axis < 0)
            try std.fmt.allocPrint(allocator, "@intCast((@as(i64, @intCast({s}.shape.len)) + {d}) %% @as(i64, @intCast({s}.shape.len)))", .{ input_tensor_string, axis, input_tensor_string })
        else
            try std.fmt.allocPrint(allocator, "{d}", .{axis});
        defer allocator.free(axis_str);

        try writer.print(
            \\
            \\    // Create default split size array for evenly dividing the tensor
            \\    const dim_size = {0s}.shape[{1s}];
            \\    const num_splits = {2d};
            \\    if (dim_size % num_splits != 0) @panic("Cannot evenly split dimension");
            \\    const split_size = dim_size / num_splits;
            \\    
            \\    const default_split_sizes_{4s} = allocator.alloc(usize, num_splits) catch @panic("Out of memory");
            \\    defer allocator.free(default_split_sizes_{4s});
            \\    for (default_split_sizes_{4s}) |*split_size_item| {{
            \\        split_size_item.* = split_size;
            \\    }}
            \\
            \\    // Call split_lean with default split sizes
            \\    tensMath.split_lean(T, {0s}, {3d}, default_split_sizes_{4s}, &temp_tensors_{4s}) catch unreachable;
        , .{ input_tensor_string, axis_str, node.outputs.items.len, axis, try utils.getSanitizedName(node.outputs.items[0].name) });
    }

    // Now copy the data from temp_tensors to the output tensors
    try writer.print(
        \\
        \\    // Copy data to existing output tensor arrays
        \\    for (temp_tensors_{0s}, 0..) |*src, i| {{
        \\        // Copy data directly to the existing array
        \\        const size_to_copy = @min(src.size, output_ptrs_{0s}[i].size);
        \\        if (size_to_copy > 0) {{
        \\            @memcpy(output_ptrs_{0s}[i].data[0..size_to_copy], src.data[0..size_to_copy]);
        \\        }}
        \\     
        \\        // Shape is pre-allocated statically, just update if needed
        \\        const shape_size_to_copy = @min(src.shape.len, output_ptrs_{0s}[i].shape.len);
        \\        if (shape_size_to_copy > 0) {{
        \\            @memcpy(output_ptrs_{0s}[i].shape[0..shape_size_to_copy], src.shape[0..shape_size_to_copy]);
        \\        }}
        \\        
        \\        // Update the size
        \\        output_ptrs_{0s}[i].size = src.size;
        \\    }}
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});

    // End with a function that returns an error union
    try writer.writeAll(
        \\
        \\    // Final dummy operation that returns an error union
        \\    _ = @import("std").fmt.bufPrint(&[_]u8{}, "", .{})
    );
}

inline fn write_resize(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Resize.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    //      - roi (optional) - T2: ROI (region of interest) tensor
    //      - scales (optional, heterogeneous) - tensor(float): The scale array along each dimension
    //      - sizes (optional, heterogeneous) - tensor(int64): Target size of the output tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Resized output tensor
    // ATTRIBUTES:
    //      - antialias - INT (default is '0')
    //      - axes - INTS
    //      - coordinate_transformation_mode - STRING (default is 'half_pixel')
    //      - cubic_coeff_a - FLOAT (default is '-0.75')
    //      - exclude_outside - INT (default is '0')
    //      - extrapolation_value - FLOAT (default is '0.0')
    //      - keep_aspect_ratio_policy - STRING (default is 'stretch')
    //      - mode - STRING (default is 'nearest')
    //      - nearest_mode - STRING (default is 'round_prefer_floor')
    //

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // ---- optional inputs
    var tensor_roi_string: []const u8 = try allocator.dupe(u8, "null");
    defer {
        if (node.inputs.items.len >= 2 and node.inputs.items[1] != null) {
            allocator.free(tensor_roi_string);
        }
    }
    var data_scales_string: []const u8 = try allocator.dupe(u8, "null");
    defer {
        if (node.inputs.items.len >= 3 and node.inputs.items[2] != null) {
            allocator.free(data_scales_string);
        }
    }
    var data_sizes_string: []const u8 = try allocator.dupe(u8, "null");
    defer {
        if (node.inputs.items.len >= 4 and node.inputs.items[3] != null) {
            allocator.free(data_sizes_string);
        }
    }

    if (node.inputs.items.len >= 2 and node.inputs.items[1] != null) { //----create tensor_roi_string
        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
            tensor_roi_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(node.inputs.items[1].?.name),
                ")",
            });
        } else {
            tensor_roi_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
        }
    }

    if (node.inputs.items.len >= 3 and node.inputs.items[2] != null) { //----create tensor_scales_string
        if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
            data_scales_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "param_lib.tensor_",
                try utils.getSanitizedName(node.inputs.items[2].?.name),
                ".data",
            });
        } else {
            data_scales_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(node.inputs.items[2].?.name), ".data" });
        }
    }

    if (node.inputs.items.len >= 4 and node.inputs.items[3] != null) { //----create tensor_sizes_string
        if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
            data_sizes_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "param_lib.tensor_",
                try utils.getSanitizedName(node.inputs.items[3].?.name),
                ".data",
            });
        } else {
            data_sizes_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(node.inputs.items[3].?.name), ".data" });
        }
    }

    // ---- gasthering ATTRIBUTES from protoNode
    var antialias: i64 = 0;
    var axes: []i64 = &[_]i64{};
    defer allocator.free(axes);
    var coordinate_transformation_mode: []const u8 = try allocator.dupe(u8, "half_pixel");

    var cubic_coeff_a: f64 = -0.75;
    var exclude_outside: i64 = 0;
    var extrapolation_value: f64 = 0.0;
    var keep_aspect_ratio_policy: []const u8 = try allocator.dupe(u8, "stretch");
    defer allocator.free(keep_aspect_ratio_policy);
    var mode: []const u8 = try allocator.dupe(u8, "nearest");

    var nearest_mode: []const u8 = try allocator.dupe(u8, "round_prefer_floor");
    defer allocator.free(nearest_mode);

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "antialias")) |_| {
            if (attr.type == AttributeType.INT) antialias = attr.i else return error.ResizeAnitialiasNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "axes")) |_| {
            if (attr.type == AttributeType.INTS) axes = attr.ints else return error.ResizeAxesNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "coordinate_transformation_mode")) |_| {
            if (attr.type == AttributeType.STRING) coordinate_transformation_mode = attr.s else return error.Resize_coordinate_transformation_mode_NotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "cubic_coeff_a")) |_| {
            if (attr.type == AttributeType.FLOAT) cubic_coeff_a = attr.f else return error.Resize_cubic_coeff_a_NotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "exclude_outside")) |_| {
            if (attr.type == AttributeType.INT) exclude_outside = attr.i else return error.Resize_exclude_outside_NotINT;
        } else if (std.mem.indexOf(u8, attr.name, "extrapolation_value")) |_| {
            if (attr.type == AttributeType.FLOAT) extrapolation_value = attr.f else return error.Resize_extrapolation_value_NotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "keep_aspect_ratio_policy")) |_| {
            if (attr.type == AttributeType.STRING) keep_aspect_ratio_policy = attr.s else return error.Resize_keep_aspect_ratio_policy_NotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "mode")) |_| {
            if (attr.type == AttributeType.STRING) mode = attr.s else return error.Resize_mode_NotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "nearest_mode")) |_| {
            if (attr.type == AttributeType.STRING) nearest_mode = attr.s else return error.Resize_nearest_mode_NotSTRING;
        }
    }

    // ---- CREATING ATTRIBUTES strings
    const axes_string = try utils.i64SliceToUsizeArrayString(axes);
    _ = axes_string;

    //pub fn rezise_lean(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8, output_tensor: *Tensor(T)) !void {
    _ = try writer.print(
        \\
        \\    tensMath.resize_lean(
        \\      T, 
        \\      {s}, //*Tensor(T)
        \\      "{s}", //mode
        \\      {s}, //scales: ?[]const f32
        \\      {s}, //sizes: ?[]const usize
        \\      "{s}", //coordinate_transformation_mode: []const u8
        \\      &tensor_{s}, //output_tensor: *Tensor(T)
        \\    )
    ,
        .{
            tensor_X_string, // input
            mode,
            data_scales_string,
            data_sizes_string,
            coordinate_transformation_mode,
            try utils.getSanitizedName(node.outputs.items[0].name), //output
        },
    );
}

inline fn write_neg(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Neg.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor with flipped elements

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.neg_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_mean(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Mean.html
    // INPUTS:
    //      - Variadic input tensors (data_0, data_1, ...). All inputs must have the same data type.
    // OUTPUTS:
    //      - Output tensor with shape determined by broadcasting the input shapes.
    // ATTRIBUTES:
    //      - None

    if (node.inputs.items.len == 0) {
        return error.EmptyInputList;
    }

    // Costruisci l'array degli input
    var input_strings = std.ArrayList([]u8).init(allocator);
    defer {
        for (input_strings.items) |str| allocator.free(str);
        input_strings.deinit();
    }

    for (node.inputs.items) |input| {
        var input_str: []u8 = undefined;
        if (input.?.tag == globals.TensorTag.INITIALIZER) {
            input_str = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(input.?.name),
                ")",
            });
        } else {
            input_str = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(input.?.name),
            });
        }
        try input_strings.append(input_str);
    }

    // Costruisci la stringa dell'array degli input
    var inputs_array_str = std.ArrayList(u8).init(allocator);
    defer inputs_array_str.deinit();
    try inputs_array_str.writer().writeAll("[_]*Tensor(f32){ ");
    for (input_strings.items, 0..) |input_str, i| {
        if (i > 0) try inputs_array_str.writer().writeAll(", ");
        try inputs_array_str.writer().writeAll(input_str);
    }
    try inputs_array_str.writer().writeAll(" }");

    // Scrivi la chiamata a tensMath.mean_lean
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);
    _ = try writer.print(
        \\
        \\
        \\    var inputs_{s} = {s};
        \\    tensMath.mean_lean(f32, &inputs_{s}, &tensor_{s})
    , .{
        output_name, // Nome della variabile temporanea degli input
        inputs_array_str.items, // Array dei puntatori ai tensori di input
        output_name, // Nome del tensore di output
        output_name, // Nome della variabile temporanea per il riferimento all'array
    });
}

inline fn write_pads(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Pad.html
    // INPUTS:
    //      - data (T): Input tensor.
    //      - pads (tensor(int64)): Tensor of integers indicating the number of padding elements.
    //          Shape [2 * num_axes], format [x1_begin, x2_begin, ..., x1_end, x2_end,...]
    //      - constant_value (optional, T): Scalar constant value to use for constant mode. Defaults to 0.
    //      - axes (optional, tensor(int64)): Axes to pad. If not provided, all axes are padded.
    // OUTPUTS:
    //      - output (T): Tensor after padding.
    // ATTRIBUTES:
    //      - mode (STRING, default is 'constant'): Supported modes: constant, reflect, edge, wrap.

    // Get mode attribute
    var mode_str: []const u8 = "constant"; // Default
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "mode")) {
            if (attr.type == AttributeType.STRING) mode_str = attr.s;
            break;
        }
    }
    // Convert mode string to PadMode enum
    var pad_mode_enum: []const u8 = undefined;
    if (std.ascii.eqlIgnoreCase(mode_str, "constant")) {
        pad_mode_enum = "tensMath.PadMode.constant";
    } else if (std.ascii.eqlIgnoreCase(mode_str, "reflect")) {
        pad_mode_enum = "tensMath.PadMode.reflect";
    } else if (std.ascii.eqlIgnoreCase(mode_str, "edge")) {
        pad_mode_enum = "tensMath.PadMode.edge";
    } else if (std.ascii.eqlIgnoreCase(mode_str, "wrap")) {
        pad_mode_enum = "tensMath.PadMode.wrap";
    } else {
        return error.UnsupportedMode;
    }

    // --- Get Input Strings ---

    // Input 0: data
    const data_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    const data_tensor_string = try std.fmt.allocPrint(allocator, "{s}tensor_{s}{s}", .{ if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) "@constCast(&param_lib." else "&", data_name, if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) ")" else "" });
    defer allocator.free(data_tensor_string);

    // Input 1: pads (must be int64 constant)
    if (node.inputs.items.len < 2 or node.inputs.items[1] == null or node.inputs.items[1].?.tag != globals.TensorTag.INITIALIZER) {
        return error.PadsInputInvalid;
    }
    const pads_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
    const pads_data_string = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data", .{pads_name});
    defer allocator.free(pads_data_string);

    // Input 2: constant_value (optional)
    var constant_value_str: []const u8 = "null";
    var constant_value_alloc: ?[]u8 = null;
    if (node.inputs.items.len > 2 and node.inputs.items[2] != null) {
        const const_val_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
        // Constant value should be a scalar, access data[0]
        if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
            constant_value_alloc = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data[0]", .{const_val_name});
        } else {
            constant_value_alloc = try std.fmt.allocPrint(allocator, "tensor_{s}.data[0]", .{const_val_name});
        }
        constant_value_str = constant_value_alloc.?;
    }
    defer if (constant_value_alloc != null) allocator.free(constant_value_alloc.?);

    // Input 3: axes (optional, int64 or int32)
    var axes_data_str: []const u8 = "null";
    var axes_alloc: ?[]u8 = null;
    var axes_code_to_generate: ?[]u8 = null; // Holds the "utils.sliceToIsizeSlice(...)" string
    var axes_var_name_arg: []const u8 = "null"; // Holds the argument for pads_lean ("null" or "axes_isize_...")
    var axes_var_name_allocated = false;
    var axes_code_allocated = false;

    if (node.inputs.items.len > 3 and node.inputs.items[3] != null) {
        const axes_name = try utils.getSanitizedName(node.inputs.items[3].?.name);
        const output_name_tmp = try utils.getSanitizedName(node.outputs.items[0].name); // Needed for var name

        if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
            axes_alloc = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data", .{axes_name});
        } else {
            axes_alloc = try std.fmt.allocPrint(allocator, "tensor_{s}.data", .{axes_name});
        }
        axes_data_str = axes_alloc.?;
        // Generate code to convert axes data to isize slice
        axes_code_to_generate = try std.fmt.allocPrint(allocator, "utils.sliceToIsizeSlice({s})", .{axes_data_str});
        axes_code_allocated = true;
        axes_var_name_arg = try std.fmt.allocPrint(allocator, "axes_isize_{s}", .{output_name_tmp}); // Generate the variable name to pass
        axes_var_name_allocated = true;
    }
    defer if (axes_alloc != null) allocator.free(axes_alloc.?);
    defer if (axes_code_allocated) allocator.free(axes_code_to_generate.?);
    defer if (axes_var_name_allocated) allocator.free(axes_var_name_arg);

    // Output tensor
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Conditionally create the isize slice for axes if needed
    if (axes_code_to_generate != null) {
        _ = try writer.print(
            \\    const {s} = {s};
            \\    defer allocator.free({s});
        , .{ axes_var_name_arg, axes_code_to_generate.?, axes_var_name_arg });
    }

    _ = try writer.print(
        \\
        \\    tensMath.pads_lean(
        \\        T, // type
        \\        {s}, // data
        \\        {s}, // pads (int64 slice)
        \\        {s}, // mode
        \\        {s}, // constant_value
        \\        {s}, // axes (isize slice)
        \\        &tensor_{s} // output
        \\    )
    , .{
        data_tensor_string,
        pads_data_string,
        pad_mode_enum,
        constant_value_str,
        axes_var_name_arg, // Use the correct variable name or "null"
        output_name,
    });
}

inline fn write_clip(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Clip.html
    // INPUTS:
    //      - input (heterogeneous) - T: Input tensor whose elements to be clipped.
    //      - min (optional, heterogeneous) - T: Minimum value, must be a scalar.
    //      - max (optional, heterogeneous) - T: Maximum value, must be a scalar.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Output tensor with clipped values.

    // Get sanitized names
    const input_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", input_name, ")" });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", input_name });
    }

    // Create optional min tensor string
    var min_tensor_string: []const u8 = "null";
    var min_alloc: ?[]u8 = null;
    if (node.inputs.items.len > 1 and node.inputs.items[1] != null) {
        const min_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
            min_alloc = try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{min_name});
        } else {
            min_alloc = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{min_name});
        }
        min_tensor_string = min_alloc.?;
    }
    defer if (min_alloc != null) allocator.free(min_alloc.?);

    // Create optional max tensor string
    var max_tensor_string: []const u8 = "null";
    var max_alloc: ?[]u8 = null;
    if (node.inputs.items.len > 2 and node.inputs.items[2] != null) {
        const max_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
        if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
            max_alloc = try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{max_name});
        } else {
            max_alloc = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{max_name});
        }
        max_tensor_string = max_alloc.?;
    }
    defer if (max_alloc != null) allocator.free(max_alloc.?);

    // Write the lean_clip function call
    _ = try writer.print(
        \\
        \\
        \\    tensMath.clip_lean(
        \\        T, // type
        \\        {s}, // input tensor
        \\        {s}, // min tensor (optional)
        \\        {s}, // max tensor (optional)
        \\        &tensor_{s} // output tensor
        \\    )
    , .{
        input_tensor_string,
        min_tensor_string,
        max_tensor_string,
        output_name,
    });
}

inline fn write_dynamicQuantizeLinear(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx_aionnx_preview_training__DynamicQuantizeLinear.html
    // INPUTS:
    //      - x (heterogeneous) - T1: Input tensor
    // OUTPUTS:
    //      - y (heterogeneous) - T2: Quantized output tensor
    //      - y_scale (heterogeneous) - tensor(float): Output scale. It's a scalar.
    //      - y_zero_point (heterogeneous) - T2: Output zero point. It's a scalar.

    // Ensure correct number of inputs and outputs
    if (node.inputs.items.len != 1) return error.InvalidInputCount; // Expects 1 input
    if (node.outputs.items.len != 3) return error.InvalidOutputCount; // Expects 3 outputs

    // Get sanitized names
    const input_x_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    const output_y_name = try utils.getSanitizedName(node.outputs.items[0].name);
    const output_scale_name = try utils.getSanitizedName(node.outputs.items[1].name);
    const output_zp_name = try utils.getSanitizedName(node.outputs.items[2].name);

    // Create input tensor string (needs const cast as lean function expects *const)
    var input_x_string: []u8 = undefined;
    defer allocator.free(input_x_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_", input_x_name, ")",
        });
    } else {
        input_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_", input_x_name, ")",
        });
    }

    // Output tensors are always non-const variables in the generated code
    const output_y_string = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{output_y_name});
    defer allocator.free(output_y_string);
    const output_scale_string = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{output_scale_name});
    defer allocator.free(output_scale_string);
    const output_zp_string = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{output_zp_name});
    defer allocator.free(output_zp_string);

    // Write the lean function call
    _ = try writer.print(
        \\    tensMath.dynamicQuantizeLinear_lean(
        \\        {s}, // x: *const Tensor(f32)
        \\        {s}, // y: *Tensor(u8)
        \\        {s}, // y_scale: *Tensor(f32)
        \\        {s}  // y_zero_point: *Tensor(u8)
        \\    )
    , .{
        input_x_string,
        output_y_string,
        output_scale_string,
        output_zp_string,
    });
}

inline fn write_cast(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Cast.html
    // INPUTS:
    //      - input (heterogeneous) - T1: Input tensor to be cast.
    // OUTPUTS:
    //      - output (heterogeneous) - T2: Output tensor with the same shape as input and specified type.
    // ATTRIBUTES:
    //      - to (INT, required): The data type to cast to.

    // Get the target type from the attribute
    var target_type: DataType = undefined;
    var target_type_found = false;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "to")) {
            if (attr.type == AttributeType.INT) {
                target_type = @enumFromInt(attr.i);
                target_type_found = true;
                break;
            } else {
                return error.CastToAttributeNotINT;
            }
        }
    }

    if (!target_type_found) {
        return error.CastToAttributeNotFound;
    }

    // --- Safely get source type ---
    var source_type: DataType = .UNDEFINED;
    const input_ready_tensor_ptr = globals.tensorHashMap.getPtr(node.inputs.items[0].?.name);

    if (input_ready_tensor_ptr) |rt_ptr| {
        // Prioritize ReadyTensor.dtype if it's valid
        if (rt_ptr.dtype != DataType.UNDEFINED) { // Check if dtype is set
            source_type = rt_ptr.dtype;
        } else if (rt_ptr.tensorProto) |tp| {
            // Fallback to tensorProto if dtype is not set
            source_type = tp.data_type;
        } else {
            mathHandler_log.warn("Error: Could not determine source type for Cast input '{s}' from either dtype or tensorProto\n", .{node.inputs.items[0].?.name});
            return error.DataTypeNotFound; // Or another appropriate error
        }
    } else {
        mathHandler_log.warn("Error: Cast input tensor '{s}' not found in map\n", .{node.inputs.items[0].?.name});
        return error.TensorNotFound; // Or another appropriate error
    }

    if (source_type == DataType.UNDEFINED) {
        mathHandler_log.warn("Error: Determined source type for Cast input '{s}' is UNDEFINED\n", .{node.inputs.items[0].?.name});
        return error.DataTypeNotFound;
    }
    // --- End safe source type retrieval ---

    const source_type_string = try utils.getTypeString(source_type);
    const target_type_string = try utils.getTypeString(target_type);

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\
        \\    @setEvalBranchQuota(10000);
        \\    tensMath.cast_lean(
        \\        {s}, // Source type T1
        \\        {s}, // Target type T2
        \\        {s}, // Input tensor (*const Tensor(T1))
        \\        &tensor_{s}, // Output tensor (*Tensor(T2))
        \\        zant.onnx.DataType.{s} // Target DataType enum
        \\    )
    , .{
        source_type_string, // Pass source type string
        target_type_string, // Pass target type string
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
        @tagName(target_type), // Pass the DataType enum value as the 5th arg
    });
}

inline fn write_convInteger(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // INPUTS:
    //      - x: Input tensor (u8 or i8)
    //      - w: Weight tensor (u8 or i8)
    //      - x_zero_point: Zero point for input tensor x (optional, u8 or i8)
    //      - w_zero_point: Zero point for weight tensor w (optional, u8 or i8)
    // OUTPUTS:
    //      - y: Output tensor (i32)
    // ATTRIBUTES:
    //      - auto_pad, dilations, group, kernel_shape, pads, strides (similar to Conv)

    var auto_pad: []const u8 = "NOTSET";
    var dilations: ?[]i64 = null;
    var group: i64 = 1;
    var kernel_shape: ?[]i64 = null;
    var pads: ?[]i64 = null;
    var strides: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
            if (attr.type == AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
        }
    }

    //----create tensor_x_string
    var tensor_x_string: []u8 = undefined;
    defer allocator.free(tensor_x_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_w_string
    var tensor_w_string: []u8 = undefined;
    defer allocator.free(tensor_w_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_w_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_w_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    //----create ?x_zero_point string
    var x_zp_string: []u8 = undefined;
    var free_x_zp = false;
    if (node.inputs.items.len > 2 and node.inputs.items[2] != null) { // Index 2 might be x_zero_point
        const x_zp_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
        if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
            x_zp_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", x_zp_name, ")" });
        } else {
            x_zp_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", x_zp_name, ")" });
        }
        free_x_zp = true;
    } else {
        x_zp_string = try allocator.dupe(u8, "null");
        free_x_zp = true;
    }
    defer if (free_x_zp) allocator.free(x_zp_string);

    //----create ?w_zero_point string
    var w_zp_string: []u8 = undefined;
    var free_w_zp = false;
    if (node.inputs.items.len > 3 and node.inputs.items[3] != null) { // Index 3 might be w_zero_point
        const w_zp_name = try utils.getSanitizedName(node.inputs.items[3].?.name);
        if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
            w_zp_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", w_zp_name, ")" });
        } else {
            w_zp_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", w_zp_name, ")" });
        }
        free_w_zp = true;
    } else {
        w_zp_string = try allocator.dupe(u8, "null");
        free_w_zp = true;
    }
    defer if (free_w_zp) allocator.free(w_zp_string);

    //----create stride string (mandatory)
    if (strides == null) return error.StrideNotFound;
    const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(strides.?);

    //----create ?pads string
    var pads_string: []const u8 = "null";
    if (pads != null) {
        if (pads.?.len > 0) { // Check if the slice is actually non-empty
            pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
            // Assuming no allocation needed to be freed, following write_conv
        } else {
            pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
        }
    } // else pads_string remains "null"

    //----create ?dilations string
    var dilat_string: []const u8 = "null";
    if (dilations != null) {
        if (dilations.?.len > 0) {
            dilat_string = try utils.i64SliceToUsizeArrayString(dilations.?);
        } else {
            dilat_string = "&[_]usize{}";
        }
    } // else dilat_string remains "null"

    // Get the specific data types for input and weight tensors
    const input_x_type = globals.tensorHashMap.get(node.inputs.items[0].?.name).?.dtype;
    const input_w_type = globals.tensorHashMap.get(node.inputs.items[1].?.name).?.dtype;

    const type_str_x = try utils.getTypeString(input_x_type);
    const type_str_w = try utils.getTypeString(input_w_type);

    _ = try writer.print(
        \\    
        \\
        \\    tensMath.convInteger_lean(
        \\        {s}, // T1: Input data type (u8 or i8)
        \\        {s}, // T2: Weight data type (u8 or i8)
        \\        {s}, // x
        \\        {s}, // w
        \\        {s}, // x_zero_point
        \\        {s}, // w_zero_point
        \\        &tensor_{s}, // y (Output is always i32)
        \\        {s}, // stride
        \\        {s}, // pads
        \\        {s}, // dilations
        \\        {}, // group
        \\        "{s}", // auto_pad
        \\    )
    , .{
        type_str_x, // T1 type string
        type_str_w, // T2 type string
        tensor_x_string, // x
        tensor_w_string, // w
        x_zp_string, // x_zero_point
        w_zp_string, // w_zero_point
        try utils.getSanitizedName(node.outputs.items[0].name), // y
        stride_string, // Strides
        pads_string, // Pads
        dilat_string, // Dilations
        group, // Group
        auto_pad, // auto_pad
    });
}

// Helper function to safely get tensor type string
fn getSafeTensorTypeString(input_node_item: *globals.ReadyTensor, parent_node_name: []const u8) ![]const u8 {
    const input_name = input_node_item.name; // Name is not optional on ReadyTensor
    const tensor_global = input_node_item; // tensor_global is the input_node_item itself

    // Prioritize ReadyTensor.dtype if available and valid
    if (tensor_global.dtype != .UNDEFINED) {
        return utils.getTypeString(tensor_global.dtype);
    }

    // Fallback to tensorProto if dtype is not available/valid
    const onnx_tensor_proto = tensor_global.tensorProto orelse {
        std.log.err(
            \\Error in node '{s}': tensorProto is null AND ReadyTensor.dtype is UNDEFINED for input tensor '{s}'.
            \\Tensor details: ready={}, tag={s}, shape={any}, dtype={s}.
            \\This means type information is missing for this tensor.
        , .{
            parent_node_name,
            input_name,
            tensor_global.ready,
            @tagName(tensor_global.tag),
            tensor_global.shape,
            @tagName(tensor_global.dtype), // Also log the dtype
        });
        return error.CodegenMissingTypeInformation; // New, more specific error (ensure this is defined)
    };

    return utils.getTypeString(onnx_tensor_proto.data_type);
}

inline fn write_batch_norm(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1
    //      - scale (heterogeneous) - T1: Scale tensor of shape ©.
    //      - B (heterogeneous) - T1: Bias tensor of shape ©.
    //      - input_mean (heterogeneous) - T2: running (training) or estimated (testing) mean tensor of shape ©.
    //      - input_var (heterogeneous) - T2: running (training) or estimated (testing) variance tensor of shape ©.
    // OUTPUT:
    //      - Y (heterogeneous) - T: The output tensor of the same shape as X
    // ATTRIBUTES:
    //      - epsilon - FLOAT (default is '1e-05'): The epsilon value to use to avoid division by zero.
    //      - momentum - FLOAT (default is '0.9'): Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum).
    //      - training_mode - INT (default is '0'): If set to true, it indicates BatchNormalization is being used for training, and outputs 1 and 2 are to be computed.

    var epsilon: f32 = 1e-05;
    var momentum: f32 = 0.9;
    // var training_mode: bool = false; -> NOT USED, ALWAYS FALSE for Zant

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "epsilon")) |_| {
            if (attr.type == AttributeType.FLOAT) epsilon = attr.f else return error.BatchNorm_epsilon_NotFloat;
        } else if (std.mem.indexOf(u8, attr.name, "momentum")) |_| {
            if (attr.type == AttributeType.FLOAT) momentum = attr.f else return error.BatchNorm_momentum_NotFloat;
        } else if (std.mem.indexOf(u8, attr.name, "training_mode")) |_| {
            if (attr.type == AttributeType.INT) if (attr.i != 0) return error.BatchNorm_training_NotAvailable;
        }
    }

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_scale_string
    var tensor_scale_string: []u8 = undefined;
    defer allocator.free(tensor_scale_string);

    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    //----create tensor_scale_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);

    if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[2].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[2].?.name), ")" });
    }

    //----create tensor_input_mean_string
    var tensor_input_mean_string: []u8 = undefined;
    defer allocator.free(tensor_input_mean_string);

    if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_input_mean_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[3].?.name),
            ")",
        });
    } else {
        tensor_input_mean_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[3].?.name), ")" });
    }

    //----create tensor_input_var_string
    var tensor_input_var_string: []u8 = undefined;
    defer allocator.free(tensor_input_var_string);

    if (node.inputs.items[4].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_input_var_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[4].?.name),
            ")",
        });
    } else {
        tensor_input_var_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[4].?.name), ")" });
    }

    // pub inline fn batchNormalization_lean( comptime T: anytype, comptime T1: anytype, comptime T2: anytype, input: *Tensor(T), scales: *Tensor(T1), B: *Tensor(T1), input_mean: Tensor(T2), input_var: Tensor(T2), epsilon: f32, momentum: f32, training_mode: bool, output: *Tensor(T))
    _ = try writer.print(
        \\    
        \\
        \\    tensMath.batchNormalization_lean(
        \\        {s}, //type 0
        \\        {s}, //type 1
        \\        {s}, //type 2
        \\        {s}, //input
        \\        {s}, //scales
        \\        {s}, //B
        \\        {s}, //input_mean
        \\        {s}, //input_var
        \\        {}, //epsilon
        \\        {}, //momentum
        \\        false, //training_mode
        \\        &tensor_{s}, //output
        \\    )
    , .{
        try getSafeTensorTypeString(node.inputs.items[0].?, node.nodeProto.name orelse "UnnamedBatchNormInput0"), // MODIFIED: Use helper for input X type
        try getSafeTensorTypeString(node.inputs.items[1].?, node.nodeProto.name orelse "UnnamedBatchNormInput1"), // MODIFIED: Use helper for input scale type
        try getSafeTensorTypeString(node.inputs.items[3].?, node.nodeProto.name orelse "UnnamedBatchNormInput3"), // MODIFIED: Use helper for input mean/var type (check ONNX spec for correct index if this is not mean's type)
        tensor_X_string,
        tensor_scale_string,
        tensor_B_string,
        tensor_input_mean_string,
        tensor_input_var_string,
        epsilon,
        momentum,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}
