const std = @import("std");
const zant = @import("zant");
const UOpBuilder = zant.uops.UOpBuilder;
const allocator = zant.utils.allocator.allocator;
const NodeZant = zant.IR_graph.NodeZant;
const operators = zant.IR_graph.operators;
const math = zant.core.tensor.math_standard;
const TensorType = zant.IR_graph.TensorType;
const DType = zant.uops.DType;

// TODO where to get out_Dtype?
pub fn tensorTypeToDtype(tensor_type: TensorType) DType {
    return switch (tensor_type) {
        .f16 => DType.f16,
        .f32 => DType.f32,
        .f64 => DType.f64,
        .i8 => DType.i8,
        .i16 => DType.i16,
        .i32 => DType.i32,
        .i64 => DType.i64,
        .u8 => DType.u8,
        .u16 => DType.u16,
        .u32 => DType.u32,
        .u64 => DType.u64,
        .bool => DType.bool,
    };
}

pub fn render_lower_math_op(builder: *UOpBuilder, nodeZant: *NodeZant) !void {

    // Ensure the node has a name for debugging purposes
    if (nodeZant.nodeProto.name == null) {
        // Generate a name like "OpType_OutputName"
        const op_type = nodeZant.nodeProto.op_type;
        const output_name = nodeZant.outputs.items[0].name; // Directly assign since it's not optional
        _ = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ op_type, output_name }); // Keep allocation for potential local use or logging if needed, but don't assign. Free later if stored.
        // Note: This allocated name needs to be managed if the NodeProto lifetime extends beyond this scope.
        // Assuming the global allocator lives long enough or NodeProto is processed quickly.
    }

    if (std.mem.eql(u8, nodeZant.op_type, "Add")) {
        //https://onnx.ai/onnx/operators/onnx__Add.html
        try render_lower_add(builder, nodeZant.op.add);
    } else if (std.mem.eql(u8, nodeZant.op_type, "AveragePool")) {
        //try renders.render_lower_averagePool(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "BatchNormalization")) {
        //https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
        //try renders.render_lower_batchNormalization(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Cast")) {
        // https://onnx.ai/onnx/operators/onnx__Cast.html
        //try renders.render_lower_cast(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Ceil")) {
        //https://onnx.ai/onnx/operators/onnx__Ceil.html
        //try renders.render_lower_ceil(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Clip")) {
        //https://onnx.ai/onnx/operators/onnx__Clip.html
        //try renders.render_lower_clip(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Concat")) {
        //https://onnx.ai/onnx/operators/onnx__Concat.html
        //try renders.render_lower_concat(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Constant")) {
        //https://onnx.ai/onnx/operators/onnx__Constant.html
        //try renders.render_lower_constant(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Conv")) {
        //https://onnx.ai/onnx/operators/onnx__Conv.html
        try render_lower_conv2d(builder, nodeZant.op.conv);
        //try renders.render_lower_conv(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "ConvInteger")) {
        //https://onnx.ai/onnx/operators/onnx__ConvInteger.html
        //try renders.render_lower_convInteger(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Div")) {
        //https://onnx.ai/onnx/operators/onnx__Div.html
        //try renders.render_lower_Div(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "DynamicQuantizeLinear")) {
        // https://onnx.ai/onnx/operators/onnx_aionnx_preview_training__DynamicQuantizeLinear.html
        //try renders.render_lower_dynamicQuantizeLinear(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Flatten")) {
        return error.OperationWIP;
    } else if (std.mem.eql(u8, nodeZant.op_type, "Gather")) {
        //try renders.render_lower_gather(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Gemm")) {
        //https://onnx.ai/onnx/operators/onnx__Gemm.html
        //try renders.render_lower_gemm(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "LeakyRelu")) {
        //try renders.render_lower_leaky_relu(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "LogSoftmax")) {
        //try renders.render_lower_longsoftmax(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "MatMul")) {
        //try renders.render_lower_matmul(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "MaxPool")) {
        try render_lower_maxpool2d(builder, nodeZant.op.maxpool);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Mul")) {
        //https://onnx.ai/onnx/operators/onnx__Mul.html
        //try renders.render_lower_mul(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Neg")) {
        //https://onnx.ai/onnx/operators/onnx__Neg.html
        try render_lower_neg(builder, nodeZant.op.neg);
    } else if (std.mem.eql(u8, nodeZant.op_type, "OneHot")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, nodeZant.op_type, "Pad")) {
        //https://onnx.ai/onnx/operators/onnx__Pad.html
        //try renders.render_lower_pads(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "ReduceMean")) {
        //try renders.render_lower_reducemean(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Relu")) {
        //https://onnx.ai/onnx/operators/onnx__Relu.html
        try render_lower_relu(builder, nodeZant.op.relu);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Reshape")) {
        // https://onnx.ai/onnx/operators/onnx__Reshape.html
        try render_lower_reshape(builder, nodeZant.op.reshape);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Resize")) {
        //try renders.render_lower_resize(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Shape")) {
        //try renders.render_lower_shape(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Sigmoid")) {
        //try renders.render_lower_sigmoid(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Softmax")) {
        //https://onnx.ai/onnx/operators/onnx__Softmax.html
        //try renders.render_lower_softmax(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Slice")) {
        //try renders.render_lower_slice(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Split")) {
        //https://onnx.ai/onnx/operators/onnx__Split.html
        //try renders.render_lower_split(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Sub")) {
        //https://onnx.ai/onnx/operators/onnx__Sub.html
        //try renders.render_lower_Sub(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Tanh")) {
        //try renders.render_lower_tanh(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Transpose")) {
        //try renders.render_lower_transpose(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Unsqueeze")) {
        //try renders.render_lower_unsqueeze(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Identity")) {
        //https://onnx.ai/onnx/operators/onnx__Identity.html
        //try renders.render_lower_identity(writer, nodeZant);
    } else if (std.mem.eql(u8, nodeZant.op_type, "Mean")) {
        // https://onnx.ai/onnx/operators/onnx__Mean.html
        //try renders.render_lower_mean(writer, nodeZant);
    } else {
        return error.OperationNotSupported;
    }
}

pub fn render_lower_add(builder: *UOpBuilder, add: operators.Add) !void {
    const A_id = add.input_A.get_tensorZantID();
    const B_id = add.input_B.get_tensorZantID();
    const out_shape = add.get_output_shape();
    const strideA = add.input_A.stride;
    const strideB = add.input_B.stride;
    const out_dtype = tensorTypeToDtype(add.output_C.ty);

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = math.lowerAdd(
        builder,
        A_id,
        B_id,
        out_shape,
        strideA,
        strideB,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}

pub fn render_lower_maxpool2d(builder: *UOpBuilder, maxpool: operators.MaxPool) !void {
    const X_id = maxpool.input_X.get_tensorZantID();
    const out_shape = maxpool.get_output_shape();
    const in_stride = maxpool.input_X.stride;
    const pads = [2]usize{ @as(usize, maxpool.pads[0]), @as(usize, maxpool.pads[1]) };
    const strides_hw = [2]usize{ maxpool.strides[0], maxpool.strides[1] };
    const dil_hw = [2]usize{ maxpool.dilations[0], maxpool.dilations[1] };
    const kHW = [2]usize{ maxpool.kernel_shape[0], maxpool.kernel_shape[1] };
    const out_dtype = tensorTypeToDtype(maxpool.output_Y.ty);

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = math.lowerMaxPool2d(
        &builder,
        X_id,
        out_shape,
        in_stride,
        pads,
        strides_hw,
        dil_hw,
        kHW,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}

pub fn render_lower_matMul(builder: *UOpBuilder, matmul: operators.MatMul) !void {
    const A_id = matmul.input_A.get_tensorZantID();
    const B_id = matmul.input_B.get_tensorZantID();
    const out_shape = matmul.get_output_shape();
    const strideA = matmul.input_A.stride;
    const strideB = matmul.input_B.stride;
    const out_dtype = tensorTypeToDtype(matmul.output_C.ty);

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = math.lowerMatMul(
        &builder,
        A_id,
        B_id,
        out_shape,
        strideA,
        strideB,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}

pub fn render_lower_neg(builder: *UOpBuilder, neg: operators.Neg) !void {
    const A_id = neg.input_X.get_tensorZantID();
    const StrideA = neg.input_X.stride;
    const out_shape = neg.get_output_shape();
    const out_dtype = tensorTypeToDtype(neg.output_Y.ty);

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = math.lowerNeg(
        &builder,
        A_id,
        StrideA,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}

pub fn render_lower_relu(builder: *UOpBuilder, relu: operators.Relu) !void {
    const X_id = relu.input_X.get_tensorZantID();
    const out_shape = relu.get_output_shape();
    const out_dtype = tensorTypeToDtype(relu.output_Y.ty);

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = math.lowerReLU(
        &builder,
        X_id,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}

pub fn render_lower_reshape(builder: *UOpBuilder, reshape: operators.Reshape) !void {
    const X_id = reshape.data.get_tensorZantID();
    const out_shape = reshape.get_output_shape();
    const out_dtype = tensorTypeToDtype(reshape.reshaped.ty);

    const out_buf_id = try math.lowerReshape(
        &builder,
        X_id,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}

pub fn render_lower_conv2d(builder: *UOpBuilder, conv: operators.Conv) !void {
    const X_id = conv.input_X.get_tensorZantID();
    const out_shape = conv.get_output_shape();
    const out_dtype = tensorTypeToDtype(conv.output_Y.ty);

    const out_buf_id = try math.lowerConv2d(
        &builder,
        X_id,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning
}
