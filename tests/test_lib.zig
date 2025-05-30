const std = @import("std");
const test_options = @import("test_options");
const test_name = test_options.test_name;

comptime {
    _ = @import("Core/Tensor/TensorMath/test_op_convolution.zig");
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("ImageToTensor/jpeg/test_jpeg_decoder.zig");
    _ = @import("ImageToTensor/test_utils.zig");
    _ = @import("CodeGen//renderer//test_zig_renderer.zig");
    _ = @import("IR_graph/IR_graph.zig");

    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "libElementWise")) {
    //     _ = @import("Core/Tensor/TensorMath/test_lib_elementWise_math.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "libLogical")) {
    //     _ = @import("Core/Tensor/TensorMath/test_lib_logical_math.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "libReduction")) {
    //     _ = @import("Core/Tensor/TensorMath/test_lib_reduction_math.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "libShape")) {
    //     _ = @import("Core/Tensor/TensorMath/test_lib_shape_math.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "opConvolution")) {
    //     _ = @import("Core/Tensor/TensorMath/test_op_convolution.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "opElu")) {
    //     _ = @import("Core/Tensor/TensorMath/test_op_elu.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "opGemm")) {
    //     _ = @import("Core/Tensor/TensorMath/test_op_gemm.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "opOneHot")) {
    //     _ = @import("Core/Tensor/TensorMath/test_op_oneHot.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "opMat")) {
    //     _ = @import("Core/Tensor/TensorMath/test_op_mat_mul.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "opPooling")) {
    //     _ = @import("Core/Tensor/TensorMath/test_op_pooling.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "tensorMath")) {
    //     _ = @import("Core/Tensor/TensorMath/test_tensor_math.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "tensor")) {
    //     _ = @import("Core/Tensor/test_tensor.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "core")) {
    //     _ = @import("Core/test_core.zig");
    // }
    // if (test_name.len == 0 or std.mem.eql(u8, test_name, "utils")) {
    //     _ = @import("Utils/test_utils.zig");
    // }
}
