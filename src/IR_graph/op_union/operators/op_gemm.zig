const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const tensorMath = zant.core.tensor.math_standard;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const TensorCategory = tensorZant.TensorCategory;

const utils = @import("codegen").utils;

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

pub const Gemm = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    input_C: ?*TensorZant,
    output: *TensorZant,
    //attributes:
    alpha: f32, // = 1.0;
    beta: f32, // = 1.0;
    transA: bool, // = false;
    transB: bool, // = false;

    pub fn init(nodeProto: *NodeProto) !Gemm {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const input_C = if (nodeProto.input.len > 2) if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_C_notFound else null;
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 1.0;
        var beta: f32 = 1.0;
        var transA: bool = false;
        var transB: bool = false;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
                if (attr.type == onnx.AttributeType.INT) transA = if (attr.i != 0) true else false else return error.GemmTransANotINT;
            } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
                if (attr.type == onnx.AttributeType.INT) transB = if (attr.i != 0) true else false else return error.GemmTransBNotINT;
            }
        }

        return Gemm{
            .input_A = input_A,
            .input_B = input_B,
            .input_C = input_C,
            .output = output,
            .alpha = alpha,
            .beta = beta,
            .transA = transA,
            .transB = transB,
        };
    }

    pub fn get_output_shape(self: Gemm) []usize {
        return self.output.getShape();
    }

    pub fn get_output_tensor(self: Gemm) *TensorZant {
        return self.output;
    }

    pub fn write_op(self: Gemm, writer: std.fs.File.Writer) !void {
        //----create tensor_A_string
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);

        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_A.name) });
        }

        //----create tensor_B_string
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_B.name) });
        }

        // Input Tensor C is optional! verify the presence
        var tensor_C_string: []u8 = undefined;
        if (self.input_C) |in_C| {
            const sanitized_tensor_C = try utils.getSanitizedName(in_C.name);
            tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (in_C.tc == TensorCategory.INITIALIZER) "param_lib." else "",
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
            self.alpha,
            self.beta,
            if (self.transA) "true" else "false",
            if (self.transB) "true" else "false",
            try utils.getSanitizedName(self.output.name), // Output
        });
    }

    pub fn compute_output_shape() []usize {} // TODO

    pub fn print(self: Gemm) void { // TODO
        std.debug.print("\n Gemm:\n {any}", .{self});
    }
};
