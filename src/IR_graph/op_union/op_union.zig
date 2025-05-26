const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const allocator = std.heap.page_allocator;
pub const operators = @import("operators/operators.zig");

pub const Op_union = union(enum) {
    add: operators.Add,
    averagePool: operators.AveragePool,
    batchNormalization: operators.BatchNormalization,
    ceil: operators.Ceil,
    concat: operators.Concat,
    conv: operators.Conv,
    div: operators.Div,
    elu: operators.Elu,
    flatten: operators.Flatten,
    gather: operators.Gather,
    gemm: operators.Gemm,
    identity: operators.Identity,
    leakyRelu: operators.LeakyRelu,
    matMul: operators.MatMul,
    maxPool: operators.MaxPool,
    mul: operators.Mul,
    neg: operators.Neg,
    reduceMean: operators.ReduceMean,
    relu: operators.Relu,
    reshape: operators.Reshape,
    resize: operators.Resize,
    shape: operators.Shape,
    sigmoid: operators.Sigmoid,
    slice: operators.Slice,
    softmax: operators.Softmax,
    split: operators.Split,
    sub: operators.Sub,
    tanh: operators.Tanh,
    transpose: operators.Transpose,
    unsqueeze: operators.Unsqueeze,

    useless: operators.Useless,

    pub fn init(nodeProto: *NodeProto) !Op_union {
        const op_type = nodeProto.op_type;
        if (std.mem.eql(u8, op_type, "Add")) {
            return Op_union{ .add = try operators.Add.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "AveragePool")) {
            return Op_union{ .averagePool = try operators.AveragePool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "BatchNormalization")) {
            return Op_union{ .batchNormalization = try operators.BatchNormalization.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Ceil")) {
            return Op_union{ .ceil = try operators.Ceil.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Concat")) {
            return Op_union{ .concat = try operators.Concat.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Conv")) {
            return Op_union{ .conv = try operators.Conv.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Div")) {
            return Op_union{ .div = try operators.Div.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Elu")) {
            return Op_union{ .elu = try operators.Elu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Flatten")) {
            return Op_union{ .flatten = try operators.Flatten.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gather")) {
            return Op_union{ .gather = try operators.Gather.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gemm")) {
            return Op_union{ .gemm = try operators.Gemm.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Identity")) {
            return Op_union{ .identity = try operators.Identity.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "LeakyRelu")) {
            return Op_union{ .leakyRelu = try operators.LeakyRelu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "MatMul")) {
            return Op_union{ .matMul = try operators.MatMul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "MaxPool")) {
            return Op_union{ .maxPool = try operators.MaxPool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Mul")) {
            return Op_union{ .mul = try operators.Mul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Neg")) {
            return Op_union{ .neg = try operators.Neg.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "ReduceMean")) {
            return Op_union{ .reduceMean = try operators.ReduceMean.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Relu")) {
            return Op_union{ .relu = try operators.Relu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Reshape")) {
            return Op_union{ .reshape = try operators.Reshape.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Resize")) {
            return Op_union{ .resize = try operators.Resize.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Shape")) {
            return Op_union{ .shape = try operators.Shape.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Sigmoid")) {
            return Op_union{ .sigmoid = try operators.Sigmoid.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Slice")) {
            return Op_union{ .slice = try operators.Slice.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Softmax")) {
            return Op_union{ .softmax = try operators.Softmax.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Split")) {
            return Op_union{ .split = try operators.Split.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Sub")) {
            return Op_union{ .sub = try operators.Sub.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Tanh")) {
            return Op_union{ .tanh = try operators.Tanh.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Transpose")) {
            return Op_union{ .transpose = try operators.Transpose.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Unsqueeze")) {
            return Op_union{ .unsqueeze = try operators.Unsqueeze.init(nodeProto) };
        } else {
            std.debug.print("\n\nERROR: init() is not available for {s} operator!! \n\n", .{op_type});
            return Op_union{ .useless = try operators.Useless.init(nodeProto) };
        }
    }

    pub fn get_output_shape(self: Op_union) []usize {
        switch (self) {
            .add => |ptr| return ptr.get_output_shape(),
            .averagePool => |ptr| return ptr.get_output_shape(),
            .batchNormalization => |ptr| return ptr.get_output_shape(),
            .ceil => |ptr| return ptr.get_output_shape(),
            .concat => |ptr| return ptr.get_output_shape(),
            .conv => |ptr| return ptr.get_output_shape(),
            .div => |ptr| return ptr.get_output_shape(),
            .elu => |ptr| return ptr.get_output_shape(),
            .flatten => |ptr| return ptr.get_output_shape(),
            .gather => |ptr| return ptr.get_output_shape(),
            .gemm => |ptr| return ptr.get_output_shape(),
            .identity => |ptr| return ptr.get_output_shape(),
            .leakyRelu => |ptr| return ptr.get_output_shape(),
            .matMul => |ptr| return ptr.get_output_shape(),
            .maxPool => |ptr| return ptr.get_output_shape(),
            .mul => |ptr| return ptr.get_output_shape(),
            .neg => |ptr| return ptr.get_output_shape(),
            .reduceMean => |ptr| return ptr.get_output_shape(),
            .relu => |ptr| return ptr.get_output_shape(),
            .reshape => |ptr| return ptr.get_output_shape(),
            .resize => |ptr| return ptr.get_output_shape(),
            .shape => |ptr| return ptr.get_output_shape(),
            .sigmoid => |ptr| return ptr.get_output_shape(),
            .slice => |ptr| return ptr.get_output_shape(),
            .softmax => |ptr| return ptr.get_output_shape(),
            .split => |ptr| return ptr.get_output_shape(),
            .sub => |ptr| return ptr.get_output_shape(),
            .tanh => |ptr| return ptr.get_output_shape(),
            .transpose => |ptr| return ptr.get_output_shape(),
            .unsqueeze => |ptr| return ptr.get_output_shape(),
            else => {
                std.debug.print("\n\nERROR: get_output_shape() is not available!! \n\n", .{});
                return error.OpNotAvailable;
            },
        }
    }

    pub fn get_output_tensor(self: Op_union) void {
        switch (self) {
            .add => |ptr| ptr.get_output_tensor(),
            .averagePool => |ptr| ptr.get_output_tensor(),
            .batchNormalization => |ptr| ptr.get_output_tensor(),
            .ceil => |ptr| ptr.get_output_tensor(),
            .concat => |ptr| ptr.get_output_tensor(),
            .conv => |ptr| ptr.get_output_tensor(),
            .div => |ptr| ptr.get_output_tensor(),
            .elu => |ptr| ptr.get_output_tensor(),
            .flatten => |ptr| ptr.get_output_tensor(),
            .gather => |ptr| ptr.get_output_tensor(),
            .gemm => |ptr| ptr.get_output_tensor(),
            .identity => |ptr| ptr.get_output_tensor(),
            .leakyRelu => |ptr| ptr.get_output_tensor(),
            .matMul => |ptr| ptr.get_output_tensor(),
            .maxPool => |ptr| ptr.get_output_tensor(),
            .mul => |ptr| ptr.get_output_tensor(),
            .neg => |ptr| ptr.get_output_tensor(),
            .reduceMean => |ptr| ptr.get_output_tensor(),
            .relu => |ptr| ptr.get_output_tensor(),
            .reshape => |ptr| ptr.get_output_tensor(),
            .resize => |ptr| ptr.get_output_tensor(),
            .shape => |ptr| ptr.get_output_tensor(),
            .sigmoid => |ptr| ptr.get_output_tensor(),
            .slice => |ptr| ptr.get_output_tensor(),
            .softmax => |ptr| ptr.get_output_tensor(),
            .split => |ptr| ptr.get_output_tensor(),
            .sub => |ptr| ptr.get_output_tensor(),
            .tanh => |ptr| ptr.get_output_tensor(),
            .transpose => |ptr| ptr.get_output_tensor(),
            .unsqueeze => |ptr| ptr.get_output_tensor(),
            else => {
                std.debug.print("\n\nERROR: get_output_tensor() is not available!! \n\n", .{});
                return error.get_output_tensor_op_notAvailable;
            },
        }
    }

    pub fn write_op(self: Op_union, writer: std.fs.File.Writer) !void {
        switch (self) {
            .add => |ptr| try ptr.write_op(writer),
            .averagePool => |ptr| try ptr.write_op(writer),
            .batchNormalization => |ptr| try ptr.write_op(writer),
            .ceil => |ptr| try ptr.write_op(writer),
            .concat => |ptr| try ptr.write_op(writer),
            .conv => |ptr| try ptr.write_op(writer),
            .div => |ptr| try ptr.write_op(writer),
            .elu => |ptr| try ptr.write_op(writer),
            .flatten => |ptr| try ptr.write_op(writer),
            .gather => |ptr| try ptr.write_op(writer),
            .gemm => |ptr| try ptr.write_op(writer),
            .identity => |ptr| try ptr.write_op(writer),
            .leakyRelu => |ptr| try ptr.write_op(writer),
            .matMul => |ptr| try ptr.write_op(writer),
            .maxPool => |ptr| try ptr.write_op(writer),
            .mul => |ptr| try ptr.write_op(writer),
            .neg => |ptr| try ptr.write_op(writer),
            .reduceMean => |ptr| try ptr.write_op(writer),
            .relu => |ptr| try ptr.write_op(writer),
            .reshape => |ptr| try ptr.write_op(writer),
            .resize => |ptr| try ptr.write_op(writer),
            .shape => |ptr| try ptr.write_op(writer),
            .sigmoid => |ptr| try ptr.write_op(writer),
            // .slice => |ptr| try ptr.write_op(writer), //not present, see Slice node
            .softmax => |ptr| try ptr.write_op(writer),
            // .split => |ptr| try ptr.write_op(writer), //not working! error: .FAULT => unreachable,
            .sub => |ptr| try ptr.write_op(writer),
            .tanh => |ptr| try ptr.write_op(writer),
            .transpose => |ptr| try ptr.write_op(writer),
            .unsqueeze => |ptr| try ptr.write_op(writer),
            else => {
                std.debug.print("\n\nERROR: write_op() is not available!! \n\n", .{});
                return error.write_op_notAvailable;
            },
        }
    }

    pub fn print(self: Op_union) !void {
        switch (self) {
            .add => |ptr| ptr.print(),
            .averagePool => |ptr| ptr.print(),
            .batchNormalization => |ptr| ptr.print(),
            .ceil => |ptr| ptr.print(),
            .concat => |ptr| ptr.print(),
            .conv => |ptr| ptr.print(),
            .div => |ptr| ptr.print(),
            .elu => |ptr| ptr.print(),
            .flatten => |ptr| ptr.print(),
            .gather => |ptr| ptr.print(),
            .gemm => |ptr| ptr.print(),
            .identity => |ptr| ptr.print(),
            .leakyRelu => |ptr| ptr.print(),
            .matMul => |ptr| ptr.print(),
            .maxPool => |ptr| ptr.print(),
            .mul => |ptr| ptr.print(),
            .neg => |ptr| ptr.print(),
            .reduceMean => |ptr| ptr.print(),
            .relu => |ptr| ptr.print(),
            .reshape => |ptr| ptr.print(),
            .resize => |ptr| ptr.print(),
            .shape => |ptr| ptr.print(),
            .sigmoid => |ptr| ptr.print(),
            .slice => |ptr| ptr.print(),
            .softmax => |ptr| ptr.print(),
            .split => |ptr| ptr.print(),
            .sub => |ptr| ptr.print(),
            .tanh => |ptr| ptr.print(),
            .transpose => |ptr| ptr.print(),
            .unsqueeze => |ptr| ptr.print(),
            else => {
                std.debug.print("\n\nERROR: print() is not available!! \n\n", .{});
                return error.print_notAvailable;
            },
        }
    }
};
