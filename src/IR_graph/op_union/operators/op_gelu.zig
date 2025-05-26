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
const IR_utils = @import("../../utils.zig"); //this is IR utils
const utils = @import("codegen").utils;

pub const Gelu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    approximate: []const u8,

    pub fn init(nodeProto: *NodeProto) !Gelu {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var approximate: []const u8 = "none";

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "approximate")) {
                if (attr.type != onnx.AttributeType.STRING) {
                    return error.InvalidAttributeType;
                }
                approximate = attr.s;
            }
        }
        return Gelu{
            .input_X = input_X,
            .output_Y = output_Y,
            .approximate = approximate,
        };
    }

    pub fn get_output_shape(self: Gelu) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: Gelu) *TensorZant {
        return self.output_Y;
    }

    pub fn compute_output_shape(self: Gelu) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_gelu_output_shape(self.input_X.ptr.?.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn write_op(self: Gelu, writer: std.fs.File.Writer) !void {
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\    tensMath.gelu_lean(T, {s}, "{s}", &tensor_{s})
        , .{
            input_tensor_string,
            self.approximate,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn print(self: Gelu) void {
        std.debug.print("\n Gelu:\n {any}", .{self});
    }
};
