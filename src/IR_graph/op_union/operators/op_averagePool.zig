const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");

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

const tensorMath = zant.core.tensor.math_standard;

const utils = @import("codegen").utils;

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

pub const AveragePool = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    //attributes:
    auto_pad: []const u8, // default = "NOTSET",
    ceil_mode: i64, // default = 0;
    count_include_pad: i64, // default = 0;
    dilations: ?[]i64, // default = null;
    kernel_shape: ?[]i64, // default = null; but mandatory
    pads: ?[]i64, // default = null;
    strides: ?[]i64, // default = null;

    pub fn init(nodeProto: *NodeProto) !AveragePool {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var ceil_mode: i64 = 0;
        var count_include_pad: i64 = 0;
        var dilations: ?[]i64 = null;
        var kernel_shape: ?[]i64 = null; //mandatory
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.MaxPoolAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
                if (attr.type == onnx.AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "count_include_pad")) |_| {
                if (attr.type == onnx.AttributeType.INT) count_include_pad = attr.i else return error.AveragePoolCountIncludePadNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.MaxPoolDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.MaxPoolKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.MaxPoolPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.MaxPoolStridesNotINTS;
            }
        }

        return AveragePool{
            .input_X = input_X,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .ceil_mode = ceil_mode,
            .count_include_pad = count_include_pad,
            .dilations = dilations,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: AveragePool) []usize { // TODO
        return self.output_Y.get_shape();
    }

    pub fn get_output_tensor(self: AveragePool) *TensorZant {
        return self.output_Y;
    }

    pub fn write_op(self: AveragePool, writer: std.fs.File.Writer) !void {

        //input_X string equivalent
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        // kernel_shape string equivalent
        var kernel_shape_string: []const u8 = undefined;
        if (self.kernel_shape != null) {
            kernel_shape_string = try utils.i64SliceToUsizeArrayString(self.kernel_shape.?);
            // defer allocator.free(kernel_shape_string);
        } else {
            return error.Kernel_shapeNotFound;
        }

        // strides string equivalent
        var strides_string: []const u8 = undefined;
        if (self.strides != null) {
            strides_string = try utils.i64SliceToUsizeArrayString(self.strides.?);
            // defer allocator.free(strides_string);
        } else {
            return error.StridesNotFound;
        }

        // Crea stringa per dilations
        var dilations_string: []const u8 = undefined;
        if (self.dilations != null) {
            dilations_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
            defer allocator.free(dilations_string);
        } else {
            dilations_string = try utils.i64SliceToUsizeArrayString(&[_]i64{ 1, 1, 1, 1 }); // TODO: Hardcoded in 4D, not the most elegant solutionk
        }

        // Crea stringa per pads
        var pads_string: []const u8 = undefined;
        if (self.pads != null) {
            pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
            // defer allocator.free(pads_string);
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
            try utils.getSanitizedName(self.output_Y.name), // Output
            kernel_shape_string, // kernel_shape
            strides_string, // strides
            dilations_string, // dilations
            pads_string, // pads
            self.auto_pad, // auto_pad
            if (self.count_include_pad == 1) "true" else "false", // count_include_pad
        });
    }

    pub fn compute_averagePool_output_shape(self: AveragePool) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_onnx_averagepool_output_shape(
            self.input_X.shape,
            try utils.i64SliceToUsizeSlice(self.kernel_shape.?),
            try utils.i64SliceToUsizeSlice(self.strides.?),
            try utils.i64SliceToUsizeSlice(self.dilations.?),
            try utils.i64SliceToUsizeSlice(self.pads.?),
            "NOTSET",
            try utils.i64SliceToUsizeSlice(self.ceil_mode),
        );
        return output_shape;
    }

    pub fn print(self: AveragePool) void { // TODO
        std.debug.print("\n AveragePool:\n {any}", .{self});
    }
};
