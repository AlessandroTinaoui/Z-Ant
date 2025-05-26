const std = @import("std");
const zant = @import("zant");
const tensorMath = zant.core.tensor.math_standard;

const allocator = zant.utils.allocator.allocator;

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

pub const Conv = struct {
    input_X: *TensorZant,
    input_W: *TensorZant,
    input_B: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init(nodeProto: *NodeProto) !Conv {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_W = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_W_notFound;
        const input_B = if (nodeProto.input.len > 2) if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_B_notFound else null;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null; //mandatory

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
                if (attr.type == onnx.AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
            }
        }

        return Conv{
            .input_X = input_X,
            .input_W = input_W,
            .input_B = input_B,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: Conv) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: Conv) *TensorZant {
        return self.output_Y;
    }

    pub fn write_op(self: Conv, writer: std.fs.File.Writer) !void {
        //----create tensor_X_string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        //----create tensor_W_string
        var tensor_W_string: []u8 = undefined;
        defer allocator.free(tensor_W_string);
        if (self.input_W.tc == TensorCategory.INITIALIZER) {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_W.name),
                ")",
            });
        } else {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_W.name), ")" });
        }

        //----create ?bias string
        var bias_string: []u8 = undefined;
        // Bias Tensor B is optional! verify the presence
        if (self.input_B) |input_B| {
            const B_name = try utils.getSanitizedName(input_B.name);
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //----create stride string (mandatory)
        // TODO: implement default stride, see docs above
        if (self.strides == null) return error.StrideNotFound;
        const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(self.strides.?);

        //----create ?pads string
        var pads_string: []const u8 = "null";
        if (self.pads != null) {
            if (self.pads.?.len > 0) { // Check if the slice is actually non-empty
                pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
                // Assuming no allocation needed to be freed, following write_conv
            } else {
                pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
            }
        } // else pads_string remains "null"

        //----create ?dilatations string
        var dilat_string: []const u8 = "null";
        if (self.dilations != null) {
            if (self.dilations.?.len > 0) {
                dilat_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
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
            try utils.getSanitizedName(self.output_Y.name), //Output
            bias_string, //Bias
            stride_string, //Strides
            pads_string, //Pads
            dilat_string, //Dilatations
            self.group, //Group
            self.auto_pad, //auto_pad
        });
    }

    pub fn compute_output_shape(self: Conv) []usize {
        var output_shape: []usize = undefined;
        const input_shape = self.input_X.get_shape();
        const kernel_shape = self.input_W.get_shape();
        const stride = self.strides;
        const pads = self.pads;
        const dilations = self.dilations;
        const auto_pad = self.auto_pad;
        output_shape = try tensorMath.get_convolution_output_shape(
            input_shape,
            kernel_shape,
            try utils.i64SliceToUsizeSlice(stride.?),
            if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
            try utils.i64SliceToUsizeSlice(dilations.?),
            auto_pad,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Conv) void { //TODO
        std.debug.print("\n CONV:\n {any}", .{self});
    }
};
