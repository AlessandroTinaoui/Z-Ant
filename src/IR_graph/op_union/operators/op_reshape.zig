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

// https://onnx.ai/onnx/operators/onnx__Reshape.html#l-onnx-doc-reshape
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
//      - shape (heterogeneous) - tensor(int64): Specified shape for output
// OUTPUTS:
//      - reshaped (heterogeneous) - T: Reshaped data.
// ATTRIBUTES:
//      - allowzero - INT (default is '0'): If '1', the shape can contain zero. TODO
//      - shape (ints): Alternative way to provide shape (used if input 'shape' is not provided). -> shape_attribute

pub const Reshape = struct {
    data: *TensorZant,
    shape: *TensorZant,
    reshaped: *TensorZant,
    allowzer0: bool,
    shape_attribute: ?[]const i64,

    pub fn init(nodeProto: *NodeProto) !Reshape {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const shape = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.shape_notFound;
        const reshaped = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var allowzer0: bool = false;
        var shape_attribute: ?[]const i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "allowzero")) {
                if (attr.type == onnx.AttributeType.INT) allowzer0 = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "shape")) {
                if (attr.type == onnx.AttributeType.INTS) shape_attribute = attr.ints;
            }
        }

        return Reshape{
            .data = data,
            .shape = shape,
            .reshaped = reshaped,
            .allowzer0 = allowzer0,
            .shape_attribute = shape_attribute,
        };
    }

    pub fn get_output_shape(self: Reshape) []usize {
        return self.reshaped.getShape();
    }

    pub fn get_output_tensor(self: Reshape) *TensorZant {
        return self.reshaped;
    }

    pub fn write_op(self: Reshape, writer: std.fs.File.Writer) !void {
        // Input tensor string creation
        const sanitized_input_name = try utils.getSanitizedName(self.data.name);
        const input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            if (self.data.tc == TensorCategory.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_input_name,
        });
        defer allocator.free(input_string);

        // Shape slice generation logic
        var shape_slice_code = std.ArrayList(u8).init(allocator);
        defer shape_slice_code.deinit();
        const output_sanitized_name = try utils.getSanitizedName(self.reshaped.name);
        var shape_from_attr = false; // Track source of shape

        if (self.shape_attribute) |attr_shape| {
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

            const shape_input_tensor = self.shape;
            const sanitized_shape_name = try utils.getSanitizedName(shape_input_tensor.name);
            const shape_tensor_name = try std.mem.concat(allocator, u8, &[_][]const u8{
                if (self.shape.tc == TensorCategory.INITIALIZER) "param_lib." else "",
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

        const input_type_string = self.data.ty.toString();

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
            \\    // Reshape Operation 
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
            shape_slice_code.items, // Arg 1 for shape code
            input_type_string, // Arg 2 for input type
            input_string, // Arg 3 for input tensor
            shape_slice_arg, // Arg 4 for shape slice
            if (self.allowzer0) "true" else "false", // Arg 5 for allowzero
            output_tensor_arg, // Arg 6 for output tensor
        });
    }

    pub fn get_reshape_output_shape(self: Reshape) ![]usize {
        var output_shape: []usize = undefined;
        const new_shape_spec = try allocator.alloc(isize, self.shape.shape.len);
        defer allocator.free(new_shape_spec);
        for (self.shape.shape, 0..) |val, i| {
            new_shape_spec[i] = @as(isize, @intCast(val));
        }
        output_shape = try tensorMath.get_reshape_output_shape(
            self.data.shape,
            new_shape_spec,
            false,
        );
        self.reshaped.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Reshape) void {
        std.debug.print("\n Reshape:\n {any}", .{self});
    }
};
