const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
pub const AnyTensor = zant.core.tensor.AnyTensor;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const ValueInfoProto = onnx.ValueInfoProto;

const allocator = std.heap.page_allocator;

// --- zant ---
const TensorType = @import("tensorZant.zig").TensorType;
pub const tensorZant_lib = @import("tensorZant.zig");
pub const TensorZant = tensorZant_lib.TensorZant;
pub const TensorCategory = tensorZant_lib.TensorCategory;

pub fn getValueInfoTensorFromGraphInfo(name: []const u8, protoGraph: *GraphProto) ?*ValueInfoProto {
    for (protoGraph.value_info) |vi| {
        if (std.mem.eql(u8, vi.name.?, name)) {
            return vi;
        }
    }
    return null;
}

pub fn getShapeFromModelInfo(model: *ModelProto) ?[]i64 {
    for (model.value_info) |vi| {
        return getTensorShapeFromValueInfo(vi);
    }
}

pub fn getTensorShapeFromValueInfo(vi: *ValueInfoProto) ?[]i64 {
    return vi.type.?.tensor_type.?.shape.?.shape;
}

pub fn getAnyTensorType(anyTensor: AnyTensor) TensorType {
    return switch (anyTensor) {
        .i64 => TensorType.i64,
        .f64 => TensorType.i64,
        .u64 => TensorType.i64,
        .f32 => TensorType.i64,
        .i32 => TensorType.i64,
        .u32 => TensorType.i64,
        .f16 => TensorType.i64,
        .i16 => TensorType.i64,
        .u16 => TensorType.i64,
        .i8 => TensorType.i64,
        .u8 => TensorType.i64,
    };
}

//Returns the sanitized tensor's name, removes all non alphanumeric chars
pub inline fn getSanitizedName(name: []const u8) ![]const u8 {
    var sanitized = try allocator.alloc(u8, name.len);

    for (name, 0..) |char, i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
            std.ascii.toLower(char)
        else
            '_';
    }

    //std.log.debug("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

// ----------------- DATA TYPE management -------------

pub inline fn i64SliceToUsizeSlice(input: []const i64) ![]usize {
    var output = try allocator.alloc(usize, input.len);

    const maxUsize = std.math.maxInt(usize);

    for (input, 0..) |value, index| {
        if (value < 0) {
            return error.NegativeValue;
        }
        if (value > maxUsize) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

pub fn usizeSliceToI64Slice(input: []usize) ![]const i64 {
    var output = try allocator.alloc(i64, input.len);

    for (input, 0..) |value, index| {
        if (value > std.math.maxInt(i64)) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

// ----------------- STRUCT TYPE management -------------

pub fn protoTensor2AnyTensor(proto: *TensorProto) !AnyTensor {
    // Allocate shape array
    var shape = try allocator.alloc(usize, proto.dims.len);
    for (proto.dims, 0..) |dim, i| {
        if (dim < 0) {
            return error.NegativeDimension;
        }
        shape[i] = @intCast(dim);
    }
    defer allocator.free(shape);

    if (proto.float_data) |float_data| {
        const tensor = try allocator.create(Tensor(f32));
        tensor.* = try Tensor(f32).fromArray(&allocator, float_data, shape);
        return AnyTensor{ .f32 = tensor };
    } else if (proto.int32_data) |int32_data| {
        const tensor = try allocator.create(Tensor(i32));
        tensor.* = try Tensor(i32).fromArray(&allocator, int32_data, shape);
        return AnyTensor{ .i32 = tensor };
    } else if (proto.int64_data) |int64_data| {
        const tensor = try allocator.create(Tensor(i64));
        tensor.* = try Tensor(i64).fromArray(&allocator, int64_data, shape);
        return AnyTensor{ .i64 = tensor };
    } else if (proto.double_data) |double_data| {
        const tensor = try allocator.create(Tensor(f64));
        tensor.* = try Tensor(f64).fromArray(&allocator, double_data, shape);
        return AnyTensor{ .f64 = tensor };
    } else if (proto.uint64_data) |uint64_data| {
        const tensor = try allocator.create(Tensor(u64));
        tensor.* = try Tensor(u64).fromArray(&allocator, uint64_data, shape);
        return AnyTensor{ .u64 = tensor };
    } else if (proto.uint16_data) |uint16_data| {
        const tensor = try allocator.create(Tensor(u16));
        tensor.* = try Tensor(u16).fromArray(&allocator, uint16_data, shape);
        return AnyTensor{ .u16 = tensor };
    } else {
        return error.UnsupportedDataType;
    }
}

pub fn broadcastShapes(general_allocator: std.mem.Allocator, shape1: []usize, shape2: []usize) ![]usize {
    const max_len = std.math.max(shape1.len, shape2.len);

    var output = try general_allocator.alloc(usize, max_len);
    errdefer general_allocator.free(output);

    // Compute padding
    const pad1 = max_len - shape1.len;
    const pad2 = max_len - shape2.len;

    for (0..max_len) |i| {
        // Get the corresponding dimensions, using 1 for padding if necessary
        const dim1 = if (i < pad1) @as(usize, 1) else shape1[i - pad1];
        const dim2 = if (i < pad2) @as(usize, 1) else shape2[i - pad2];

        // Check compatibility and compute the output dimension
        if (dim1 == dim2) {
            output[i] = dim1;
        } else if (dim1 == 1) {
            output[i] = dim2;
        } else if (dim2 == 1) {
            output[i] = dim1;
        } else {
            allocator.free(output);
            return error.IncompatibleShapes;
        }
    }
    return output;
}

pub fn getInitializers(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var initializers = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.INITIALIZER) {
            try initializers.append(entry.value_ptr.*);
        }
    }
    return initializers.toOwnedSlice();
}
