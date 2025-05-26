const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkgAllocator = zant.utils.allocator.allocator;

pub fn get_onehot_output_shape(indices_shape: []const usize, depth: i64, axis: i64) ![]usize {
    // Normalizza axis
    const rank = @as(i64, @intCast(indices_shape.len));
    const normalized_axis = if (axis < 0) axis + rank + 1 else axis;
    if (normalized_axis < 0 or normalized_axis > rank) {
        return TensorMathError.InvalidAxes;
    }

    // Crea la forma dell'output: rank(indices) + 1
    var output_shape = try pkgAllocator.alloc(usize, indices_shape.len + 1);
    errdefer pkgAllocator.free(output_shape);

    // Copia indices_shape e inserisci depth nella posizione axis
    for (indices_shape, 0..) |dim, i| {
        if (i < normalized_axis) {
            output_shape[i] = dim;
        } else {
            output_shape[i + 1] = dim;
        }
    }
    output_shape[@intCast(normalized_axis)] = @intCast(depth);

    return output_shape;
}

pub fn onehot(comptime T: type, indices: *const Tensor(i64), depth: *const Tensor(i64), values: *const Tensor(T), axis: i64) !Tensor(T) {
    // Controlla i tipi
    const allowed_types = [_]type{
        f32,  f64,
        bool, i8,
        i16,  i32,
        i64,  u8,
        u16,  u32,
        u64,
    };

    var valid_type = false;
    inline for (allowed_types) |Allowed| {
        if (T == Allowed) {
            valid_type = true;
            break;
        }
    }
    if (!valid_type) {
        return TensorMathError.InvalidDataType;
    }

    // Controlla depth (scalare o rango 1 con un elemento)
    if (depth.shape.len > 1 or (depth.shape.len == 1 and depth.shape[0] != 1)) {
        return TensorMathError.InvalidDepthShape;
    }
    const depth_val = depth.data[0];
    if (depth_val <= 0) {
        return TensorMathError.InvalidDepthValue;
    }

    // Controlla values (rango 1 con 2 elementi)
    if (values.shape.len != 1 or values.shape[0] != 2) {
        return TensorMathError.InvalidValuesShape;
    }

    // Calcola la forma dell'output
    const output_shape = try get_onehot_output_shape(indices.shape, depth_val, axis);
    var output = try Tensor(T).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();
    defer pkgAllocator.free(output_shape);

    // Chiama la versione lean
    try onehot_lean(T, indices, depth_val, values, axis, &output);

    return output;
}

pub fn onehot_lean(comptime T: type, indices: *const Tensor(i64), depth: i64, values: *const Tensor(T), axis: i64, output: *Tensor(T)) !void {
    // Inizializza l'output con off_value
    for (output.data) |*val| {
        val.* = values.data[0]; // off_value
    }

    // Normalizza axis
    const rank = @as(i64, @intCast(indices.shape.len));
    const normalized_axis = if (axis < 0) axis + rank + 1 else axis;

    // Itera sugli indici
    const total_elements = blk: {
        var prod: usize = 1;
        for (indices.shape) |dim| prod *= dim;
        break :blk prod;
    };

    for (0..total_elements) |flat_idx| {
        const index_val = indices.data[flat_idx];

        // Ignora indici fuori range
        if (index_val < -depth or index_val >= depth) {
            continue;
        }
        const index = if (index_val < 0) index_val + depth else index_val;

        // Calcola le coordinate multi-dimensionali
        var input_coords = try pkgAllocator.alloc(usize, indices.shape.len);
        defer pkgAllocator.free(input_coords);
        var temp_idx = flat_idx;
        for (indices.shape, 0..) |dim, i| {
            input_coords[indices.shape.len - 1 - i] = temp_idx % dim;
            temp_idx /= dim;
        }

        // Calcola le coordinate dell'output
        var output_coords = try pkgAllocator.alloc(usize, indices.shape.len + 1);
        defer pkgAllocator.free(output_coords);
        for (input_coords, 0..) |coord, i| {
            if (i < normalized_axis) {
                output_coords[i] = coord;
            } else {
                output_coords[i + 1] = coord;
            }
        }

        output_coords[@intCast(normalized_axis)] = @intCast(index);

        // Imposta on_value
        const output_idx = try output.get_flat_index(output_coords);
        output.data[output_idx] = values.data[1]; // on_value
    }
}
