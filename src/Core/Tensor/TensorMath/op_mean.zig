const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;
const TensMath = @import("tensor_math_standard.zig");
const op_mat_mul = @import("op_mat_mul.zig");

//l'operazione mean in onnx calcola la media elemento per elemento di una
//lista variabile di tensori in input, supportando da 1 a un numero teoricamente
//illimitato di tensori, e produce un unico tensore in output con lo stesso tipo
//di dati degli input (ad esempio float o double). Lo fa allineando i tensori attraverso
//il broadcasting multidirezionale in stile NumPy, che permette di gestire forme diverse
//espandendo virtualmente le dimensioni più piccole (quando compatibili, cioè uguali o pari a 1)
//per matchare quelle più grandi, senza modificare fisicamente i dati. Per ogni posizione nell'output,
//somma i valori corrispondenti degli input (mappati tramite broadcasting) e divide per il numero
//di tensori, garantendo un risultato che riflette la media aritmetica lungo tutti gli input,
//con una forma dedotta come il massimo delle dimensioni compatibili."

pub fn mean_standard(comptime T: anytype, inputs: []*Tensor(T)) !Tensor(T) {
    if (inputs.len == 0) {
        return TensorMathError.EmptyTensorList;
    }

    const type_info = @typeInfo(T);
    if (type_info != .float or (T != f32 and T != f64 and T != f16)) {
        return TensorMathError.InvalidDataType;
    }
    for (inputs) |tensor| {
        if (@TypeOf(tensor.data) != @TypeOf(inputs[0].data)) {
            return TensorMathError.MismatchedDataTypes;
        }
    }

    var input_shapes = try pkg_allocator.alloc([]usize, inputs.len);
    defer pkg_allocator.free(input_shapes);
    for (inputs, 0..) |tensor, i| {
        input_shapes[i] = tensor.shape;
    }

    const output_shape = try get_mean_output_shape(input_shapes);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try mean_lean(T, inputs, &output);

    return output;
}

pub inline fn mean_lean(comptime T: anytype, inputs: []*Tensor(T), output: *Tensor(T)) !void {
    // Itera su ogni posizione nell'output
    for (0..output.size) |idx| {
        // Converte l'indice lineare in coordinate multidimensionali
        const coords = indexToCoords(idx, output.shape) catch unreachable; // Errore gestito in mean_standard
        defer pkg_allocator.free(coords);

        // Calcola la somma dei valori degli input per questa posizione
        var sum: T = 0;
        for (inputs) |tensor| {
            const input_idx = getBroadcastIndex(coords, tensor.shape, output.shape);
            sum += tensor.data[input_idx];
        }

        // Calcola la media e scrive nel tensore di output
        output.data[idx] = sum / @as(T, @floatFromInt(inputs.len));
    }
}

pub fn get_mean_output_shape(inputs: []const []usize) ![]usize {
    if (inputs.len == 0) {
        return TensorMathError.EmptyTensorList;
    }

    var max_rank: usize = 0;
    for (inputs) |shape| {
        if (shape.len > max_rank) {
            max_rank = shape.len;
        }
    }

    var output_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(output_shape);
    @memset(output_shape, 1);

    for (inputs) |shape| {
        const rank_diff = max_rank - shape.len;
        for (shape, 0..) |dim, i| {
            const out_idx = rank_diff + i;
            const current_out_dim = output_shape[out_idx];

            if (dim != current_out_dim) {
                if (current_out_dim == 1) {
                    output_shape[out_idx] = dim;
                } else if (dim != 1) {
                    pkg_allocator.free(output_shape);
                    return TensorMathError.IncompatibleBroadcastShapes;
                }
            }
        }
    }

    return output_shape;
}

/// Calcola l'indice lineare in un tensore di input dato una posizione nell'output, applicando il broadcasting.
/// - `output_coords`: Coordinate multidimensionali nel tensore di output.
/// - `input_shape`: Forma del tensore di input.
/// - `output_shape`: Forma del tensore di output (per allineamento).
/// Restituisce l'indice lineare nel tensore di input.
pub fn getBroadcastIndex(output_coords: []const usize, input_shape: []const usize, output_shape: []const usize) usize {
    std.debug.assert(output_coords.len == output_shape.len); // Coordinate devono matchare l'output
    std.debug.assert(input_shape.len <= output_shape.len); // L'input può avere meno dimensioni

    // Calcola l'offset per allineare le forme da destra
    const rank_diff = output_shape.len - input_shape.len;
    var input_index: usize = 0;

    // Itera sulle dimensioni dell'output
    for (output_shape, output_coords, 0..) |_, coord, i| {
        // Se siamo oltre le dimensioni dell'input, non contribuiscono all'indice
        if (i < rank_diff) continue;

        // Indice corrispondente nella forma dell'input
        const input_dim_idx = i - rank_diff;
        const in_dim = input_shape[input_dim_idx];

        // Broadcasting: se la dimensione dell'input è 1, usa 0, altrimenti usa la coordinata
        const effective_coord = if (in_dim == 1) 0 else coord;
        std.debug.assert(effective_coord < in_dim); // Verifica che la coordinata sia valida

        // Calcola il contributo all'indice lineare
        var stride: usize = 1;
        for (input_shape[input_dim_idx + 1 ..]) |dim| {
            stride *= dim;
        }
        input_index += effective_coord * stride;
    }

    return input_index;
}

/// Converte un indice lineare in coordinate multidimensionali basate sulla forma del tensore.
/// - `index`: Indice lineare (0-based).
/// - `shape`: Forma del tensore.
/// Restituisce un array di coordinate (da liberare dal chiamante).
pub fn indexToCoords(index: usize, shape: []const usize) ![]usize {
    if (index >= product(shape)) {
        return error.IndexOutOfBounds;
    }

    var coords = try pkg_allocator.alloc(usize, shape.len);
    errdefer pkg_allocator.free(coords);

    var remaining = index;
    for (shape, 0..) |_, i| {
        if (i == shape.len - 1) {
            coords[i] = remaining; // Ultima dimensione: resto diretto
        } else {
            const stride = product(shape[i + 1 ..]); // Prodotto delle dimensioni successive
            coords[i] = remaining / stride;
            remaining = remaining % stride;
        }
    }

    return coords;
}

/// Converte coordinate multidimensionali in un indice lineare basato sulla forma del tensore.
/// - `coords`: Coordinate multidimensionali.
/// - `shape`: Forma del tensore.
/// Restituisce l'indice lineare corrispondente.
pub fn coordsToIndex(coords: []const usize, shape: []const usize) usize {
    std.debug.assert(coords.len == shape.len); // Coordinate devono matchare la forma

    var index: usize = 0;
    for (shape, coords, 0..) |dim, coord, i| {
        std.debug.assert(coord < dim); // Verifica che la coordinata sia valida
        const stride = if (i == shape.len - 1) 1 else product(shape[i + 1 ..]);
        index += coord * stride;
    }

    return index;
}

/// Calcola il prodotto di un array di usize.
fn product(slice: []const usize) usize {
    var result: usize = 1;
    for (slice) |val| {
        result *= val;
    }
    return result;
}
