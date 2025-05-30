const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const assert = std.debug.assert;

const ArchitectureError = zant.utils.error_handler.ArchitectureError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const Uops = zant.uops;

const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

// Optimize for L1 cache size (typically 32KB)
const BLOCK_SIZE_M: usize = 32;
const BLOCK_SIZE_N: usize = 32;
const BLOCK_SIZE_K: usize = 32;

// Use largest available SIMD width
// const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const UNROLL_FACTOR: usize = 10;

// TODO: add support for matrix multiplication for matrix distribuited in multi-batch/multi-channel tensors (for example of shape {2, 3, 5, 5}), now supports only tensors with shape {1, 1, N, M}
/// Performs classic matrix multiplication on given tensors using the least 2 dimensions
pub inline fn mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T)) !Tensor(T) {
    // std.log.debug("\nStarting matrix multiplication validation...\n", .{});

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        // std.log.debug("Error: Input tensors have different dimensions. A: {}, B: {}\n", .{ A.shape.len, B.shape.len });
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // Special handling for 1D tensors (vectors)
    if (dim_num == 1) {
        // For 1D vectors, we treat it as a dot product
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        // Create a scalar output (1x1 tensor)
        const allocator = pkg_allocator;
        var out_shape = try allocator.alloc(usize, 1);
        defer allocator.free(out_shape);
        out_shape[0] = 1;

        var Y = try Tensor(T).fromShape(&allocator, out_shape);
        errdefer Y.deinit();

        @memset(Y.data, 0);
        try lean_mat_mul(T, A, B, &Y);

        return Y;
    }

    // For tensors with >= 2 dimensions

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        // std.log.debug("Error: Incompatible matrix dimensions for multiplication. A[{}]={}, B[{}]={}\n", .{ dim_num - 1, A.shape[dim_num - 1], dim_num - 2, B.shape[dim_num - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    // Create output tensor
    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Check if the input tensors are empty
    if (M * N == 0 or K == 0) {
        // std.log.debug("Error: Empty input tensors. M={}, N={}, K={}\n", .{ M, N, K });
        return TensorMathError.InputTensorsWrongShape;
    }

    // std.log.debug("Validation passed, proceeding with multiplication\n", .{});

    // Setup output tensor shape
    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, dim_num);
    defer allocator.free(out_shape);
    errdefer allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor
    var Y = try Tensor(T).fromShape(&allocator, out_shape);
    errdefer Y.deinit();

    // std.log.debug("Output tensor shape: ", .{});
    // for (Y.shape) |dim| std.log.debug("{} ", .{dim});
    // std.log.debug("\n", .{});

    @memset(Y.data, 0);

    try lean_mat_mul(T, A, B, &Y);

    return Y;
}

pub inline fn lean_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), Y: *Tensor(T)) !void {
    const DEFAULT_VECTOR_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    const dim_num = A.shape.len;

    // Handle 1D tensors as special case
    if (dim_num == 1) {
        if (B.shape.len != 1) {
            return TensorMathError.InputTensorDifferentShape;
        }

        // For 1D vectors, we treat them as a dot product
        // A is a 1D vector (1xK), B is a 1D vector (Kx1), Y is a scalar (1x1)
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        if (Y.shape.len != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }
        if (Y.shape[0] != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }

        var sum: T = 0;
        for (0..K) |k| {
            sum += A.data[k] * B.data[k];
        }

        Y.data[0] = sum;
        return;
    }

    // Regular matrix multiplication for dim_num >= 2
    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Add dimension validation
    if (M >= std.math.maxInt(usize) / 2 or
        N >= std.math.maxInt(usize) / 2 or
        K >= std.math.maxInt(usize) / 2)
    {
        return TensorMathError.InputTensorsWrongShape;
    }

    // Add shape validation
    if (B.shape[dim_num - 2] != K) {
        return TensorMathError.InputTensorsWrongShape;
    }

    // Validate output tensor shape
    if (Y.shape[dim_num - 2] != M or Y.shape[dim_num - 1] != N) {
        return TensorMathError.OutputTensorWrongShape;
    }

    // Debug prints only when needed
    if (false) {
        std.log.debug("\nMatrix multiplication dimensions: M={}, N={}, K={}\n", .{ M, N, K });
        std.log.debug("Input tensor A shape: ", .{});
        for (A.shape) |dim| std.log.debug("{} ", .{dim});
        std.log.debug("\nInput tensor B shape: ", .{});
        for (B.shape) |dim| std.log.debug("{} ", .{dim});
        std.log.debug("\nOutput tensor Y shape: ", .{});
        for (Y.shape) |dim| std.log.debug("{} ", .{dim});
        std.log.debug("\n", .{});
    }

    // SIMD vector type
    const Vec = @Vector(DEFAULT_VECTOR_WIDTH, T);
    const VecOut = @Vector(DEFAULT_VECTOR_WIDTH, T);

    // Get pointers for faster access
    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const Y_ptr = Y.data.ptr;

    // Main matrix multiplication loop with SIMD
    var i: usize = 0;
    while (i < M) : (i += 1) {
        // if (i % 100 == 0) std.log.debug("Processing row {}/{}\n", .{ i, M });
        const row_offset = i * K;
        const out_offset = i * N;

        var j: usize = 0;
        while (j + DEFAULT_VECTOR_WIDTH <= N) : (j += DEFAULT_VECTOR_WIDTH) {
            var sum_vec: VecOut = @splat(0);
            const out_idx = out_offset + j;

            // Inner product with SIMD
            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_val = A_ptr[row_offset + k];
                const b_offset = k * N + j;

                // Load B values directly into vector
                var b_vec: Vec = undefined;
                comptime var v: usize = 0;
                inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                    b_vec[v] = B_ptr[b_offset + v];
                }

                // Convert and multiply
                const a_vec: VecOut = @splat(@as(T, a_val));
                const b_vec_out: VecOut = @as(VecOut, b_vec);
                sum_vec += a_vec * b_vec_out;
            }

            // Store result
            comptime var v: usize = 0;
            inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                Y_ptr[out_idx + v] = sum_vec[v];
            }
        }

        // Handle remaining columns
        while (j < N) : (j += 1) {
            var sum: T = 0;
            const out_idx = out_offset + j;

            var k: usize = 0;
            while (k < K) : (k += 1) {
                sum += @as(T, A_ptr[row_offset + k]) *
                    @as(T, B_ptr[k * N + j]);
            }
            Y_ptr[out_idx] = sum;
        }
    }

    // std.log.debug("Matrix multiplication completed\n", .{});
}

const CACHE_BLOCK_SIZE_BYTES: usize = std.atomic.cache_line;

pub inline fn blocked_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T)) !Tensor(T) {
    // std.log.debug("\nStarting matrix multiplication validation...\n", .{});

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        // std.log.debug("Error: Input tensors have different dimensions. A: {}, B: {}\n", .{ A.shape.len, B.shape.len });
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // Special handling for 1D tensors (vectors)
    if (dim_num == 1) {
        // For 1D vectors, we treat it as a dot product
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        // Create a scalar output (1x1 tensor)
        const allocator = pkg_allocator;
        var out_shape = try allocator.alloc(usize, 1);
        defer allocator.free(out_shape);
        out_shape[0] = 1;

        var Y = try Tensor(T).fromShape(&allocator, out_shape);
        errdefer Y.deinit();

        @memset(Y.data, 0);

        // Since this is just a dot product, we'll calculate it directly
        var sum: T = 0;
        for (0..K) |k| {
            sum += A.data[k] * B.data[k];
        }

        Y.data[0] = sum;
        return Y;
    }

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        // std.log.debug("Error: Incompatible matrix dimensions for multiplication. A[{}]={}, B[{}]={}\n", .{ dim_num - 1, A.shape[dim_num - 1], dim_num - 2, B.shape[dim_num - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    // The input tensors must have at least 2 dimensions
    if (dim_num < 2) {
        // std.log.debug("Error: Input tensors must have at least 2 dimensions. Got: {}\n", .{dim_num});
        return TensorMathError.InputTensorsWrongShape;
    }

    // Create output tensor

    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Check if the input tensors are empty
    if (M * N == 0 or K == 0) {
        // std.log.debug("Error: Empty input tensors. M={}, N={}, K={}\n", .{ M, N, K });
        return TensorMathError.InputTensorsWrongShape;
    }

    // std.log.debug("Validation passed, proceeding with multiplication\n", .{});

    // Setup output tensor shape

    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, dim_num);
    defer allocator.free(out_shape);
    errdefer allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor

    var Y = try Tensor(T).fromShape(&allocator, out_shape);
    errdefer Y.deinit();

    // std.log.debug("Output tensor shape: ", .{});
    // for (Y.shape) |dim| std.log.debug("{} ", .{dim});
    // std.log.debug("\n", .{});

    @memset(Y.data, 0);

    try lean_blocked_mat_mul(T, A, B, &Y);

    return Y;
}

//Loosely inspired from https://coffeebeforearch.github.io/2020/06/23/mmul.html
//Easy to implement, works, loses some efficiency on non-square matrices or really large B matrices
pub inline fn lean_blocked_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), C: *const Tensor(T)) !void {
    const dim_num = A.shape.len;

    // Handle 1D tensors as special case
    if (dim_num == 1) {
        if (B.shape.len != 1) {
            return TensorMathError.InputTensorDifferentShape;
        }

        // For 1D vectors, we treat them as a dot product
        // A is a 1D vector (1xK), B is a 1D vector (Kx1), C is a scalar (1x1)
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        if (C.shape.len != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }
        if (C.shape[0] != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }

        var sum: T = 0;
        for (0..K) |k| {
            sum += A.data[k] * B.data[k];
        }

        C.data[0] = sum;
        return;
    }

    // Regular matrix multiplication for dim_num >= 2
    const cache_block_size = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));

    const a_rows = A.shape[A.shape.len - 2];
    const a_cols = A.shape[A.shape.len - 1];

    const b_cols = B.shape[B.shape.len - 1];
    const b_rows = a_cols;

    const c_rows = a_rows;
    const c_cols = b_cols;

    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const C_ptr = C.data.ptr;

    const nearest_c_cols = cache_block_size * (c_cols / cache_block_size);
    //const remaining_c_cols = c_cols - nearest_c_cols;
    const nearest_b_rows = cache_block_size * (b_rows / cache_block_size);
    const remaining_b_rows = b_rows - nearest_b_rows;

    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    var a_vec: @Vector(VEC_WIDTH, T) = undefined;
    var b_vec: @Vector(VEC_WIDTH, T) = undefined;
    var c_vec: @Vector(VEC_WIDTH, T) = undefined;

    var c_chunk_column: usize = 0;

    while (c_chunk_column + cache_block_size <= nearest_c_cols) : (c_chunk_column += cache_block_size) {
        for (0..c_rows) |c_chunk_row| {
            var tile: usize = 0;
            while (tile < nearest_b_rows) : (tile += cache_block_size) {
                for (0..cache_block_size) |t_row| {
                    simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
                }
            }
            //Handle rows that are not a multiple of cache_block_size
            var last_tile: usize = 0;
            while (last_tile < remaining_b_rows) : (last_tile += 1) {
                simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, nearest_b_rows, last_tile, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
            }
        }
    }

    for (0..c_rows) |c_chunk_row| {
        var tile: usize = 0;
        while (tile < nearest_b_rows) : (tile += cache_block_size) {
            for (0..cache_block_size) |t_row| {
                simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
            }
        }

        //Handle rows that are not a multiple of cache_block_size
        var last_tile: usize = 0;
        while (last_tile < remaining_b_rows) : (last_tile += 1) {
            simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, nearest_b_rows, last_tile, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
        }
    }
}

// inline fn tile_mul(
//     comptime T: anytype,
//     A_ptr: [*]T,
//     a_cols: usize,
//     B_ptr: [*]T,
//     b_cols: usize,
//     C_ptr: [*]T,
//     c_cols: usize,
//     tile: usize,
//     t_row: usize,
//     c_chunk_column: usize,
//     c_chunk_row: usize,
// ) void {
//     const CACHE_BLOCK_SIZE = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));
//     const end_col = CACHE_BLOCK_SIZE;

//     for (0..end_col) |t_col| {
//         C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] +=
//             A_ptr[c_chunk_row * a_cols + tile + t_row] *
//             B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col];
//     }
// }

inline fn simd_tile_mul(
    comptime T: anytype,
    A_ptr: [*]T,
    a_cols: usize,
    B_ptr: [*]T,
    b_cols: usize,
    C_ptr: [*]T,
    c_cols: usize,
    tile: usize,
    t_row: usize,
    c_chunk_column: usize,
    c_chunk_row: usize,
    a_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)),
    b_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)),
    c_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)),
) void {
    const CACHE_BLOCK_SIZE = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));

    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    //const VEC_WIDTH: usize = comptime (8);

    // Ensure that c_chunk_column + CACHE_BLOCK_SIZE does not exceed c_cols
    const end_col = @min(CACHE_BLOCK_SIZE, c_cols - c_chunk_column);

    const a_val = A_ptr[c_chunk_row * a_cols + tile + t_row];

    // Create a vector filled with the same value of A
    a_vec.* = @splat(a_val);
    // var b_vec: @Vector(VEC_WIDTH, T) = undefined;
    // var c_vec: @Vector(VEC_WIDTH, T) = undefined;

    // Iteration on columns in blocks of simd_lanes
    var t_col: usize = 0;
    while (t_col + VEC_WIDTH <= end_col) : (t_col += VEC_WIDTH) {

        // Load elements of B into a vector
        for (0..VEC_WIDTH) |i| {
            b_vec[i] = B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col + i];
        }

        // Load current values of C
        for (0..VEC_WIDTH) |i| {
            c_vec[i] = C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col + i];
        }

        // Multiply and accumulate
        c_vec.* += a_vec.* * b_vec.*;

        // Write the result in C
        for (0..VEC_WIDTH) |i| {
            C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col + i] = c_vec[i];
        }
    }

    //Handle remaining columns without SIMD
    while (t_col < end_col) : (t_col += 1) {
        C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] +=
            A_ptr[c_chunk_row * a_cols + tile + t_row] *
            B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col];
    }
}

pub fn get_mat_mul_output_shape(shape_a: []const usize, shape_b: []const usize) ![]usize {
    // Handle 1D tensors (vectors) as special case
    if (shape_a.len == 1 and shape_b.len == 1) {
        // For 1D vectors, output is a scalar (1D tensor with size 1)
        if (shape_a[0] != shape_b[0]) {
            return error.ShapeMismatch;
        }

        var output_shape = try pkg_allocator.alloc(usize, 1);
        errdefer pkg_allocator.free(output_shape);
        output_shape[0] = 1;
        return output_shape;
    }

    // Regular case for matrices/tensors
    if (shape_a.len < 2 or shape_b.len < 2) {
        return error.InvalidShape;
    }

    const a_rows = shape_a[shape_a.len - 2];
    const a_cols = shape_a[shape_a.len - 1];
    const b_rows = shape_b[shape_b.len - 2];
    const b_cols = shape_b[shape_b.len - 1];

    if (a_cols != b_rows) {
        return error.ShapeMismatch;
    }

    var output_shape = try pkg_allocator.alloc(usize, shape_a.len);
    errdefer pkg_allocator.free(output_shape);

    // Copy batch dimensions from input_a
    for (0..shape_a.len - 2) |i| {
        output_shape[i] = shape_a[i];
    }

    // Set matrix multiplication dimensions
    output_shape[output_shape.len - 2] = a_rows;
    output_shape[output_shape.len - 1] = b_cols;

    return output_shape;
}

/// Function that performs the multiplication of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_multiplication(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType), current_depth: usize, location: []usize) !void {
    if (current_depth == (t1.shape.len - 2)) {

        //declaring sum
        var sum: outputType = 0;

        //with the first two for loop I iterate over t3
        for (0..t1.shape[current_depth]) |row| { //for each row of t1

            for (0..t2.shape[current_depth + 1]) |col| { //for each col of t2

                sum = 0;

                for (0..t1.shape[current_depth + 1]) |i| {

                    //compose the location on t1
                    location[t1.shape.len - 1] = i; //location
                    location[t1.shape.len - 2] = row; //location

                    //getting the correct numbers in t1
                    const a = try t1.get_at(location);

                    //compose the location on t2
                    location[t1.shape.len - 1] = col; //location
                    location[t1.shape.len - 2] = i; //location

                    //getting the correct numbers in t2
                    const b = try t2.get_at(location);

                    sum += a * b;
                }

                //compose the location on t3
                location[t1.shape.len - 1] = col; //col on the out tensor matrix
                location[t1.shape.len - 2] = row; //row on the out tensor matrix

                try t3.set_at(location, sum);
            }
        }
    } else {
        for (0..t1.shape[current_depth]) |element_at_current_depth| {
            //print location:
            //std.log.debug("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
            location[current_depth] = element_at_current_depth;
            //otherwise I have to go deeper
            try multidim_multiplication(
                inputType,
                outputType,
                t1,
                t2,
                t3,
                current_depth + 1,
                location,
            );
        }
    }
}

pub fn benchmark_dot_product() !void {
    const allocator = pkg_allocator;

    // Create two large tensors
    var shape1 = [_]usize{ 1024, 1024 };
    var shape2 = [_]usize{ 1024, 1024 };

    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);
    defer t1.deinit();
    defer t2.deinit();

    // Fill with random data
    for (t1.data, 0..) |_, i| {
        t1.data[i] = @floatFromInt(i % 10);
        t2.data[i] = @floatFromInt(i % 10);
    }

    // Benchmark SIMD version
    const timer = try std.time.Timer.start();
    var result1 = try mat_mul(f32, &t1, &t2);
    defer result1.deinit();
    const simd_time = timer.lap();

    // Benchmark recursive version
    var shape_out = [_]usize{ 1024, 1024 };
    var result3 = try Tensor(f32).fromShape(&allocator, &shape_out);
    defer result3.deinit();
    const location = try allocator.alloc(usize, 2);
    defer allocator.free(location);

    const timer3 = try std.time.Timer.start();
    try multidim_multiplication(f32, f32, &t1, &t2, &result3, 0, location);
    const recursive_time = timer3.lap();

    // Print results
    std.log.debug("\nBenchmark Results:\n", .{});
    std.log.debug("SIMD version: {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_time)) / 1_000_000.0});
    std.log.debug("Recursive version: {d:.2} ms\n", .{@as(f64, @floatFromInt(recursive_time)) / 1_000_000.0});
    std.log.debug("\nSpeedups:\n", .{});
    std.log.debug("SIMD vs Recursive: {d:.2}x\n", .{@as(f64, @floatFromInt(recursive_time)) / @as(f64, @floatFromInt(simd_time))});

    // Verify results are the same
    for (result1.data, result3.data) |v1, v3| {
        if (@abs(v1 - v3) > 0.001) {
            std.log.warn("Warning: Results differ!\n", .{});
            break;
        }
    }
}
/// https://onnx.ai/onnx/operators/onnx__MatMul.html
pub fn lowerMatMul(
    b: *UOpBuilder,
    A_id: usize, // SSA id of input matrix A
    B_id: usize, // SSA id of input matrix B
    a_shape: []const usize, // A: shape vec (len 2)
    b_shape: []const usize, // B: shape vec (len 2)
    out_shape: []const usize, // [M, N] output shape
    out_dtype: DType,
) usize {

    // ── Tiny helpers to reduce boilerplate ────────────────────────────
    const r = struct {
        fn rng(bi: *UOpBuilder, end: usize) usize { // RANGE 0..end-1
            return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
        }
        fn kconst(bi: *UOpBuilder, v: usize) usize { // CONST <v>
            return bi.push(.CONST, .i32, &.{}, Any{ .int = v });
        }
    };

    // ── 1. Logical views for A and B (no data copies) -----------------
    // Calculate default row-major strides
    const a_strides = &[_]isize{ @intCast(a_shape[1]), 1 }; // Strides for [M, K] are [K, 1]
    const b_strides = &[_]isize{ @intCast(b_shape[1]), 1 }; // Strides for [K, N] are [N, 1]

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = a_shape, .strides = a_strides } });

    const id_viewB = b.push(.VIEW, out_dtype, &.{B_id}, Any{ .view_meta = .{ .shape = b_shape, .strides = b_strides } });

    // Output buffer
    const id_C = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── 2. Outer loops for output dimensions M x N ------------------
    const c_rows = r.rng(b, out_shape[0]); // rows of output // M
    const c_cols = r.rng(b, out_shape[1]); // columns of output // N

    // ── 3. Accumulator register (one per output element) ---------------
    const id_acc = b.push(.DEFINE_ACC, out_dtype, &.{}, null);

    // ── 4. Inner reduction loop over K dimension ----------------------
    const a_cols = r.rng(b, a_shape[1]); // inner dimension for reduction (K = A's columns = B's rows) // K

    // ── 5. GEPs for current A and B elements ------------------------
    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, c_rows, a_cols }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_gepB = b.push(.GEP, out_dtype, &.{ id_viewB, a_cols, c_cols }, Any{ .mem_info = .{ .base = id_viewB, .offset = 0, .stride = 1 } });

    // ── 6. Multiply & accumulate  acc += A*B ------------------------
    const id_a = b.push(.LOAD, out_dtype, &.{id_gepA}, null);
    const id_b = b.push(.LOAD, out_dtype, &.{id_gepB}, null);
    _ = b.push(.MULACC, out_dtype, &.{ id_acc, id_a, id_b }, null);

    // close reduction loop
    _ = b.push(.ENDRANGE, .bool, &.{a_cols}, null);

    // ── 7. Write output element ------------------------------------------
    const id_gepC = b.push(.GEP, out_dtype, &.{ id_C, c_rows, c_cols }, Any{ .mem_info = .{ .base = id_C, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepC, id_acc }, null);

    // close outer loops (reverse order)
    _ = b.push(.ENDRANGE, .bool, &.{c_cols}, null);
    _ = b.push(.ENDRANGE, .bool, &.{c_rows}, null);

    return id_C; // SSA id of the produced output matrix C
}
