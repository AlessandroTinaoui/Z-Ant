const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const Renderer = codegen.renderer;
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const Tensor = zant.core.tensor.Tensor;

const sum_tensors = zant.core.tensor.math_standard.sum_tensors;

const ZigRenderer = Renderer.ZigRenderer;
const lowerAdd = zant.core.tensor.math_standard.lowerAdd;
const lowerMatMul = zant.core.tensor.math_standard.lowerMatMul;
const lowerMaxPool2d = zant.core.tensor.math_standard.lowerMaxPool2d;
const UOpBuilder = zant.uops.UOpBuilder;
const DType = zant.uops.DType;
const DTypeValue = zant.uops.DTypeValue;
const lowerNeg = zant.core.tensor.math_standard.lowerNeg;
const lowerReshape = zant.core.tensor.math_standard.lowerReshape;
const lowerClip = zant.core.tensor.math_standard.lowerClip;

// /* REMOVED OLD TESTS
// test "Arithmetic operations" { ... }
// ... etc ...
// test "Rendering memory operations" { ... }
// */

test "LowerAdd Pipeline" {
    std.debug.print("Running zig renderer lowerAdd pipeline test (3D broadcast)\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    // No defer needed if we don't take ownership of slice

    // 2. Define inputs for lowerAdd (example shapes/strides)
    const A_id: usize = 0; // Simulated input tensor ID
    const B_id: usize = 1; // Simulated input tensor ID
    // A shape: {2, 3, 4}
    // B shape: {1, 3, 1} (broadcasted to {2, 3, 4})
    // Out shape: {2, 3, 4}
    const out_shape = &.{ 2, 3, 4 }; // Output shape
    const strideA = &.{ 12, 4, 1 }; // Strides for A (row-major for {2,3,4})
    const strideB = &.{ 0, 1, 0 }; // Strides for B (broadcast dim 0 and dim 2, actual data for B is {3})
    const out_dtype = DType.f32;

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = lowerAdd(
        &builder,
        A_id,
        B_id,
        out_shape,
        strideA,
        strideB,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    // Deinit builder immediately as its list is now empty
    builder.deinit();
    // Defer freeing the main slice AFTER freeing internal src slices
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {}; // Dump to stderr
    }
    std.debug.print("--------------------\n", .{});

    // Add defer to free duplicated src slices within the owned list
    defer {
        std.debug.print("DEBUG: Freeing internal src for {d} uops in test\n", .{uops_list.len});
        for (uops_list) |uop| {
            // Free src (only if non-empty)
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            // Free duplicated arg payloads (only if non-null and relevant type)
            if (uop.arg) |arg_val| {
                // Use switch for type-safe union payload access
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            // Only free if non-empty
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {}, // VIEW op with unexpected arg type? Ignore.
                    }
                }
                // Add else if for other duplicated args
                // else if (uop.op == .SOME_OTHER_OP) { ... }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit(); // Deinit renderer AFTER use

    // Specify which IDs are inputs
    const input_ids = &[_]usize{ A_id, B_id };
    try renderer.render_as_function(uops_list, input_ids); // Call the new method

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function ---\n{s}\n---------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/loweradd_output_function.zig"; // Save inside tests dir
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true }); // Ensure write permissions
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated function saved to {s}\n", .{output_filename});

    // Optional: Clean up the generated file
    // try std.fs.cwd().deleteFile(output_filename);
}

test "LowerMatMul Pipeline" {
    std.debug.print("Running zig renderer lowerMatMul pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    // No defer needed if we don't take ownership of slice

    // 2. Define inputs for lowerMatMul
    const A_id: usize = 0; // Simulated input tensor ID (Matrix A)
    const B_id: usize = 1; // Simulated input tensor ID (Matrix B)

    // Example: C[M, N] = A[M, K] @ B[K, N]
    // Let M=2, K=2, N=3
    const M: usize = 2;
    const K: usize = 2;
    const N: usize = 3;

    const shapeA = &.{ M, K }; // Shape of A
    const shapeB = &.{ K, N }; // Shape of B
    const out_shape = &.{ M, N }; // Shape of C (output)

    const out_dtype = DType.f32;

    // 3. Call lowerMatMul to generate UOps
    const out_buf_id = lowerMatMul(
        &builder,
        A_id,
        B_id,
        shapeA,
        shapeB,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    // Deinit builder immediately as its list is now empty
    builder.deinit();
    // Defer freeing the main slice AFTER freeing internal src slices
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (MatMul) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {}; // Dump to stderr
    }
    std.debug.print("-----------------------------\n", .{});

    // Add defer to free duplicated src slices within the owned list
    defer {
        std.debug.print("DEBUG: Freeing internal src for {d} uops in MatMul test\n", .{uops_list.len});
        for (uops_list) |uop| {
            // Free src (only if non-empty)
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            // Free duplicated arg payloads (only if non-null and relevant type)
            if (uop.arg) |arg_val| {
                // Use switch for type-safe union payload access
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            // Only free if non-empty
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {}, // VIEW op with unexpected arg type? Ignore.
                    }
                }
                // Add else if for other duplicated args
                // else if (uop.op == .SOME_OTHER_OP) { ... }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit(); // Deinit renderer AFTER use

    // Specify which IDs are inputs
    const input_ids = &[_]usize{ A_id, B_id };
    try renderer.render_as_function(uops_list, input_ids); // Call the new method

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (MatMul) ---\n{s}\n---------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lower_matmul_output_function.zig"; // Save inside tests dir
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true }); // Ensure write permissions
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated matmul function saved to {s}\n", .{output_filename});

    // Optional: Clean up the generated file
    // try std.fs.cwd().deleteFile(output_filename);
}
test "Test Generated LowerMatMul Kernel" {
    std.debug.print("Testing generated kernel from lower_matmul_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lower_matmul_output_function.zig"); // Import the generated file

    // 1. Define input data (A[M, K], B[K, N])
    const M = 2;
    const K = 2;
    const N = 3;
    const input_data_0: [M * K]f32 = .{ // A: 2x2
        1.0, 2.0, // Row 0
        3.0, 4.0, // Row 1
    };
    const input_data_1: [K * N]f32 = .{ // B: 2x3
        5.0, 6.0, 7.0, // Row 0
        8.0, 9.0, 10.0, // Row 1
    };

    // 2. Call the generated kernel
    const result_slice = try kernel.generated_kernel(allocator, &input_data_0, &input_data_1);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size M*N)

    // 3. Define expected output C[M, N]
    const expected_result: [M * N]f32 = .{
        21.0, 24.0, 27.0, // Row 0
        47.0, 54.0, 61.0, // Row 1
    };

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, &expected_result, result_slice);

    std.debug.print("Generated matmul kernel test passed!\n", .{});
}

test "Test Generated LowerAdd Kernel" {
    std.debug.print("Testing generated kernel from loweradd_output_function.zig (3D broadcast)\n", .{});
    const allocator = std.testing.allocator;
    // Import the generated kernel file
    const kernel = @import("loweradd_output_function.zig");

    // 1. Define input data based on LowerAdd Pipeline (3D broadcast)
    // A shape {2, 3, 4}, stride {12, 4, 1} -> 24 elements, row-major
    var input_data_0_list: [24]f32 = undefined;
    for (0..24) |i| {
        input_data_0_list[i] = @as(f32, @floatFromInt(i + 1)); // 1.0 to 24.0
    }
    const input_data_0 = &input_data_0_list;

    // B has effective shape {3} for data, broadcasted via strides {0, 1, 0}
    // to match output shape {2, 3, 4}.
    // The kernel will access input_1 using an index like ((flat_idx / out_shape[2]) % out_shape[1])
    // which is ((flat_idx / 4) % 3) for out_shape {2,3,4}
    const input_data_1 = [_]f32{ 100.0, 200.0, 300.0 }; // Data for the broadcasted dimension of B

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, input_data_0, &input_data_1);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 24)

    // 3. Define expected output
    // C[i,j,k] = A[i,j,k] + B_broadcasted[j]
    // Output shape {2,3,4}, total 24 elements.
    var expected_result_list: [24]f32 = undefined;
    var flat_idx: usize = 0;
    for (0..2) |_| { // Corresponds to out_shape[0]
        for (0..3) |j| { // Corresponds to out_shape[1]
            for (0..4) |_| { // Corresponds to out_shape[2]
                expected_result_list[flat_idx] = input_data_0[flat_idx] + input_data_1[j];
                flat_idx += 1;
            }
        }
    }
    const expected_result = &expected_result_list;

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, expected_result, result_slice);

    std.debug.print("Generated LowerAdd kernel test (3D broadcast) passed!\n", .{});
}

test "LowerMaxPool2d Pipeline" {
    std.debug.print("Running zig renderer lowerMaxPool2d pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);

    // 2. Define inputs for lowerMaxPool2d
    const A_id: usize = 0; // Simulated input tensor ID
    const out_dtype = DType.f32;

    // Input: NCHW = [1, 1, 4, 4]
    // const shapeA = &.{ 1, 1, 4, 4 }; // Removed unused variable
    // Strides for A (NCHW): [C*H*W, H*W, W, 1] = [16, 16, 4, 1]
    const strideA = &.{ 16, 16, 4, 1 }; // Define input strides
    // Kernel K = [2, 2]
    const kernel_size = .{ 2, 2 }; // Use array literal
    // Stride S = [2, 2]
    const stride = .{ 2, 2 }; // Use array literal
    // Padding P = [0, 0] (top, left) -> Function expects [2]usize
    const padding = .{ 0, 0 }; // Use array literal
    // Dilation D = [1, 1]
    const dilation = .{ 1, 1 }; // Use array literal

    // Output NCHW = [1, 1, 2, 2] (Calculated)
    // H_out = floor((H_in + P_top + P_bottom - D_h * (K_h - 1) - 1) / S_h + 1)
    //       = floor((4 + 0 + 0 - 1 * (2 - 1) - 1) / 2 + 1) = floor((4 - 1 - 1)/2 + 1) = floor(2/2 + 1) = 2
    // W_out = floor((W_in + P_left + P_right - D_w * (K_w - 1) - 1) / S_w + 1)
    //       = floor((4 + 0 + 0 - 1 * (2 - 1) - 1) / 2 + 1) = floor((4 - 1 - 1)/2 + 1) = floor(2/2 + 1) = 2
    const out_shape = &.{ 1, 1, 2, 2 };

    // 3. Call lowerMaxPool2d to generate UOps
    const out_buf_id = lowerMaxPool2d(
        &builder,
        A_id, // X_id
        out_shape, // out_shape
        strideA, // in_stride
        padding, // pads: [2]usize
        stride, // strides_hw: [2]usize
        dilation, // dil_hw: [2]usize
        kernel_size, // kHW: [2]usize
        out_dtype, // out_dtype
        false, // ceil_mode
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (MaxPool2d) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("-------------------------------\n", .{});

    // Defer freeing internal src/args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in MaxPool2d test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) allocator.free(@constCast(uop.src));
            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
                // Add other duplicated args if needed
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{A_id};
    try renderer.render_as_function(uops_list, input_ids);

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (MaxPool2d) ---\n{s}\n-------------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lower_maxpool2d_output_function.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated maxpool2d function saved to {s}\n", .{output_filename});

    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerMaxPool2d Kernel" {
    std.debug.print("Testing generated kernel from lower_maxpool2d_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    // Import the generated kernel file
    // IMPORTANT: Build system needs to know about this generated file or test needs to run after generation
    // For simplicity here, assume it's generated before `zig test` is run on this file.
    const kernel = @import("lower_maxpool2d_output_function.zig");

    // 1. Define input data based on LowerMaxPool2d Pipeline shapes/strides
    // Input shapeA: [1, 1, 4, 4], flat size = 16
    const input_data_0 = [_]f32{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, &input_data_0);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 4)

    // 3. Define expected output
    // Output shape: [1, 1, 2, 2], flat size = 4
    // K=[2,2], S=[2,2], P=[0,0], D=[1,1]
    // O[0,0] = max(I[0,0], I[0,1], I[1,0], I[1,1]) = max(1, 2, 5, 6) = 6
    // O[0,1] = max(I[0,2], I[0,3], I[1,2], I[1,3]) = max(3, 4, 7, 8) = 8
    // O[1,0] = max(I[2,0], I[2,1], I[3,0], I[3,1]) = max(9, 10, 13, 14) = 14
    // O[1,1] = max(I[2,2], I[2,3], I[3,2], I[3,3]) = max(11, 12, 15, 16) = 16
    const expected_result = [_]f32{
        6.0, 8.0, 14.0, 16.0,
    };

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, &expected_result, result_slice);

    std.debug.print("Generated LowerMaxPool2d kernel test passed!\n", .{});
}

test "LowerConv2d Pipeline" {
    std.debug.print("Running zig renderer lowerConv2d pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);

    // 2. Define inputs for lowerConv2d
    const X_id: usize = 0; // Input Tensor ID
    const W_id: usize = 1; // Weight Tensor ID
    const out_dtype = DType.f32;

    // Input X: NCHW = [1, 1, 3, 3]
    // const shapeX = &.{ 1, 1, 3, 3 }; // Removed unused variable
    // Strides for X (NCHW): [C*H*W, H*W, W, 1] = [9, 9, 3, 1]
    const strideX = &.{ 9, 9, 3, 1 };

    // Weights W: OIHW = [1, 1, 2, 2] (O=1, I=1/groups=1, KH=2, KW=2)
    // const shapeW = &.{ 1, 1, 2, 2 }; // Removed unused variable
    // Strides for W (OIHW): [I*KH*KW, KH*KW, KW, 1] = [4, 4, 2, 1] (assuming groups=1)
    const strideW = &.{ 4, 4, 2, 1 };

    // Kernel K = [2, 2]
    const kernel_size = .{ 2, 2 };
    // Stride S = [1, 1]
    const stride = .{ 1, 1 };
    // Padding P = [0, 0, 0, 0] (top, left, bottom, right) -> Func expects [4]usize
    const padding = .{ 0, 0, 0, 0 };
    // Dilation D = [1, 1]
    const dilation = .{ 1, 1 };
    // Groups G = 1
    const groups: usize = 1;

    // Output NCHW = [1, 1, 2, 2] (Calculated based on params)
    // H_out = floor((3 + 0 + 0 - 1 * (2 - 1) - 1) / 1 + 1) = 2
    // W_out = floor((3 + 0 + 0 - 1 * (2 - 1) - 1) / 1 + 1) = 2
    const out_shape = &.{ 1, 1, 2, 2 };

    const C_in: usize = 1; // Input channels from shapeX[1]
    const M_out: usize = 1; // Output channels from out_shape[1]
    const C_per_grp = C_in / groups;
    const M_per_grp = M_out / groups;

    // 3. Call lowerConv2d to generate UOps
    const out_buf_id = zant.core.tensor.math_standard.lowerConv2d(
        &builder,
        X_id, // X_id
        W_id, // W_id
        out_shape, // out_shape
        strideX, // in_stride
        strideW, // w_stride
        groups, // groups
        .{ padding[0], padding[1] }, // pads: [2]usize {top, left}
        stride, // strides_hw: [2]usize
        dilation, // dil_hw: [2]usize
        kernel_size, // kHW: [2]usize
        C_per_grp, // C' input channels per group
        M_per_grp, // M' output channels per group
        out_dtype, // out_dtype
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (Conv2d) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("-----------------------------\n", .{});

    // Defer freeing internal src/args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in Conv2d test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) allocator.free(@constCast(uop.src));
            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
                // Add other duplicated args if needed
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{ X_id, W_id }; // X, W are inputs (Bias B_id removed)
    try renderer.render_as_function(uops_list, input_ids);

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (Conv2d) ---\n{s}\n-----------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lower_conv2d_output_function.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated conv2d function saved to {s}\n", .{output_filename});

    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerConv2d Kernel" {
    std.debug.print("Testing generated kernel from lower_conv2d_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    // Import the generated kernel file
    const kernel = @import("lower_conv2d_output_function.zig");

    // 1. Define input data based on LowerConv2d Pipeline
    // Input X: [1, 1, 3, 3], flat size = 9
    const input_data_0 = [_]f32{ // X
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    // Weights W: [1, 1, 2, 2], flat size = 4
    const input_data_1 = [_]f32{ // W
        1.0, 1.0,
        1.0, 1.0,
    };
    // Bias B: [1], flat size = 1 -- REMOVED, bias handled separately
    // const input_data_2 = [_]f32{ // B
    //     0.5,
    // };

    // 2. Call the generated kernel
    // Signature expected: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, &input_data_0, &input_data_1); // REMOVED input_data_2
    defer allocator.free(result_slice); // Kernel allocates output slice (size 4)

    // 3. Define expected output (WITHOUT BIAS)
    // Output shape: [1, 1, 2, 2], flat size = 4
    // K=[2,2], S=[1,1], P=[0,0,0,0], D=[1,1], G=1
    // O[n,o,h,w] = sum(I[n, g*I_c + i, h*S_h + kh*D_h - P_t, w*S_w + kw*D_w - P_l] * W[o, i, kh, kw])
    // O[0,0,0,0] = I[0,0,0,0]*W[0,0,0,0] + I[0,0,0,1]*W[0,0,0,1] + I[0,0,1,0]*W[0,0,1,0] + I[0,0,1,1]*W[0,0,1,1]
    //            = 1*1 + 2*1 + 4*1 + 5*1 = 12.0
    // O[0,0,0,1] = I[0,0,0,1]*W[0,0,0,0] + I[0,0,0,2]*W[0,0,0,1] + I[0,0,1,1]*W[0,0,1,0] + I[0,0,1,2]*W[0,0,1,1]
    //            = 2*1 + 3*1 + 5*1 + 6*1 = 16.0
    // O[0,0,1,0] = I[0,0,1,0]*W[0,0,0,0] + I[0,0,1,1]*W[0,0,0,1] + I[0,0,2,0]*W[0,0,1,0] + I[0,0,2,1]*W[0,0,1,1]
    //            = 4*1 + 5*1 + 7*1 + 8*1 = 24.0
    // O[0,0,1,1] = I[0,0,1,1]*W[0,0,0,0] + I[0,0,1,2]*W[0,0,0,1] + I[0,0,2,1]*W[0,0,1,0] + I[0,0,2,2]*W[0,0,1,1]
    //            = 5*1 + 6*1 + 8*1 + 9*1 = 28.0
    const expected_result = [_]f32{
        12.0, 16.0, 24.0, 28.0, // Removed bias of 0.5 from original expected
    };

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, &expected_result, result_slice);

    std.debug.print("Generated LowerConv2d kernel test passed!\n", .{});
}

test "LowerNeg Pipeline" {
    std.debug.print("Running zig renderer lowerNeg pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);

    // 2. Define inputs for lowerNeg
    const A_id: usize = 0; // Simulated input tensor ID
    // A shape: {2, 3}
    const input_shape = &.{ @as(usize, 2), @as(usize, 3) }; // Explicitly type for array literal
    const out_shape = input_shape; // For Neg, output shape is same as input
    const strideA = &.{ @as(isize, 3), @as(isize, 1) }; // Strides for A (row-major for {2,3})
    const out_dtype = DType.f32;

    // 3. Call lowerNeg to generate UOps
    const out_buf_id = lowerNeg(
        &builder,
        A_id,
        strideA,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (Neg) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("--------------------------\n", .{});

    // Defer to free duplicated src slices and view_meta args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in Neg test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{A_id};
    try renderer.render_as_function(uops_list, input_ids);

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (Neg) ---\n{s}\n--------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lowerneg_output_function.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated neg function saved to {s}\n", .{output_filename});

    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerNeg Kernel" {
    std.debug.print("Testing generated kernel from lowerneg_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lowerneg_output_function.zig");

    // 1. Define input data
    // Input shape {2, 3}, flat size = 6
    const input_data_0_list = [_]f32{
        1.0,  -2.0, 3.0,
        -4.0, 5.0,  0.0,
    };
    const input_data_0 = &input_data_0_list;

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, input_data_0);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 6)

    // 3. Define expected output
    // Output shape {2, 3}, flat size = 6
    const expected_result_list = [_]f32{
        -1.0, 2.0, -3.0,
        4.0, -5.0, -0.0, // Note: -0.0 is f32 representation
    };
    const expected_result = &expected_result_list;

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, expected_result, result_slice);

    std.debug.print("Generated LowerNeg kernel test passed!\n", .{});
}

test "LowerClip Pipeline with f32" {
    std.debug.print("Running zig renderer lowerClip pipeline test with f32\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);

    // 2. Define inputs for lowerClip
    const A_id: usize = 0; // Simulated input tensor ID
    // A shape: {2, 3}
    const input_shape = &.{ @as(usize, 2), @as(usize, 3) }; // Explicitly type for array literal
    const out_shape = input_shape; // For Clip, output shape is same as input
    const strideA = &.{ @as(isize, 3), @as(isize, 1) };
    const out_dtype = DType.f32;
    const max = DTypeValue{ .f32 = 2.0 };
    const min = DTypeValue{ .f32 = -2.0 };

    // 3. Call lowerNeg to generate UOps
    const out_buf_id = lowerClip(
        &builder,
        A_id,
        out_shape,
        strideA,
        out_dtype,
        min,
        max,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps

    std.debug.print("--- Generated UOps (Neg) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("--------------------------\n", .{});

    // Defer to free duplicated src slices and view_meta args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in Clip test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }

            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{A_id};
    try renderer.render_as_function(uops_list, input_ids);

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (Clip) ---\n{s}\n--------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lowerclip_output_function.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated clip function saved to {s}\n", .{output_filename});

    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerClip Kernel with f32" {
    std.debug.print("Testing generated kernel from lowerclip_output_function.zig with f32\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lowerclip_output_function.zig");

    // 1. Define input data
    // Input shape {2, 3}, flat size = 6
    const input_data_0_list = [_]f32{
        1.0,  -2.0, 3.0,
        -4.0, 5.0,  0.0,
    };
    const input_data_0 = &input_data_0_list;

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, input_data_0);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 6)

    // 3. Define expected output
    // Output shape {2, 3}, flat size = 6
    const expected_result_list = [_]f32{
        1.0, -2.0, 2.0,
        -2.0, 2.0, 0.0, // Note: -0.0 is f32 representation
    };
    const expected_result = &expected_result_list;

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, expected_result, result_slice);

    std.debug.print("Generated LowerClip kernel test passed!\n", .{});
}

test "LowerClip Pipeline with u16" {
    std.debug.print("Running zig renderer lowerClip pipeline test with u16\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);

    // 2. Define inputs for lowerClip
    const A_id: usize = 0; // Simulated input tensor ID
    // A shape: {2, 3}
    const input_shape = &.{ @as(usize, 2), @as(usize, 3) }; // Explicitly type for array literal
    const out_shape = input_shape; // For Clip, output shape is same as input
    const strideA = &.{ @as(isize, 3), @as(isize, 1) };
    const out_dtype = DType.u16;
    const max = DTypeValue{ .u16 = 8 };
    const min = DTypeValue{ .u16 = 2 };

    // 3. Call lowerNeg to generate UOps
    const out_buf_id = lowerClip(
        &builder,
        A_id,
        out_shape,
        strideA,
        out_dtype,
        min,
        max,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (Neg) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("--------------------------\n", .{});

    // Defer to free duplicated src slices and view_meta args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in Clip test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{A_id};
    try renderer.render_as_function(uops_list, input_ids);

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (Clip) ---\n{s}\n--------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lowerclip_output_function2.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated clip function saved to {s}\n", .{output_filename});

    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerClip Kernel with u16" {
    std.debug.print("Testing generated kernel from lowerclip_output_function.zig with u16\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lowerclip_output_function2.zig");

    // 1. Define input data
    // Input shape {2, 3}, flat size = 6
    const input_data_0_list = [_]u16{
        1, 2, 10,
        6, 3, 16,
    };
    const input_data_0 = &input_data_0_list;

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, input_data_0);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 6)

    // 3. Define expected output
    // Output shape {2, 3}, flat size = 6
    const expected_result_list = [_]u16{
        2, 2, 8,
        6, 3, 8,
    };
    const expected_result = &expected_result_list;

    // 4. Compare results
    try std.testing.expectEqualSlices(u16, expected_result, result_slice);

    std.debug.print("Generated LowerClip kernel test passed!\n", .{});
}

test "LowerReshape Pipeline" {
    std.debug.print("Running zig renderer lowerReshape pipeline test\n", .{});
    const allocator = std.testing.allocator;
    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    const A_id: usize = 0; // Simula
    const out_dtype = DType.f32;
    const out_shape = &.{ 2, 3 }; // Shape of input tensor
    const out_buf_id = try lowerReshape(
        &builder,
        A_id,
        out_shape,
        out_dtype,
    );

    _ = out_buf_id; // Prevent unused warning
    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);
    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (Reshape) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("--------------------------\n", .{});
    // Defer to free duplicated src slices and view_meta args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in Clip test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
            }
        }
    }

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());

    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{A_id};
    try renderer.render_as_function(uops_list, input_ids);

    defer {
        var iter = renderer.view_map.valueIterator();
        while (iter.next()) |val| {
            allocator.free(val.arg.view_meta.strides);
        }
    }

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);
    std.debug.print("\n--- Generated Function (Reshape) ---\n{s}\n--------------------------------\n", .{actual_code});
    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lowerreshape_output_function.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated reshape function saved to {s}\n", .{output_filename});
    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerReshape Kernel" {
    std.debug.print("Testing generated kernel from lowerreshape_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lowerreshape_output_function.zig");

    // 1. Define input data
    // Input shape {2, 3}, flat size = 6
    const input_data_0_list = [_]f32{
        1, 2, 3, 4, 5, 6,
    };
    const input_data_0 = &input_data_0_list;

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, input_data_0);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 6)

    // 3. Define expected output
    // Output shape {2, 3}, flat size = 6
    const expected_result_list = [_]f32{
        1, 2, 3, 4, 5, 6,
    };
    const expected_result = &expected_result_list;

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, expected_result, result_slice);

    std.debug.print("Generated LowerReshape kernel test passed!\n", .{});
}

test "LowerReshape Pipeline 2" {
    std.debug.print("Running zig renderer lowerReshape pipeline test\n", .{});
    const allocator = std.testing.allocator;
    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    const A_id: usize = 0; // Simula
    const out_dtype = DType.f32;
    const out_shape = &.{ 2, 3, 4 }; // Shape of input tensor
    const out_buf_id = try lowerReshape(
        &builder,
        A_id,
        out_shape,
        out_dtype,
    );

    _ = out_buf_id; // Prevent unused warning
    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    builder.deinit();
    defer allocator.free(uops_list);
    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (Reshape) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {};
    }
    std.debug.print("--------------------------\n", .{});
    // Defer to free duplicated src slices and view_meta args
    defer {
        std.debug.print("DEBUG: Freeing internal src/args for {d} uops in Clip test\n", .{uops_list.len});
        for (uops_list) |uop| {
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            if (uop.arg) |arg_val| {
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {},
                    }
                }
            }
        }
    }

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());

    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const input_ids = &[_]usize{A_id};
    try renderer.render_as_function(uops_list, input_ids);
    defer {
        var iter = renderer.view_map.valueIterator();
        while (iter.next()) |val| {
            allocator.free(val.arg.view_meta.strides);
        }
    }
    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);
    std.debug.print("\n--- Generated Function (Reshape) ---\n{s}\n--------------------------------\n", .{actual_code});
    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lowerreshape_output_function2.zig";
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true });
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated reshape function saved to {s}\n", .{output_filename});
    // Optional: Clean up
    // try std.fs.cwd().deleteFile(output_filename);
}

test "Test Generated LowerReshape Kernel 2" {
    std.debug.print("Testing generated kernel from lowerreshape_output_function2.zig\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lowerreshape_output_function2.zig");

    // 1. Define input data
    // Input shape {2, 3}, flat size = 6
    const input_data_0_list = [_]f32{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    };
    const input_data_0 = &input_data_0_list;

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, input_data_0);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 6)

    // 3. Define expected output
    // Output shape {2, 3}, flat size = 6
    const expected_result_list = [_]f32{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    };
    const expected_result = &expected_result_list;

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, expected_result, result_slice);

    std.debug.print("Generated LowerReshape kernel test passed!\n", .{});
}
