const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const PoolingType = zant.core.tensor.math_standard.PoolingType;
const AutoPadType = zant.core.tensor.math_standard.AutoPadType;

const tests_log = std.log.scoped(.test_pooling);

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const lowerMaxPool2d = zant.core.tensor.math_standard.lowerMaxPool2d;

test "lowerMaxPool2d - print UOps sequence" {
    std.debug.print("\n     test: lowerMaxPool2d - print UOps sequence\n", .{});

    const test_allocator = pkgAllocator.allocator;
    var b = UOpBuilder.init(test_allocator);
    defer b.deinit();

    // Create dummy input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&test_allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    const X_id = b.push(.DEFINE_GLOBAL, .f32, &.{}, Any{ .shape = &input_shape });

    // Define output shape and other parameters for max pooling
    const out_shape = [_]usize{ 1, 1, 3, 3 }; // With 2x2 kernel and stride 1, output is (5-2+1)x(5-2+1)
    const in_stride = [_]isize{ 25, 25, 5, 1 }; // Strides for 1x1x5x5 tensor in row-major format
    const pads = [_]usize{ 0, 0 }; // No padding
    const strides_hw = [_]usize{ 1, 1 }; // Stride of 1 in height and width
    const dil_hw = [_]usize{ 1, 1 }; // No dilation
    const kHW = [_]usize{ 2, 2 }; // 2x2 kernel

    // Call lowerMaxPool2d
    _ = lowerMaxPool2d(
        &b,
        X_id,
        &out_shape,
        &in_stride,
        pads,
        strides_hw,
        dil_hw,
        kHW,
        .f32,
        false,
    );

    // Print the generated UOps sequence
    std.debug.print("\nUOps sequence for MaxPool2d:\n", .{});
    for (b.list.items, 0..) |op, i| {
        std.debug.print("{d:3}: {s}\n", .{ i, @tagName(op.op) });
    }
}

test "Pooling 2D" {
    tests_log.info("\n     test: Pooling 2D\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3 input, same as original
    var shape_tensor: [2]usize = [_]usize{ 3, 3 };
    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 40.0, 50.0, 60.0 },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    // Calculate W = number of windows
    const input_rows = input1.shape[0];
    const input_cols = input1.shape[1];
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; // 2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; // 2
    const W = out_rows * out_cols; // 4

    // Instead of used_input being shape of input, now [W,3,3]
    var used_input1_shape = [_]usize{ W, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output1: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output1.deinit();

    // Same checks for output as original
    try std.testing.expectEqual(output1.shape.len, input1.shape.len);
    try std.testing.expectEqual(output1.shape[0], 2);
    try std.testing.expectEqual(output1.shape[1], 2);

    try std.testing.expectEqual(output1.data[0], 5);
    try std.testing.expectEqual(output1.data[1], 6);
    try std.testing.expectEqual(output1.data[2], 50);
    try std.testing.expectEqual(output1.data[3], 60);

    // Now we have 4 windows (W=4), each 3x3
    // The original code expected certain pattern in used_input:
    // (1,1)=5 for first window
    // (1,2)=6 for second window
    // (2,1)=50 for third window
    // (2,2)=60 for fourth window
    //
    // Windows:
    // w=0: top-left max at (1,1)
    // w=1: top-right max at (1,2)
    // w=2: bottom-left max at (2,1)
    // w=3: bottom-right max at (2,2)

    // Check w=0 window:
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * 9 + 1 * 3 + 1]);

    // w=1: top-right (1,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[1 * 9 + 1 * 3 + 2]);

    // w=2: bottom-left (2,1)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[2 * 9 + 2 * 3 + 1]);

    // w=3: bottom-right (2,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[3 * 9 + 2 * 3 + 2]);

    // Another test with kernel2 and stride2
    var kernel2: [2]usize = [2]usize{ 2, 2 };
    var stride2: [2]usize = [2]usize{ 2, 2 };

    const out_rows_2 = (3 - 2 + 1) / 2;
    const out_cols_2 = (3 - 2 + 1) / 2;
    const W2 = out_rows_2 * out_cols_2;

    var used_input2_shape = [_]usize{ W2, 3, 3 };
    var used_input2 = try Tensor(u8).fromShape(&allocator, used_input2_shape[0..]);
    defer used_input2.deinit();
    for (used_input2.data) |*val| val.* = 0;

    var output2: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input2, &kernel2, &stride2, PoolingType.Max);
    defer output2.deinit();

    try std.testing.expectEqual(output2.shape.len, input1.shape.len);
    try std.testing.expectEqual(output2.shape[0], 1);
    try std.testing.expectEqual(output2.shape[1], 1);
    try std.testing.expectEqual(output2.data[0], 5);

    // single window w=0 max at (1,1)
    try std.testing.expectEqual(@as(u1, 1), used_input2.data[0 * (3 * 3) + 1 * 3 + 1]);
}

test "Pooling multidim" {
    tests_log.info("\n     test: Pooling multidim\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3x3 input as original
    var shape_tensor: [3]usize = [_]usize{ 3, 3, 3 };
    var inputArray: [3][3][3]f32 = [_][3][3]f32{
        [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ 10.0, 20.0, 30.0 },
            [_]f32{ 40.0, 0.0, -10.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ -1.0, -2.0, -3.0 },
            [_]f32{ -4.0, -5.0, -6.0 },
            [_]f32{ -40.0, -50.0, -60.0 },
        },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    const d = input1.shape[0]; // 3
    const input_rows = input1.shape[1]; //3
    const input_cols = input1.shape[2]; //3
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; //2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; //2
    const W_tot = d * out_rows * out_cols; // 3*2*2=12

    var used_input1_shape = [_]usize{ W_tot, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output.deinit();

    try std.testing.expectEqual(output.shape.len, input1.shape.len);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 2);
    try std.testing.expectEqual(output.shape[2], 2);

    // We do not check used_input details here, just shapes as in original.
}

test "ONNX MaxPool - NOTSET padding" {
    tests_log.info("\n     test: ONNX MaxPool - NOTSET padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x4x4
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 2, 2 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 }; // top, left, bottom, right

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[3]);

    try std.testing.expectEqual(@as(f32, 6), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 14), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 16), result.output.data[3]);
}

test "ONNX MaxPool - SAME_UPPER padding" {
    tests_log.info("\n     test: ONNX MaxPool - SAME_UPPER padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 }; // not used in SAME_UPPER

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .SAME_UPPER,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), result.output.shape[3]);

    try std.testing.expectEqual(@as(f32, 5), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[3]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[4]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[5]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[6]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[7]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[8]);
}

test "ONNX MaxPool - with dilation" {
    tests_log.info("\n     test: ONNX MaxPool - with dilation\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x4x4
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 2, 2 }; // Dilated kernel
    var pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[3]);

    // With dilation=2, each kernel element skips one position
    // So kernel covers positions: [[1,3],[9,11]] for first window
    try std.testing.expectEqual(@as(f32, 11), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 12), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 15), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 16), result.output.data[3]);
}

test "ONNX MaxPool - ceil mode" {
    tests_log.info("\n     test: ONNX MaxPool - ceil mode\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 2, 2 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        true, // ceil_mode = true
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[2]); // Ceil mode makes this 2 instead of 1
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[3]);

    try std.testing.expectEqual(@as(f32, 5), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[3]);
}

test "ONNX MaxPool - explicit padding" {
    tests_log.info("\n     test: ONNX MaxPool - explicit padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 1, 1, 1, 1 }; // pad 1 on all sides

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), result.output.shape[3]);

    // First row includes padding (0s)
    try std.testing.expectEqual(@as(f32, 1), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 2), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 3), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 3), result.output.data[3]);

    // Middle rows
    try std.testing.expectEqual(@as(f32, 4), result.output.data[4]);
    try std.testing.expectEqual(@as(f32, 5), result.output.data[5]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[6]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[7]);

    try std.testing.expectEqual(@as(f32, 7), result.output.data[8]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[9]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[10]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[11]);

    // Last row includes padding (0s)
    try std.testing.expectEqual(@as(f32, 7), result.output.data[12]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[13]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[14]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[15]);
}

test "ONNX AveragePool - NOTSET padding" {
    tests_log.info("\n     test: ONNX AveragePool - NOTSET padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x4x4
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 2, 2 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 }; // top, left, bottom, right

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
        false, // count_include_pad = false
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.5), output.data[0]); // (1+2+5+6)/4
    try std.testing.expectEqual(@as(f32, 5.5), output.data[1]); // (3+4+7+8)/4
    try std.testing.expectEqual(@as(f32, 11.5), output.data[2]); // (9+10+13+14)/4
    try std.testing.expectEqual(@as(f32, 13.5), output.data[3]); // (11+12+15+16)/4
}

test "ONNX AveragePool - SAME_UPPER padding" {
    tests_log.info("\n     test: ONNX AveragePool - SAME_UPPER padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 }; // non usati con SAME_UPPER

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .SAME_UPPER,
        false,
        false,
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.0), output.data[0]); // (1+2+4+5)/4
    try std.testing.expectEqual(@as(f32, 4.0), output.data[1]); // (2+3+5+6)/4
    try std.testing.expectEqual(@as(f32, 4.5), output.data[2]); // (3+6+0+0)/2
    try std.testing.expectEqual(@as(f32, 6.0), output.data[3]); // (4+5+7+8)/4
    try std.testing.expectEqual(@as(f32, 7.0), output.data[4]); // (5+6+8+9)/4
    try std.testing.expectEqual(@as(f32, 7.5), output.data[5]); // (6+9+0+0)/2
    try std.testing.expectEqual(@as(f32, 7.5), output.data[6]); // (7+8+0+0)/2
    try std.testing.expectEqual(@as(f32, 8.5), output.data[7]); // (8+9+0+0)/2
    try std.testing.expectEqual(@as(f32, 9.0), output.data[8]); // (9+0+0+0)/1
}

test "ONNX AveragePool - with dilation" {
    tests_log.info("\n     test: ONNX AveragePool - with dilation\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x4x4
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 2, 2 };
    var pads = [_]usize{ 0, 0, 0, 0 };

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
        false,
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 6.0), output.data[0]); // (1+3+9+11)/4
    try std.testing.expectEqual(@as(f32, 7.0), output.data[1]); // (2+4+10+12)/4
    try std.testing.expectEqual(@as(f32, 10.0), output.data[2]); // (5+7+13+15)/4
    try std.testing.expectEqual(@as(f32, 11.0), output.data[3]); // (6+8+14+16)/4
}

test "ONNX AveragePool - ceil mode" {
    tests_log.info("\n     test: ONNX AveragePool - ceil mode\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 2, 2 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 };

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        true, // ceil_mode = true
        false,
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.0), output.data[0]); // (1+2+4+5)/4
    try std.testing.expectEqual(@as(f32, 4.5), output.data[1]); // (3+6)/2
    try std.testing.expectEqual(@as(f32, 7.5), output.data[2]); // (7+8)/2
    try std.testing.expectEqual(@as(f32, 9.0), output.data[3]); // (9)/1
}

test "ONNX AveragePool - explicit padding with count_include_pad" {
    tests_log.info("\n     test: ONNX AveragePool - explicit padding with count_include_pad\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 1, 1 }; // bottom, right padding

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
        true, // count_include_pad = true
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.0), output.data[0]); // (1+2+4+5)/4
    try std.testing.expectEqual(@as(f32, 4.0), output.data[1]); // (2+3+5+6)/4
    try std.testing.expectEqual(@as(f32, 2.25), output.data[2]); // (3+6+0+0)/4
    try std.testing.expectEqual(@as(f32, 6.0), output.data[3]); // (4+5+7+8)/4
    try std.testing.expectEqual(@as(f32, 7.0), output.data[4]); // (5+6+8+9)/4
    try std.testing.expectEqual(@as(f32, 3.75), output.data[5]); // (6+9+0+0)/4
    try std.testing.expectEqual(@as(f32, 3.75), output.data[6]); // (7+8+0+0)/4
    try std.testing.expectEqual(@as(f32, 4.25), output.data[7]); // (8+9+0+0)/4
    try std.testing.expectEqual(@as(f32, 2.25), output.data[8]); // (9+0+0+0)/4
}

test "Manual AveragePool Test" {
    const T = f32;
    var allocator = std.testing.allocator;

    // Input tensor [1, 1, 4, 4]
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    var input = try Tensor(T).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    // Output tensor [1, 1, 3, 3]
    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output = try Tensor(T).fromShape(&allocator, &output_shape);
    defer output.deinit();

    const kernel_shape = [_]usize{ 2, 2 };
    const strides = [_]usize{ 1, 1 };
    const dilations = [_]usize{ 1, 1 };
    const pads = [_]usize{ 0, 0, 0, 0 };

    try TensMath.onnx_averagepool_lean(T, &input, &output, &kernel_shape, &strides, &dilations, &pads, .NOTSET, false);

    const expected = [_]f32{ 3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5 };
    for (output.data, expected) |got, exp| {
        try std.testing.expectApproxEqAbs(got, exp, 1e-5);
    }
}
