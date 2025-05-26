const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const utils = codegen.utils;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const tests_log = std.log.scoped(.test_utils);

const model = @import("model_options.zig");

const ITERATION_COUNT: u32 = 100;

test "Static Library - Random data Prediction Test" {
    std.testing.log_level = .info;

    tests_log.info("\n     test: Static Library - Model: {s}  - Random data Prediction Test\n", .{model.name});

    var input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (input_shape) |dim| {
        input_data_size *= dim;
    }

    // Create input data array directly instead of ArrayList
    var input_data = try allocator.alloc(model.data_type, input_data_size);
    defer allocator.free(input_data);

    // Generate random data
    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    for (0..ITERATION_COUNT) |_| {
        // Fill with random values
        for (0..input_data_size) |i| {
            input_data[i] = rand.float(model.data_type) * 100;
        }

        var result: [*]model.data_type = undefined;

        // Run prediction
        model.lib.predict(
            input_data.ptr,
            @ptrCast(&input_shape),
            input_shape.len,
            &result,
        );
    }
    tests_log.info("\nRan 100 fuzzy tests on model \"{s}\", done without errors:\n", .{model.name});
}

test "Static Library - Inputs Prediction Test" {
    std.testing.log_level = .info;

    tests_log.info("\n     test: Codegen one-op model: \"{s}\" compare with Pre-Generated results.\n", .{model.name});

    var input_shape = model.input_shape;

    var input_data_len: u32 = 1;
    for (input_shape) |dim| {
        input_data_len *= dim;
    }

    // Read json for output to compare it with.
    const user_tests_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/user_tests.json", .{model.name});
    defer allocator.free(user_tests_path);

    tests_log.info("{s}", .{user_tests_path});

    const parsed_user_tests = try utils.loadUserTests(model.data_type, user_tests_path);
    defer parsed_user_tests.deinit();

    const user_tests = parsed_user_tests.value;

    tests_log.info("\nUser tests loaded.\n", .{});

    for (user_tests) |user_test| {
        tests_log.info("\n\tRunning user test: {s}\n\n", .{user_test.name});

        try std.testing.expectEqual(user_test.input.len, input_data_len);

        var result: [*]model.data_type = undefined;

        // Run prediction
        model.lib.predict(
            user_test.input.ptr,
            @ptrCast(&input_shape),
            input_shape.len,
            &result,
        );

        for (0.., user_test.output) |i, expected_output| {
            const result_value = result[i];
            const expected_output_value = expected_output;
            const approx_eq = std.math.approxEqAbs(model.data_type, expected_output_value, result_value, 0.001);
            if (!approx_eq)
                tests_log.warn("Test failed for input: {d} expected: {} got: {}\n", .{ i, expected_output_value, result_value });

            try std.testing.expect(approx_eq);
        }
    }
}
