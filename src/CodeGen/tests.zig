const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen.zig");
const globals = codegen.globals;
const utils = codegen.utils;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const codegen_options = @import("codegen_options");
const allocator = zant.utils.allocator.allocator;

pub fn UserTest(comptime T: type) type {
    return struct {
        name: []u8,
        type: []u8,
        input: []T,
        output: []T,
        expected_class: usize,
    };
}

fn writeModelOptionsFile(model_name: []const u8, model_path: []const u8) !void {
    // Generate model_options.zig

    const model_options_path = try std.fmt.allocPrint(allocator, "{s}model_options.zig", .{model_path});
    const model_options_file = try std.fs.cwd().createFile(model_options_path, .{});
    defer model_options_file.close();

    const writer = model_options_file.writer();

    ////////////

    var output_data_len: i64 = 1;

    for (globals.networkOutput.shape) |dim| {
        output_data_len *= dim;
    }

    _ = try writer.print(
        \\
        \\pub const lib = @import("lib_{s}.zig");
        \\pub const name = "{s}";
        \\pub const input_shape = [{d}]u32{any};
        \\pub const output_data_len = {d};
        \\pub const data_type = {s};
        \\pub const enable_user_tests : bool = {any};
        \\pub const user_tests_path = "{s}";
    , .{
        model_name,
        model_name,
        globals.networkInput.shape.len,
        globals.networkInput.shape,
        output_data_len,
        codegen_options.type,
        codegen_options.user_tests.len > 0,
        try std.fmt.allocPrint(allocator, "{s}user_tests.json", .{model_path}),
    });

    ////////////
}

pub fn writeTestFile(model_name: []const u8, model_path: []const u8) !void {

    // Copy test file template into the generated test file
    const test_file_path = try std.fmt.allocPrint(allocator, "{s}test_{s}.zig", .{ model_path, model_name });

    try utils.copyFile("tests/CodeGen/test_model.template.zig", test_file_path);
    std.log.info("\n\nGenerated test file: {s}\n", .{test_file_path});

    // Copy user test file into the generated test file
    if (codegen_options.user_tests.len > 0) {
        const provided_user_tests_path = codegen_options.user_tests;
        const user_tests_path = try std.fmt.allocPrint(allocator, "{s}user_tests.json", .{model_path});
        try utils.copyFile(provided_user_tests_path, user_tests_path);
    }

    try writeModelOptionsFile(model_name, model_path);
}

pub fn writeSlimTestFile(model_name: []const u8, model_path: []const u8) !void {
    // Copy test file template into the generated test file
    const test_file_path = try std.fmt.allocPrint(allocator, "{s}test_{s}.zig", .{ model_path, model_name });

    try utils.copyFile("tests/CodeGen/test_model.slim.template.zig", test_file_path);
    std.log.info("\n\nGenerated test file: {s}\n", .{test_file_path});

    try writeModelOptionsFile(model_name, model_path);
}
