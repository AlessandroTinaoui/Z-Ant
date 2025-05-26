//! The aim of this file is that, given a onnx oprator (https://onnx.ai/onnx/operators/),
//! it returns the onnx.ModelProto containing only one node for the operation

const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const onnx = zant.onnx;
const codeGen = @import("codegen");

const tests_log = std.log.scoped(.test_oneOP);

pub fn main() !void {
    tests_log.info("One ONNX Operator Model Generator", .{});

    tests_log.info("\n     opening available_operations...", .{});
    const op_file = try std.fs.cwd().openFile("tests/CodeGen/Python-ONNX/available_operations.txt", .{});
    defer op_file.close();
    tests_log.info(" done", .{});

    const file_size = try op_file.getEndPos();
    const buffer = try allocator.alloc(u8, @intCast(file_size));

    const bytes_read = try op_file.readAll(buffer);
    if (bytes_read != file_size) {
        return error.UnexpectedEOF;
    }

    // Read the available operations from file.
    const ops_data = buffer[0..file_size];
    defer allocator.free(ops_data);

    // Split file into lines.
    var lines_iter = std.mem.splitAny(u8, ops_data, "\n");

    // Create folders if not exist
    try std.fs.cwd().makePath("generated/oneOpModels/");

    // There will be a file named test_oneop_models.zig that will contain all the tests for the one operation models.
    // So to test all the models, you can just run the global file

    // Create test_oneop_models.zig
    const test_oneop_file = try std.fs.cwd().createFile("generated/oneOpModels/test_oneop_models.zig", .{});
    defer test_oneop_file.close();

    const test_oneop_writer = test_oneop_file.writer();

    try test_oneop_writer.writeAll("const std = @import(\"std\");\n");
    try test_oneop_writer.writeAll("\n");
    try test_oneop_writer.writeAll("test {");
    try test_oneop_writer.writeAll("\n");

    while (true) {
        // Get the next line from the iterator.
        const maybe_line = lines_iter.next();

        if (maybe_line) |ml| tests_log.info("maybe_line: {any}\n", .{ml}) else {
            tests_log.info("maybe_line: null -----> break\n", .{});
            break;
        }

        const raw_line = maybe_line.?;
        // Trim whitespace from the line.
        const trimmed_line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (trimmed_line.len > 0) {
            tests_log.info("Operation: {s}\n", .{trimmed_line});
        }

        // Construct the model file path: "Phython-ONNX/{op}_0.onnx"
        const model_path = try std.fmt.allocPrint(allocator, "datasets/oneOpModels/{s}_0.onnx", .{trimmed_line});
        defer allocator.free(model_path);
        tests_log.info("model_path : {s}", .{model_path});

        // Load the model.
        var model = try onnx.parseFromFile(allocator, model_path);

        //Printing the model:
        //DEBUG
        model.print();

        tests_log.info("\n CODEGENERATING {s} ...", .{model_path});

        // Create the generated model directory if not present
        const generated_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/", .{trimmed_line});
        defer allocator.free(generated_path);
        try std.fs.cwd().makePath(generated_path);

        // CORE PART -------------------------------------------------------
        // ONNX model parsing
        try codeGen.globals.setGlobalAttributes(model);

        // Create the code for the model
        try codeGen.skeleton.writeZigFile(trimmed_line, generated_path, model, false);

        // Create relative tests
        try codeGen.tests.writeSlimTestFile(trimmed_line, generated_path);

        // Copy user test file into the generated test file
        const dataset_test_model_path = try std.fmt.allocPrint(allocator, "datasets/oneOpModels/{s}_0_user_tests.json", .{trimmed_line});
        defer allocator.free(dataset_test_model_path);

        const generated_test_model_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/user_tests.json", .{trimmed_line});
        defer allocator.free(generated_test_model_path);

        try codeGen.utils.copyFile(dataset_test_model_path, generated_test_model_path);
        tests_log.info("Written user test for {s}", .{trimmed_line});

        // Add relative one op test to global tests file
        try test_oneop_writer.print("\t _ = @import(\"{s}/test_{s}.zig\"); \n", .{ trimmed_line, trimmed_line });

        //try codeGen.globals.setGlobalAttributes(model);
        model.deinit(allocator);
    }

    // Adding last global test line
    try test_oneop_writer.writeAll("} \n\n");
}
