const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const tests_log = std.log.scoped(.test_utils);

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

var models: std.ArrayList([]const u8) = std.ArrayList([]const u8).init(allocator);
var failed_parsed_models: std.ArrayList(ErrorDetail) = std.ArrayList(ErrorDetail).init(allocator);

test " Onnx loader" {
    tests_log.info("\n     test:  Onnx loader\n", .{});

    var dir = try std.fs.cwd().openDir("datasets/models", .{ .iterate = true });
    defer dir.close();

    // Iterate over directory entries
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind == .directory) {
            // Print the directory name
            try models.append(entry.name);
        }
    }

    for (models.items) |model_name| {

        // Format model path according to model_name
        const model_path = try std.mem.concat(allocator, u8, &[_][]const u8{ "datasets/models/", model_name, "/", model_name, ".onnx" });
        defer allocator.free(model_path);

        var model = onnx.parseFromFile(allocator, model_path) catch |err| {
            const errorDetail: ErrorDetail = ErrorDetail{
                .modelName = model_name,
                .errorLoad = err,
            };
            try failed_parsed_models.append(errorDetail);
            continue;
        };
        defer model.deinit(allocator);
        model.print();
    }

    if (failed_parsed_models.items.len != 0) {
        tests_log.info("\n\n FAILED ONNX PARSED MODELS: ", .{});
        for (failed_parsed_models.items) |fm| tests_log.info("\n model:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    } else {
        tests_log.info("\n\n ---- SUCCESFULLY PARSED ALL ONNX MODELS ---- \n\n", .{});
    }

    models.deinit();
    failed_parsed_models.deinit();
}
