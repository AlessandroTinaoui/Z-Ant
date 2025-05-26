const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

const NodeZant = IR_zant.NodeZant;
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor;

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

var models: std.ArrayList([]const u8) = std.ArrayList([]const u8).init(allocator);
var failed_parsed_models: std.ArrayList(ErrorDetail) = std.ArrayList(ErrorDetail).init(allocator);

test "linearizing mnist-8 " {
    std.debug.print("\n\n ------TEST: linearizing mnist-8 ", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
    defer graphZant.deinit();

    const linearizedGraph = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit();

    std.debug.print("\n\nLinearized Graph Nodes:\n", .{});
    for (linearizedGraph.items) |node| {
        std.debug.print(" - Node: {s}\n", .{node.nodeProto.name orelse "<unnamed>"});
    }
}

// test "linearizing mnist-1 " {
//     std.debug.print("\n\n ------TEST: linearizing mnist-1 ", .{});

//     var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-1/mnist-1.onnx");
//     defer model.deinit(allocator);

//     // model.print();

//     var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
//     defer graphZant.deinit();

//     const linearizedGraph = try graphZant.linearize(allocator);

//     std.debug.print("\n\nLinearized Graph Nodes:\n", .{});
//     for (linearizedGraph.items) |node| {
//         std.debug.print(" - Node: {s}\n", .{node.nodeProto.name orelse "<unnamed>"});
//     }

//     linearizedGraph.deinit();
// }

// test "linearize all datasets/models " {
//     std.debug.print("\n     test: linearize all datasets/models\n", .{});

//     var dir = try std.fs.cwd().openDir("datasets/models", .{ .iterate = true });
//     defer dir.close();

//     // Iterate over directory entries
//     var it = dir.iterate();
//     while (try it.next()) |entry| {
//         if (entry.kind == .directory) {
//             // Print the directory name
//             try models.append(entry.name);
//         }
//     }

//     std.debug.print("\n -- iterating on models: ", .{});
//     for (models.items) |model_name| {
//         if (std.mem.eql(u8, "best", model_name)) continue;
//         std.debug.print("\n  --- {s}", .{model_name});

//         // Format model path according to model_name
//         const model_path = try std.mem.concat(allocator, u8, &[_][]const u8{ "datasets/models/", model_name, "/", model_name, ".onnx" });
//         defer allocator.free(model_path);

//         var model = try onnx.parseFromFile(allocator, model_path);
//         defer model.deinit(allocator);

//         model.print();

//         var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
//         defer graphZant.deinit();

//         const linearizedGraph = try graphZant.linearize(allocator);

//         std.debug.print("\n\nLinearized Graph Nodes:\n", .{});
//         for (linearizedGraph.items) |node| {
//             std.debug.print(" - Node: {s}\n", .{node.nodeProto.name orelse "<unnamed>"});
//         }

//         linearizedGraph.deinit();
//     }

//     if (failed_parsed_models.items.len != 0) {
//         std.debug.print("\n\n FAILED ONNX PARSED MODELS: ", .{});
//         for (failed_parsed_models.items) |fm| std.debug.print("\n model:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
//     } else {
//         std.debug.print("\n\n ---- SUCCESFULLY PARSED ALL ONNX MODELS ---- \n\n", .{});
//     }

//     models.deinit();
//     failed_parsed_models.deinit();
// }
