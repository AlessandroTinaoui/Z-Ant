const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const UopBuilder = zant.uops.UOpBuilder;
const lower_math_op = zant.codegen.lower_math_handler.render_lower_math_op;
const Tensor = zant.core.tensor.Tensor;
const Renderer = zant.codegen.renderer;

test "mnsist 8 render" {
    std.debug.print("\n\n ------TEST: parsing mnist-8 graphZant", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: zant.IR_graph.GraphZant = try zant.IR_graph.init(&model);
    defer graphZant.deinit();

    var builder = UopBuilder.init();
    defer builder.deinit();

    for (graphZant.nodes) |node| {
        try lower_math_op(&builder, &node);
    }

    const uops_list = try builder.toOwnedSlice();
    defer allocator.free(uops_list);

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/multiple_ops.zig"; // Save inside tests dir
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true }); // Ensure write permissions
    defer file.close();
    const Writer = @TypeOf(file.writer());

    var renderer = Renderer.ZigRenderer(Writer).init(allocator, file.writer());
    defer renderer.deinit(); // Deinit renderer AFTER use

    // Specify which IDs are inputs
    const input_ids = &[_]usize{ 0, 1 };
    try renderer.render_as_function(uops_list, input_ids); // Call the new method

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
}
