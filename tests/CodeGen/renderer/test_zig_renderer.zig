const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const Renderer = codegen.renderer;
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;

const ZigRenderer = Renderer.ZigRenderer;

test "Arithmetic operations" {
    std.debug.print("Running zig renderer arithmetic test \n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .ADD, .dtype = .f32, .src = &.{ 1, 2 }, .arg = null },
        .{ .id = 1, .op = .SUB, .dtype = .f32, .src = &.{ 3, 4 }, .arg = null },
        .{ .id = 2, .op = .MUL, .dtype = .f32, .src = &.{ 5, 6 }, .arg = null },
        .{ .id = 3, .op = .FDIV, .dtype = .f32, .src = &.{ 7, 8 }, .arg = null },
        .{ .id = 4, .op = .POW, .dtype = .f32, .src = &.{ 9, 10 }, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = @as(f32, t1) + @as(f32, t2);
        \\const t1 = @as(f32, t3) - @as(f32, t4);
        \\const t2 = @as(f32, t5) * @as(f32, t6);
        \\const t3 = @as(f32, t7) / @as(f32, t8);
        \\const t4 = std.math.pow(f32, t9, t10);
        \\
    ;
    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Reduce operations" {
    std.debug.print("Running zig renderer reduce test \n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .REDUCE_ADD, .dtype = .f32, .src = &.{0}, .arg = null },
        .{ .id = 1, .op = .REDUCE_MAX, .dtype = .f32, .src = &.{1}, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const vec0: @Vector(t0.len, f32) = t0;
        \\const t0: f32 = @reduce(.Add, vec0);
        \\const vec1: @Vector(t1.len, f32) = t1;
        \\const t1: f32 = @reduce(.Max, vec1);
        \\
    ;
    const actual = try buffer.toOwnedSlice();

    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Conditional operations" {
    std.debug.print("Running zig renderer conditional test \n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .MAX, .dtype = .f32, .src = &.{ 1, 2 }, .arg = null },
        .{ .id = 1, .op = .MIN, .dtype = .f32, .src = &.{ 3, 4 }, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = if(@as(f32,t1) > @as(f32,t2)) @as(f32,t1) else @as(f32,t2);
        \\const t1 = if(@as(f32,t3) < @as(f32,t4)) @as(f32,t3) else @as(f32,t4);
        \\
    ;
    const actual = try buffer.toOwnedSlice();

    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Unary Operations" {
    std.debug.print("Running zig renderer unary test \n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .EXP2, .dtype = .f32, .src = &.{1}, .arg = null },
        .{ .id = 1, .op = .NEG, .dtype = .f32, .src = &.{2}, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = @as(f32, t1) * @as(f32, t1);
        \\const t1 = @as(f32,-t2);
        \\
    ;
    const actual = try buffer.toOwnedSlice();

    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Standard GEP operation" {
    std.debug.print("Running zig renderer standard GEP test\n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    // Create a standard GEP operation
    var uops = [_]UOp{
        .{
            .id = 0,
            .op = .GEP,
            .dtype = .f32,
            .src = &.{ 1, 2 }, // Base pointer ID and index variable ID
            .arg = .{
                .mem_info = .{
                    .base = 0x1000, // Example base address
                    .offset = 0,
                    .stride = 1,
                },
            },
        },
    };

    try renderer.render(&uops);

    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    std.debug.print("\nBuffer content: {s}\n", .{actual});
    const expected = "const t0 = 4096 + (0 + t1[2] * 1) * @as(usize, @sizeOf(f32));\n";
    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "View-based GEP operation" {
    std.debug.print("Running zig renderer view-based GEP test\n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    // Create VIEW and GEP operations
    var uops = [_]UOp{
        .{
            .id = 1,
            .op = .VIEW,
            .dtype = .f32,
            .src = &.{3}, // Source tensor ID
            .arg = .{
                .view_meta = .{
                    .shape = &.{ 3, 4 }, // 3x4 shape
                    .strides = &.{ 2, 1 }, // Row-major layout
                },
            },
        },
        .{
            .id = 0,
            .op = .GEP,
            .dtype = .f32,
            .src = &.{ 1, 2, 3 }, // View ID and index variable ID
            .arg = .{
                .mem_info = .{
                    .base = 0x1100, // Example base address
                    .offset = 10,
                    .stride = 1,
                },
            },
        },
    };

    try renderer.render(&uops);
    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    const expected = "const t0 = 4352 + (10 + t1[7] * 1) * @as(usize, @sizeOf(f32));\n";
    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Control flow operations" {
    std.debug.print("Running zig renderer range/endrange test \n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .RANGE, .dtype = .f32, .src = &.{0}, .arg = .{ .loop_bounds = .{ .start = 0, .end = 10 } } },
        .{ .id = 1, .op = .ENDRANGE, .dtype = .f32, .src = &.{0}, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\for(@as(f32,0)..@as(f32,10))|t0|{
        \\} //ending range from id t0
        \\
    ;
    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Rendering memory operations" {
    std.debug.print("Running zig renderer memory uops test \n", .{});
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .DEFINE_GLOBAL, .dtype = .f32, .src = &.{0}, .arg = .{ .int = 10 } },
        .{ .id = 1, .op = .LOAD, .dtype = .f32, .src = &.{0}, .arg = null },
        .{ .id = 2, .op = .STORE, .dtype = .f32, .src = &.{ 0, 1 }, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = try allocator.alloc(f32, 10);
        \\defer allocator.free(t0);
        \\const t1 = *t0;
        \\*t0 = t1;
        \\
    ;
    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}
