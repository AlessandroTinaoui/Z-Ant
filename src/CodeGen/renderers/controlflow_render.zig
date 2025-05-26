const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const DTypeInfo = zant.uops.DTypeInfo;
const ViewInfo = @import("view_manager.zig").ViewInfo;

pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
    view_map: *std.AutoHashMap(usize, ViewInfo),
) !void {
    // Discard allocator if not needed
    const ops = [_]UOpType{ .RANGE, .ENDRANGE };
    // Check if operation is supported
    if (!std.mem.containsAtLeast(UOpType, &ops, 1, &[_]UOpType{uop.op})) {
        return error.InvalidOperation;
    }

    if ((uop.op == .RANGE and uop.src.len != 0) or
        (uop.op == .ENDRANGE and uop.src.len != 1))
    {
        return error.InvalidOperandCount;
    }

    switch (uop.op) {
        .RANGE => {
            switch (uop.arg.?) {
                .loop_bounds => {
                    const start = uop.arg.?.loop_bounds.start;
                    const end = uop.arg.?.loop_bounds.end;
                    const index_name = ptr_map.get(uop.id) orelse return error.VariableNotFound;
                    try writer.print("var {s}: {s} = {d}; // RANGE (uop {d})\nwhile ({s} < {d}) : ({s} += 1) {{\n", .{ index_name, DTypeInfo.asString(uop.dtype), start, uop.id, index_name, end, index_name });
                },
                .loop_bounds_view => {
                    const start = uop.arg.?.loop_bounds_view.start;
                    const view_id = uop.arg.?.loop_bounds_view.view_id;
                    const view = view_map.get(view_id) orelse return error.VariableNotFound;
                    const strides_len = view.arg.view_meta.strides.len;
                    for (0..strides_len) |i| {
                        const name = ptr_map.get(uop.id) orelse return error.VariableNotFound;
                        const index_name = try std.fmt.allocPrint(allocator, "{s}_{d}", .{ name, i });
                        defer allocator.free(index_name);
                        const stride = view.arg.view_meta.strides[i];
                        try writer.print("var {s}: {s} = {d}; // RANGE (uop {d})\nwhile ({s} < {d}) : ({s} += 1) {{\n", .{ index_name, DTypeInfo.asString(uop.dtype), start, uop.id, index_name, stride, index_name });
                    }
                },
                else => unreachable,
            }
        },
        .ENDRANGE => {
            if (uop.arg == null) {
                try writer.print("}}\n", .{});
            } else {
                switch (uop.arg.?) {
                    .loop_bounds => {
                        try writer.print("}}\n", .{});
                    },
                    .loop_bounds_view => {
                        const view_id = uop.arg.?.loop_bounds_view.view_id;
                        const view = view_map.get(view_id) orelse return error.VariableNotFound;
                        const strides_len = view.arg.view_meta.strides.len;
                        for (0..strides_len) |_| {
                            try writer.print("}}\n", .{});
                        }
                    },
                    else => unreachable,
                }
            }
        },
        else => unreachable,
    }
}
