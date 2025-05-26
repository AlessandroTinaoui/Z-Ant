const std = @import("std");
const zant = @import("zant");
const ViewInfo = @import("view_manager.zig").ViewInfo;
const BufferInfo = @import("zig_renderer.zig").BufferInfo;
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub const RendererError = error{
    InvalidOp,
    NoAny,
    VarMissing,
    RankMismatch,
    OutOfMemory,
};

/// Main entry point for rendering GEP operations
pub fn render(
    alloc: std.mem.Allocator,
    _: anytype,
    uop: UOp,
    view_map: *std.AutoHashMap(usize, ViewInfo),
    _: *const std.AutoHashMap(usize, BufferInfo),
    _: *const std.AutoHashMap(usize, []const u8),
) !void {
    if (uop.op != .RESHAPE) return RendererError.InvalidOp;
    if (uop.src.len != 1) return RendererError.InvalidOp;
    if (uop.arg == null) return RendererError.NoAny;

    const shape = uop.arg.?.shape;
    if (shape.len == 0) return RendererError.RankMismatch;

    // Calculate strides

    var strides = std.ArrayList(isize).init(alloc);
    defer strides.deinit();

    try strides.resize(shape.len);

    var stride: isize = 1;
    var i: usize = shape.len;
    while (i > 0) : (i -= 1) {
        strides.items[i - 1] = stride;
        stride *= @intCast(shape[i - 1]);
    }

    const src_id = uop.src[0];

    if (view_map.get(src_id)) |vinfo| {
        var view_info = vinfo;
        view_info.arg.view_meta.strides = if (strides.items.len > 0) try alloc.dupe(isize, strides.items) else &.{1};
        try view_map.put(src_id, view_info);
        return;
    }

    // Create output view and transfer ownership of strides
    // Directly use strides without duplicating, since free management happens in the test
    const out_view = ViewInfo{
        .dtype = uop.dtype,
        .src = &.{src_id},
        .arg = .{
            .view_meta = .{
                .shape = shape,
                .strides = if (strides.items.len > 0) try alloc.dupe(isize, strides.items) else &.{1},
            },
        },
    };

    // Store the new view in the view map - the strides ownership is now transferred
    try view_map.put(uop.id, out_view);
    // // Render new strides
    // try writer.print("    const stride_{d}: []const isize = &.{{", .{src_id});
    // for (strides) |stride_num| try writer.print("{d},", .{stride_num});
    // try writer.print("}};", .{});
}
