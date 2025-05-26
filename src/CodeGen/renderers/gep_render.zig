const std = @import("std");
const zant = @import("zant");

const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;
const ViewInfo = @import("view_manager.zig").ViewInfo;
const BufferInfo = @import("zig_renderer.zig").BufferInfo;

/// Error types for the renderer operations
pub const RendererError = error{
    InvalidOp,
    NoAny,
    VarMissing,
    RankMismatch,
    OutOfMemory,
};

/// Indexing and offset calculation utilities
const IndexHelper = struct {
    /// Cast loop index to usize expression string
    fn castIndex(
        alloc: std.mem.Allocator,
        ptr_map: *const std.AutoHashMap(usize, []const u8),
        id: usize,
    ) ![]const u8 {
        const name = ptr_map.get(id) orelse return RendererError.VarMissing;
        return std.fmt.allocPrint(alloc, "@as(usize,@intCast({s}))", .{name});
    }

    /// Emit "expr * stride" if stride != 0
    fn emitTerm(w: anytype, expr: []const u8, stride: isize, first: *bool) !void {
        if (stride == 0) return;
        if (!first.*) try w.print(" + ", .{});
        try w.print("({s}*{d})", .{ expr, stride });
        first.* = false;
    }

    /// Build linear index expression from view shape and strides
    fn buildLinearIndexExpr(
        w: anytype,
        temp_alloc: std.mem.Allocator,
        linear_idx_expr: []const u8,
        shape: []const usize,
        strides: []const isize,
    ) !void {
        if (shape.len == 0) return w.print("0", .{});
        if (shape.len != strides.len) return RendererError.RankMismatch;

        try w.print("(", .{});
        var first_term = true;
        var divisor: []const u8 = "1";

        var i = shape.len - 1;
        while (true) {
            const dim_size = shape[i];
            const stride = strides[i];

            if (stride != 0) {
                var idx_expr: []const u8 = undefined;
                var buf: [256]u8 = undefined;

                idx_expr = if (dim_size == 1) "0" else blk: {
                    if (std.mem.eql(u8, divisor, "1")) {
                        break :blk try std.fmt.bufPrint(&buf, "({s} % {d})", .{ linear_idx_expr, dim_size });
                    } else {
                        break :blk try std.fmt.bufPrint(&buf, "(({s} / {s}) % {d})", .{ linear_idx_expr, divisor, dim_size });
                    }
                };

                try emitTerm(w, idx_expr, stride, &first_term);
            }

            if (i == 0) break;
            i -= 1;

            if (dim_size > 1) {
                divisor = if (std.mem.eql(u8, divisor, "1"))
                    try std.fmt.allocPrint(temp_alloc, "{d}", .{dim_size})
                else
                    try std.fmt.allocPrint(temp_alloc, "({s} * {d})", .{ divisor, dim_size });
            }
        }

        if (first_term) try w.print("0", .{});
        try w.print(")", .{});
    }

    /// Build offset expression for multidimensional index
    fn buildMultiDimOffsetExpr(
        w: anytype,
        alloc: std.mem.Allocator,
        ptr_map: *const std.AutoHashMap(usize, []const u8),
        indices: []const usize,
        strides: []const isize,
    ) !void {
        if (indices.len - 1 != strides.len) return w.print("0", .{});

        var first = true;
        for (indices[1..], 0..) |idx_id, axis| {
            const idx_expr = try castIndex(alloc, ptr_map, idx_id);
            defer alloc.free(idx_expr);
            try emitTerm(w, idx_expr, strides[axis], &first);
        }

        if (first) try w.print("0", .{});
    }
};

/// Buffer and variable naming utilities
const NameHelper = struct {
    /// Get base variable name for buffer access
    fn getBaseVarName(
        alloc: std.mem.Allocator,
        base_id: usize,
        buffer_map: *const std.AutoHashMap(usize, BufferInfo),
    ) ![]const u8 {
        const bi = buffer_map.get(base_id) orelse {
            std.debug.print("getBaseVarName: Error - base_id {d} not found\n", .{base_id});
            return RendererError.VarMissing;
        };

        return if (bi.is_input)
            std.fmt.allocPrint(alloc, "input_{d}", .{base_id})
        else
            std.fmt.allocPrint(alloc, "{s}", .{bi.name});
    }
};

/// Handles different offset calculation strategies based on source type
const OffsetBuilder = struct {
    /// Process a view-based offset calculation
    fn handleViewOffset(
        writer: anytype,
        temp_alloc: std.mem.Allocator,
        uop: UOp,
        vinfo: ViewInfo,
        base_id: usize,
        ptr_map: *const std.AutoHashMap(usize, []const u8),
    ) !void {
        const view_strides = vinfo.arg.view_meta.strides;
        const view_shape = vinfo.arg.view_meta.shape;

        if (view_shape.len != view_strides.len or view_shape.len == 0) {
            std.debug.print("GEP RankMismatch: View {d} shape/strides issue\n", .{base_id});
            return RendererError.RankMismatch;
        }

        if (uop.src.len == 1) {
            try writer.print("0", .{});
        } else if (uop.src.len == 2) {
            const idx_expr = try IndexHelper.castIndex(temp_alloc, ptr_map, uop.src[1]);
            defer temp_alloc.free(idx_expr);
            try IndexHelper.buildLinearIndexExpr(writer, temp_alloc, idx_expr, view_shape, view_strides);
        } else if (uop.src.len == view_strides.len + 1) {
            std.debug.print("Strides: {any}", .{view_strides});
            try IndexHelper.buildMultiDimOffsetExpr(writer, temp_alloc, ptr_map, uop.src, view_strides);
        } else {
            std.debug.print("GEP RankMismatch: {d} indices for view with {d} dims\n", .{ uop.src.len - 1, view_strides.len });
            return RendererError.RankMismatch;
        }
    }

    /// Process a buffer-based offset calculation
    fn handleBufferOffset(
        writer: anytype,
        temp_alloc: std.mem.Allocator,
        uop: UOp,
        buf_info: BufferInfo,
        base_id: usize,
        ptr_map: *const std.AutoHashMap(usize, []const u8),
    ) !void {
        const buffer_shape = buf_info.shape;

        if (uop.src.len == 1) {
            try writer.print("0", .{});
        } else if (uop.src.len == 2 and buffer_shape.len > 0) {
            const idx_expr = try IndexHelper.castIndex(temp_alloc, ptr_map, uop.src[1]);
            defer temp_alloc.free(idx_expr);
            try writer.print("{s}", .{idx_expr});
        } else if (uop.src.len == buffer_shape.len + 1 and buffer_shape.len > 0) {
            var strides = try temp_alloc.alloc(isize, buffer_shape.len);
            strides[buffer_shape.len - 1] = 1;

            var i = buffer_shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * @as(isize, @intCast(buffer_shape[i]));
            }

            try IndexHelper.buildMultiDimOffsetExpr(writer, temp_alloc, ptr_map, uop.src, strides);
        } else if (buffer_shape.len == 0 and uop.src.len > 1) {
            std.debug.print("GEP Error: Indexing scalar buffer {d}\n", .{base_id});
            return RendererError.RankMismatch;
        } else {
            std.debug.print("GEP RankMismatch: {d} indices for buffer with {d} dims\n", .{ uop.src.len - 1, buffer_shape.len });
            return RendererError.RankMismatch;
        }
    }
};

/// Render GEP operations
pub fn render(
    alloc: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    view_map: *std.AutoHashMap(usize, ViewInfo),
    buffer_map: *const std.AutoHashMap(usize, BufferInfo),
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    if (uop.op != .GEP) return RendererError.InvalidOp;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp_alloc = arena.allocator();

    const result_name = ptr_map.get(uop.id) orelse return RendererError.VarMissing;
    const base_id = uop.src[0];

    var base_var_name: []const u8 = undefined;
    var offset_list = std.ArrayList(u8).init(temp_alloc);
    const offset_writer = offset_list.writer();

    // Calculate the offset based on source type
    if (view_map.get(base_id)) |vinfo| {
        base_var_name = try NameHelper.getBaseVarName(temp_alloc, vinfo.src[0], buffer_map);
        try OffsetBuilder.handleViewOffset(offset_writer, temp_alloc, uop, vinfo, base_id, ptr_map);
    } else if (buffer_map.get(base_id)) |buf_info| {
        base_var_name = try NameHelper.getBaseVarName(temp_alloc, base_id, buffer_map);
        try OffsetBuilder.handleBufferOffset(offset_writer, temp_alloc, uop, buf_info, base_id, ptr_map);
    } else {
        std.debug.print("GEP Error: Base ID {d} not found\n", .{base_id});
        return RendererError.VarMissing;
    }

    // Render the final GEP instruction
    const offset_expr = try offset_list.toOwnedSlice();
    const dtype_name = DTypeInfo.asString(uop.dtype);

    try writer.print(
        "    const {s} = @intFromPtr({s}.ptr) + ({s})*@sizeOf({s}); // GEP id={d}\n",
        .{ result_name, base_var_name, offset_expr, dtype_name, uop.id },
    );
}
