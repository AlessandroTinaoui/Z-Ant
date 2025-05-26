const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    @compileLog("Warning: 'Conditional' operations are not implemented yet. 'IF ENDIF'");

    if (uop.op != .MAX and uop.op != .MIN) {
        return error.InvalidOperation;
    }

    //minimal Max and min are only attainable from two variables, not more or less.
    if (uop.src.len != 2) {
        return error.InvalidOperandCount;
    }

    const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
    defer allocator.free(result_var);

    const first_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    defer allocator.free(first_var);

    const second_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[1]});
    defer allocator.free(second_var);

    const type_str = DTypeInfo.asString(uop.dtype);

    switch (uop.op) {
        .MAX => try writer.print("const {s} = if(@as({s},{s}) > @as({s},{s})) @as({s},{s}) else @as({s},{s});\n", .{ result_var, type_str, first_var, type_str, second_var, type_str, first_var, type_str, second_var }),
        .MIN => try writer.print("const {s} = if(@as({s},{s}) < @as({s},{s})) @as({s},{s}) else @as({s},{s});\n", .{ result_var, type_str, first_var, type_str, second_var, type_str, first_var, type_str, second_var }),
        else => unreachable,
    }
}
