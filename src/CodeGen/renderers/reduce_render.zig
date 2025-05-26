const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    _ = allocator;
    // Validate operation type

    const supported_ops = [_]UOpType{ .REDUCE_ADD, .REDUCE_MAX };

    if (!std.mem.containsAtLeast(UOpType, &supported_ops, 1, &[_]UOpType{uop.op})) {
        return error.InvalidOperation;
    }

    // Validate operand count
    if (uop.src.len != 1) {
        return error.InvalidOperandCount;
    }

    // Get variable names directly from ptr_map instead of formatting them
    const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
    const input_var = ptr_map.get(uop.src[0]) orelse return error.VariableNotFound;
    const type_str = DTypeInfo.asString(uop.dtype);

    // Operation mapping
    const op_str = switch (uop.op) {
        .REDUCE_ADD => ".Add",
        .REDUCE_MAX => ".Max",
        else => unreachable,
    };

    // Write vector definition and reduction in one step
    try writer.print(
        \\const vec{d}: @Vector({s}.len, {s}) = {s};
        \\const {s}: {s} = @reduce({s}, vec{d});
        \\
    , .{ uop.id, input_var, type_str, input_var, result_var, type_str, op_str, uop.id });
}
