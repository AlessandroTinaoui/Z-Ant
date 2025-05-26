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
    // Define supported operations
    const supported_ops = [_]UOpType{ .ADD, .SUB, .MUL, .FDIV, .POW, .MAX, .MIN, .CMPLT };
    // Check if operation is supported
    if (!std.mem.containsAtLeast(UOpType, &supported_ops, 1, &[_]UOpType{uop.op})) {
        return error.InvalidOperation;
    }
    if (uop.src.len != 2) {
        return error.InvalidOperandCount;
    }
    // Get variable names from ptr_map
    const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
    const lhs_var = ptr_map.get(uop.src[0]) orelse return error.VariableNotFound;
    const rhs_var = ptr_map.get(uop.src[1]) orelse return error.VariableNotFound;

    // --- Check if LHS is an accumulator ---
    const is_accumulator_update = std.mem.startsWith(u8, lhs_var, "acc_");

    // Generate code based on whether it's an accumulator update or standard assignment
    switch (uop.op) {
        .ADD => if (is_accumulator_update) {
            try writer.print(" {s} += {s}; // ADD into Accumulator (uop {d})\n", .{ lhs_var, rhs_var, uop.id });
        } else {
            try writer.print(" const {s} = {s} + {s}; // ADD (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id });
        },
        .SUB => if (is_accumulator_update) {
            // Note: Subtraction into accumulator might be less common
            try writer.print(" {s} -= {s}; // SUB into Accumulator (uop {d})\n", .{ lhs_var, rhs_var, uop.id });
        } else {
            try writer.print(" const {s} = {s} - {s}; // SUB (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id });
        },
        .MUL => if (is_accumulator_update) {
            try writer.print(" {s} *= {s}; // MUL into Accumulator (uop {d})\n", .{ lhs_var, rhs_var, uop.id });
        } else {
            try writer.print(" const {s} = {s} * {s}; // MUL (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id });
        },
        // FDIV and POW are unlikely accumulator ops, handle as standard assignment
        .FDIV => try writer.print(" const {s} = {s} / {s}; // FDIV (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .POW => {
            const type_str = DTypeInfo.asString(uop.dtype);
            try writer.print(" const {s} = std.math.pow({s}, {s}, {s}); // POW (uop {d})\n", .{ result_var, type_str, lhs_var, rhs_var, uop.id });
        },
        .MAX => if (is_accumulator_update) {
            try writer.print(" {s} = @max({s}, {s}); // MAX into Accumulator (uop {d})\n", .{ lhs_var, lhs_var, rhs_var, uop.id });
        } else {
            try writer.print(" const {s} = @max({s}, {s}); // MAX (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id });
        },
        .MIN => if (is_accumulator_update) {
            try writer.print(" {s} = @min({s}, {s}); // MIN into Accumulator (uop {d})\n", .{ lhs_var, lhs_var, rhs_var, uop.id });
        } else {
            try writer.print(" const {s} = @min({s}, {s}); // MIN (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id });
        },
        // CMPLT always produces a new boolean result, not an accumulator update
        .CMPLT => try writer.print(" const {s} = ({s} < {s}); // CMPLT (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        else => unreachable,
    }
}
