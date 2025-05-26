const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const DType = zant.uops.DType;
const DTypeInfo = zant.uops.DTypeInfo;

const RenderError = error{
    InvalidOperation,
    InvalidOperandCount,
    VariableNotFound,
    NoAnyProvided,
    UnsupportedDType,
};

/// Helper function to get a variable name from the pointer map
fn getVar(ptr_map: *const std.AutoHashMap(usize, []const u8), id: usize) ![]const u8 {
    return ptr_map.get(id) orelse return RenderError.VariableNotFound;
}

/// Format value based on data type
fn formatValue(buf: []u8, dtype: DType, arg: anytype) ![]const u8 {
    return switch (dtype) {
        .f32 => std.fmt.bufPrint(buf, "{e}", .{arg.float}),
        .i32 => std.fmt.bufPrint(buf, "{d}", .{arg.int}),
        else => return RenderError.UnsupportedDType,
    };
}

/// Validates operation type and operand count
fn validateOp(uop: UOp) !void {
    const valid_ops = [_]UOpType{ .DEFINE_GLOBAL, .LOAD, .STORE, .CONST };

    // Check if operation type is valid
    var valid = false;
    for (valid_ops) |op| {
        if (uop.op == op) {
            valid = true;
            break;
        }
    }
    if (!valid) return RenderError.InvalidOperation;

    // Check operand count
    switch (uop.op) {
        .DEFINE_GLOBAL => if (uop.src.len != 0) return RenderError.InvalidOperandCount,
        .STORE => if (uop.src.len != 2) return RenderError.InvalidOperandCount,
        .LOAD => if (uop.src.len != 1) return RenderError.InvalidOperandCount,
        .CONST => {}, // No specific operand count check needed
        else => unreachable,
    }
}
pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    _ = allocator; // Unused parameter

    try validateOp(uop);

    const type_str = DTypeInfo.asString(uop.dtype);

    switch (uop.op) {
        .DEFINE_GLOBAL => return RenderError.InvalidOperation,

        .STORE => {
            const gep_addr_var = try getVar(ptr_map, uop.src[0]);
            const value_var = try getVar(ptr_map, uop.src[1]);

            try writer.print("    @as(*{s}, @ptrFromInt({s})).* = {s}; // STORE (uop {d})\n", .{ type_str, gep_addr_var, value_var, uop.id });
        },

        .LOAD => {
            const gep_addr_var = try getVar(ptr_map, uop.src[0]);
            const result_var = try getVar(ptr_map, uop.id);

            try writer.print("    var {s}: {s} = @as(*const {s}, @ptrFromInt({s})).*; // LOAD (uop {d})\n" ++
                "    _ = &{s};\n", .{ result_var, type_str, type_str, gep_addr_var, uop.id, result_var });
        },

        .CONST => {
            const arg = uop.arg orelse return RenderError.NoAnyProvided;
            const result_var = try getVar(ptr_map, uop.id);

            var val_str_buf: [128]u8 = undefined;
            var final_val_str: []const u8 = undefined;

            switch (uop.dtype) {
                .f32 => {
                    const f_val = arg.float;
                    if (f_val == std.math.inf(f32)) {
                        final_val_str = "std.math.inf(f32)";
                    } else if (f_val == -std.math.inf(f32)) {
                        final_val_str = "-std.math.inf(f32)";
                    } else {
                        // Use standard formatting for non-infinity floats
                        final_val_str = try std.fmt.bufPrint(&val_str_buf, "{e}", .{f_val});
                    }
                },
                .i32 => {
                    final_val_str = try std.fmt.bufPrint(&val_str_buf, "{d}", .{arg.int});
                },
                .bool => {
                    // Use Zig's native true/false literals
                    final_val_str = if (arg.bool) "true" else "false";
                },
                else => return RenderError.UnsupportedDType,
            }

            try writer.print("    var {s}: {s} = {s}; // CONST (uop {d})\n" ++
                "    _ = &{s};\n", .{ result_var, type_str, final_val_str, uop.id, result_var });
        },

        else => unreachable,
    }
}
