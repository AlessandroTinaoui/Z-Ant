const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32 {
    const output_2 = try allocator.alloc(f32, 24);
var idx_3: i32 = 0; // RANGE (uop 3)
while (idx_3 < 24) : (idx_3 += 1) {
        const addr_4 = @intFromPtr(input_0.ptr) + ((((@as(usize,@intCast(idx_3)) % 4)*1) + (((@as(usize,@intCast(idx_3)) / 4) % 3)*4) + (((@as(usize,@intCast(idx_3)) / (4 * 3)) % 2)*12)))*@sizeOf(f32); // GEP id=4
        const addr_5 = @intFromPtr(input_1.ptr) + (((((@as(usize,@intCast(idx_3)) / 4) % 3)*1)))*@sizeOf(f32); // GEP id=5
        var buf_6: f32 = @as(*const f32, @ptrFromInt(addr_4)).*; // LOAD (uop 6)
    _ = &buf_6;
        var buf_7: f32 = @as(*const f32, @ptrFromInt(addr_5)).*; // LOAD (uop 7)
    _ = &buf_7;
     const buf_8 = buf_6 + buf_7; // ADD (uop 8)
        const addr_9 = @intFromPtr(output_2.ptr) + (@as(usize,@intCast(idx_3)))*@sizeOf(f32); // GEP id=9
        @as(*f32, @ptrFromInt(addr_9)).* = buf_8; // STORE (uop 10)
}
    return output_2;
}
