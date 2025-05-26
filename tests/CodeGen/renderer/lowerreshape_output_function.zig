const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32 {
    const output_2 = try allocator.alloc(f32, 6);
// SHAPE: id=0 src=0 -> shape { 2, 3 }
var idx_4: i32 = 0; // RANGE (uop 4)
while (idx_4 < 2) : (idx_4 += 1) {
    var idx_5: i32 = 0; // RANGE (uop 5)
while (idx_5 < 3) : (idx_5 += 1) {
            const addr_6 = @intFromPtr(input_0.ptr) + ((@as(usize,@intCast(idx_4))*3) + (@as(usize,@intCast(idx_5))*1))*@sizeOf(f32); // GEP id=6
            var buf_7: f32 = @as(*const f32, @ptrFromInt(addr_6)).*; // LOAD (uop 7)
    _ = &buf_7;
            const addr_8 = @intFromPtr(output_2.ptr) + ((@as(usize,@intCast(idx_4))*3) + (@as(usize,@intCast(idx_5))*1))*@sizeOf(f32); // GEP id=8
            @as(*f32, @ptrFromInt(addr_8)).* = buf_7; // STORE (uop 9)
    }
}
    return output_2;
}
