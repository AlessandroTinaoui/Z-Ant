const std = @import("std");
const utils = @import("utils.zig");
const ColorChannels = utils.ColorChannels;
const ImToTensorError = utils.ImToTensorError;

const bmp_header_size = 14; // BITMAPFILEHEADER
const dib_header_size = 40; // BITMAPINFOHEADER
const bytes_per_pixel = 3; // always 3 bytes per pixel (RGB)

pub fn writeBmp(
    channels: ColorChannels,
    path: []const u8,
    colorspace: usize,
) !void {
    // ---------- output path --------------
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    const ext = std.fs.path.extension(path); // ".jpg" / ".jpeg"
    const base_len = path.len - ext.len;

    const suffix = switch (colorspace) {
        0 => "_rgb",
        1 => "_ycbcr",
        2 => "_gray",
        else => return ImToTensorError.InvalidColorspace,
    };

    if (base_len + suffix.len + 4 > path_buf.len)
        return ImToTensorError.NameTooLong;

    @memcpy(path_buf[0..base_len], path[0..base_len]);
    @memcpy(path_buf[base_len..][0..suffix.len], suffix);
    @memcpy(path_buf[base_len + suffix.len ..][0..4], ".bmp");
    const bmp_path = path_buf[0 .. base_len + suffix.len + 4];

    // ------ Image data ------
    const width = channels.width;
    const height = channels.height;
    const row_raw = width * bytes_per_pixel;
    const row_stride = @as(u32, (row_raw + 3) & ~@as(u32, 3));
    const pixel_array = row_stride * height;

    const file_size = bmp_header_size + dib_header_size + pixel_array;

    // Open the file for writing
    var file = try std.fs.cwd().createFile(bmp_path, .{});
    defer file.close();
    var writer = file.writer();

    // --- BITMAPFILEHEADER (14 byte) ---
    try writer.writeByte('B');
    try writer.writeByte('M');
    try writer.writeInt(u32, file_size, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, bmp_header_size + dib_header_size, .little);

    // --- BITMAPINFOHEADER (40 byte) ---
    try writer.writeInt(u32, dib_header_size, .little);
    try writer.writeInt(i32, @as(i32, width), .little);
    try writer.writeInt(i32, @as(i32, height), .little);
    try writer.writeInt(u16, 1, .little);
    try writer.writeInt(u16, bytes_per_pixel * 8, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, pixel_array, .little);
    try writer.writeInt(i32, 2835, .little);
    try writer.writeInt(i32, 2835, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, 0, .little);

    // --- Pixel data ---
    var pad_buf: [3]u8 = [_]u8{ 0, 0, 0 };
    const padding = row_stride - row_raw;

    for (0..height) |row| {
        const flipped_row = height - 1 - row;
        var idx: usize = flipped_row * width;

        for (0..width) |_| {
            const y_val = @as(u8, @intCast(channels.ch1[idx]));

            const rgb: [3]u8 = if (colorspace == 2 or channels.component_num == 1)
                .{ y_val, y_val, y_val } // grayscale
            else
                .{ @as(u8, @intCast(channels.ch3[idx])), @as(u8, @intCast(channels.ch2[idx])), y_val }; // B-G-R

            try writer.writeAll(&rgb);
            idx += 1;
        }

        if (padding != 0)
            try writer.writeAll(pad_buf[0..padding]);
    }
}
