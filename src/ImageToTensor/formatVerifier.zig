const std = @import("std");
const ImToTensorError = @import("utils.zig").ImToTensorError;

pub const ImageFormat = enum {
    PNG,
    JPEG,
};

// this file is used to verify the format of the image
// the format is verified by checking the first bytes of the image that contain the format identifier
pub fn verifyFormat(buf: []u8, format: ImageFormat) bool {
    if (format == ImageFormat.JPEG) {
        const jpegHdr = [_]u8{ 0xFF, 0xD8 };
        return buf.len >= 2 and std.mem.eql(u8, buf[0..2], &jpegHdr);
    }
    //else if (format == PNG)...
    else {
        return false;
    }
}

pub fn findFormat(buf: []u8) !ImageFormat {
    if (verifyFormat(buf, ImageFormat.JPEG)) {
        return ImageFormat.JPEG;
    }
    //else if (verifyFormat(buf, ImageFormat.PNG)) {
    //    return ImageFormat.PNG;
    //}
    else {
        return ImToTensorError.InvalidImageFormat;
    }
}
