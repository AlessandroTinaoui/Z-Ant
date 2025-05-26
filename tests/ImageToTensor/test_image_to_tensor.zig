const std = @import("std");
const zant = @import("zant");
const testing = std.testing;
const utils = zant.ImageToTensor.utils;
const ImageToTensor = zant.ImageToTensor;
const pkgAllocator = zant.utils.allocator;

test "verify output shape" {
    std.debug.print("start test: verify output shape\n", .{});
    const allocator = pkgAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/baseline.jpg";
    const imageRgb = try ImageToTensor.imageToRGB(allocator, image_path, .normalizeSigned, f32);
    defer imageRgb.deinit();

    const imageYCbCr = try ImageToTensor.imageToYCbCr(allocator, image_path, .normalizeSigned, f32);
    defer imageYCbCr.deinit();

    const imageGrayscale = try ImageToTensor.imageToGray(allocator, image_path, .normalizeSigned, f32);
    defer imageGrayscale.deinit();

    const height = 453;
    const width = 680;
    const num_channels = 3;

    try testing.expectEqual(imageRgb.shape.len, 3);
    try testing.expectEqual(imageRgb.shape[0], num_channels);
    try testing.expectEqual(imageRgb.shape[1], height);
    try testing.expectEqual(imageRgb.shape[2], width);

    try testing.expectEqual(imageYCbCr.shape.len, 3);
    try testing.expectEqual(imageYCbCr.shape[0], num_channels);
    try testing.expectEqual(imageYCbCr.shape[1], height);
    try testing.expectEqual(imageYCbCr.shape[2], width);

    try testing.expectEqual(imageGrayscale.shape.len, 3);
    try testing.expectEqual(imageGrayscale.shape[0], 1);
    try testing.expectEqual(imageGrayscale.shape[1], height);
    try testing.expectEqual(imageGrayscale.shape[2], width);
}
