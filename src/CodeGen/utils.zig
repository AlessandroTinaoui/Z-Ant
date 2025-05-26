const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const DataType = onnx.DataType;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const allocator = zant.utils.allocator.allocator;
const codegen = @import("codegen.zig");
const globals = codegen.globals;
const tests = codegen.tests;
const ReadyNode = globals.ReadyNode;
const ReadyTensor = globals.ReadyTensor;

// -------------------- GETTERS --------------------

//Given an element from DataType Enum in onnx.zig returns the equivalent zig type
pub inline fn getType(data_type: DataType) !type {
    switch (data_type) {
        .FLOAT => {
            return f32;
        },
        .UINT8 => {
            return u8;
        },
        .INT8 => {
            return i8;
        },
        .UINT16 => {
            return u16;
        },
        .INT16 => {
            return i16;
        },
        .INT32 => {
            return i32;
        },
        .INT64 => {
            return i64;
        },
        .FLOAT16 => {
            return f16;
        },
        .DOUBLE => {
            return f64;
        },
        .UNIT32 => {
            return u32;
        },
        .UINT64 => {
            return u64;
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Given an element from DataType Enum in onnx.zig returns the equivalent string of a zig type
pub inline fn getTypeString(data_type: DataType) ![]const u8 {
    switch (data_type) {
        .FLOAT => {
            return "f32";
        },
        .UINT8 => {
            return "u8";
        },
        .INT8 => {
            return "i8";
        },
        .UINT16 => {
            return "u16";
        },
        .INT16 => {
            return "i16";
        },
        .INT32 => {
            return "i32";
        },
        .INT64 => {
            return "i64";
        },
        .FLOAT16 => {
            return "f16";
        },
        .DOUBLE => {
            return "f64";
        },
        .UINT32 => {
            return "u32";
        },
        .UINT64 => {
            return "u64";
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Returns the sanitized tensor's name, removes all non alphanumeric chars
pub inline fn getSanitizedName(name: []const u8) ![]const u8 {
    var sanitized = try allocator.alloc(u8, name.len);

    for (name, 0..) |char, i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
            std.ascii.toLower(char)
        else
            '_';
    }

    //std.log.debug("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

/// Returns a List of Ready nodes
/// A node is considered "computable" if all the node's input Tensors are set as ready
pub inline fn getComputableNodes(readyGraph: *std.ArrayList(ReadyNode)) !std.ArrayList(*ReadyNode) {
    var set: std.ArrayList(*ReadyNode) = std.ArrayList(*ReadyNode).init(allocator);
    var ready_input_counter: i8 = 0;
    var null_input_counter: i8 = 0;

    for (readyGraph.items) |*node| {
        if (!node.ready) {
            for (node.inputs.items) |input| {
                if (input == null) null_input_counter += 1 else if (input.?.ready) ready_input_counter += 1;
            }
            for (node.outputs.items) |output| {
                if (output.ready) return error.OutputReadyTooEarly;
            }
            if (ready_input_counter + null_input_counter == node.inputs.items.len) {
                try set.append(node);
                //std.log.debug("\n    --- {s} is computable", .{node.nodeProto.name.?});
            }
            ready_input_counter = 0;
            null_input_counter = 0;
        }
    }

    return set;
}

pub inline fn getConstantTensorDims(nodeProto: *NodeProto) ![]const i64 {
    //check the node is a Constant
    if (std.mem.indexOf(u8, try getSanitizedName(nodeProto.op_type), "constant")) |_| {} else return error.NodeNotConstant;

    return if (nodeProto.attribute[0].t) |tensorProto| tensorProto.dims else error.ConstantTensorAttributeNotAvailable;
}

/// This method search for the existance of a Tensor named "tensorName" inside the onnx model.graph.value_info array.
/// If founded return its shape, else returns null.
pub fn getTensorShape(tensorName: []const u8) ?[]i64 {
    for (globals.onnxModel.graph.?.value_info) |vi| {
        if (std.mem.eql(u8, vi.name.?, tensorName)) {
            return vi.type.?.tensor_type.?.shape.?.shape;
        }
    }

    return null;
}
// -------------------- SETTERS --------------------

// Marks output tensors as ready for computation in all the graph
pub fn setOutputsReady(completedNode: *ReadyNode, tensorHashMap: *std.StringHashMap(ReadyTensor)) !void {
    std.log.info("\n -----> set {s} outputs to ready", .{completedNode.nodeProto.name orelse "(unnamed)"});
    completedNode.ready = true;
    for (completedNode.outputs.items) |ready_output_tensor| { //for each output tensor of the completed node
        var mutablePtr: *ReadyTensor = if (tensorHashMap.getPtr(ready_output_tensor.name)) |V_ptr| V_ptr else return error.keyNotAvailable;
        mutablePtr.ready = true;
        std.log.info("\n    {s} --> ready", .{mutablePtr.name});
    }
}

// -------------------- BOOLEANS --------------------

// returns true if all the inputs are ready
pub inline fn areAllInputsReady(node: *ReadyNode) bool {
    for (node.inputs.items) |input| {
        if (!input.ready) return false;
    }
    return true;
}

//returns true if all the inputs and all the outputs of a node are set as ready
pub inline fn isComputed(readyNode: *ReadyNode) !bool {
    for (readyNode.inputs.items) |input| {
        if (!input.ready) return false;
    }
    for (readyNode.outputs.items) |output| {
        if (!output.ready) return false;
    }
    return true;
}

//return true if the first parameter is an initializer
pub fn isInitializer(name: []const u8, initializers: []*TensorProto) bool {
    for (initializers) |init| {
        if (std.mem.eql(u8, init.name.?, name)) return true;
    }
    return false;
}

//return true if the name is an input of the nn
pub fn isInput(name: []const u8) bool {
    for (globals.onnxModel.graph.?.inputs) |input| {
        if (std.mem.eql(u8, input.name.?, name)) return true;
    }
    return false;
}

//return true if the name is an output of the nn
pub fn isOutput(name: []const u8) bool {
    for (globals.onnxModel.graph.?.outputs) |output| {
        if (std.mem.eql(u8, output.name.?, name)) return true;
    }
    return false;
}
// -------------------- PRINTERS --------------------

// Prints the list of nodes in the given computation graph.
// Outputs each node's name along with its input and output tensors and their readiness status.
pub fn printNodeList(graph: std.ArrayList(ReadyNode)) !void {
    std.debug.print("\n-------------------------------------------------------------", .{});
    std.debug.print("\n+                        READY GRAPH                        +", .{});
    std.debug.print("\n-------------------------------------------------------------\n", .{});
    for (graph.items) |node| {
        std.log.info("\n ----- node: {s}", .{node.nodeProto.name.?});

        std.log.info("\n          inputs: ", .{});
        // Write the inputs
        for (node.inputs.items) |input| {
            std.log.info("\n              ->{s} {s}", .{ input.name, if (input.ready) "--->ready" else "" });
        }

        std.log.info("\n          outputs:", .{});
        // Write the outputs
        for (node.outputs.items) |output| {
            std.log.info("\n              -> {s} {s}", .{ output.name, if (output.ready) "--->ready" else "" });
        }
    }
}

// Prints the list of nodes that are ready for computation.
// Outputs each node's name, operation type, inputs, and outputs along with their readiness status.
pub fn printComputableNodes(computableNodes: std.ArrayList(*ReadyNode), details: bool) !void {
    std.debug.print("\n------------------------------------------------------------", .{});
    std.debug.print("\n+                  COMPUTABLE NODES  n:{}                  +", .{computableNodes.items.len});
    std.debug.print("\n------------------------------------------------------------\n", .{});

    for (computableNodes.items) |node| {
        node.print(details);
    }
}

// Prints the list of unique ONNX operations present in the given graph.
// Outputs each operation type only once.
pub fn printOperations(graph: *GraphProto) !void {
    std.debug.print("\n", .{});
    std.debug.print("\n-------------------------------------------------", .{});
    std.debug.print("\n+                ONNX operations                +", .{});
    std.debug.print("\n-------------------------------------------------\n", .{});

    var op_set = std.StringHashMap(void).init(std.heap.page_allocator);
    defer op_set.deinit();

    for (graph.nodes) |node| {
        try op_set.put(node.op_type, {});
    }

    var it = op_set.iterator();
    while (it.next()) |entry| {
        std.log.debug("\n- {s}", .{entry.key_ptr.*});
    }

    std.debug.print("\n-------------------------------------------------\n", .{});
}

// Function to print all entries in the tensorHashMap
pub fn printTensorHashMap(map: std.StringHashMap(ReadyTensor)) void {
    std.debug.print("\n-------------------------------------------------------------", .{});
    std.debug.print("\n+                       READY HASHMAP                       +", .{});
    std.debug.print("\n-------------------------------------------------------------\n", .{});

    var it = map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const tensor = entry.value_ptr.*;
        std.log.info("\nTensor Name: {s}", .{key});
        std.log.info("\n     Ready: {}", .{tensor.ready});
        std.log.info("\n     Shape: [{any}]", .{tensor.shape});
    }
}

// ----------------- DATA TYPE management -------------

pub inline fn i64SliceToUsizeSlice(input: []const i64) ![]usize {
    var output = try allocator.alloc(usize, input.len);

    const maxUsize = std.math.maxInt(usize);

    for (input, 0..) |value, index| {
        if (value < 0) {
            return error.NegativeValue;
        }
        if (value > maxUsize) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

pub fn usizeSliceToI64Slice(input: []usize) ![]const i64 {
    var output = try allocator.alloc(i64, input.len);

    for (input, 0..) |value, index| {
        if (value > std.math.maxInt(i64)) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

/// Converts any integer value to usize with proper bounds checking
/// Returns error.NegativeValue if the input is negative (for signed types)
/// Returns error.ValueTooLarge if the input exceeds the maximum usize value
pub inline fn toUsize(comptime T: type, value: T) !usize {
    // Ensure T is an integer type
    comptime {
        if (@typeInfo(T) != .Int) {
            @compileError("toUsize only supports integer types");
        }
    }

    // Check for negative values if T is signed
    if (@typeInfo(T).Int.signedness == .signed and value < 0) {
        return error.NegativeValue;
    }

    // Check if value exceeds maximum usize
    const maxUsize = std.math.maxInt(usize);
    if (@as(u128, @intCast(if (@typeInfo(T).Int.signedness == .signed) @as(u128, @intCast(@max(0, value))) else @as(u128, @intCast(value)))) > maxUsize) {
        return error.ValueTooLarge;
    }

    return @intCast(value);
}

pub inline fn sliceToUsizeSlice(slice: anytype) []usize {
    const T = @TypeOf(slice);
    const info = @typeInfo(T);

    switch (info) {
        .pointer => {
            const child = info.pointer.child;
            const child_info = @typeInfo(child);

            var output = allocator.alloc(usize, slice.len) catch @panic("Out of memory in sliceToUsizeSlice");
            const maxUsize = std.math.maxInt(usize);

            for (slice, 0..) |value, index| {
                if (child_info == .int) {
                    // Handle integer types
                    if (value < 0) {
                        if (value == -1) {
                            output[index] = std.math.maxInt(usize);
                        } else {
                            @panic("Invalid negative value in sliceToUsizeSlice (only -1 is allowed)");
                        }
                    } else {
                        if (@as(u128, @intCast(value)) > maxUsize) {
                            @panic("Value too large in sliceToUsizeSlice");
                        }
                        output[index] = @intCast(value);
                    }
                } else if (child_info == .float) {
                    // Handle float types
                    if (value < 0) {
                        if (value == -1.0) {
                            output[index] = std.math.maxInt(usize);
                        } else {
                            @panic("Invalid negative value in sliceToUsizeSlice (only -1 is allowed)");
                        }
                    } else {
                        if (value > @as(f64, @floatFromInt(maxUsize))) {
                            @panic("Value too large in sliceToUsizeSlice");
                        }
                        output[index] = @intFromFloat(value);
                    }
                } else {
                    @compileError("Unsupported element type for sliceToUsizeSlice: " ++ @typeName(child));
                }
            }

            return output;
        },
        else => {
            @compileError("Unsupported type for sliceToUsizeSlice: " ++ @typeName(T));
        },
    }
}

// Modify signature to accept allocator
pub inline fn sliceToIsizeSlice(alloc: std.mem.Allocator, slice: anytype) []isize {
    const T = @TypeOf(slice);
    const info = @typeInfo(T);

    switch (info) {
        .pointer => {
            const child = info.pointer.child;
            const child_info = @typeInfo(child);

            // Use the passed allocator
            var output = alloc.alloc(isize, slice.len) catch @panic("Out of memory in sliceToIsizeSlice");
            const maxIsize = std.math.maxInt(isize);
            const minIsize = std.math.minInt(isize);

            for (slice, 0..) |value, index| {
                if (child_info == .int) {
                    // Handle integer types
                    if (value < minIsize or value > maxIsize) {
                        @panic("Value out of isize range in sliceToIsizeSlice");
                    }
                    output[index] = @intCast(value);
                } else if (child_info == .float) {
                    // Handle float types
                    if (value < @as(f64, @floatFromInt(minIsize)) or value > @as(f64, @floatFromInt(maxIsize))) {
                        @panic("Value out of isize range in sliceToIsizeSlice");
                    }
                    output[index] = @intFromFloat(value);
                } else {
                    @compileError("Unsupported element type for sliceToIsizeSlice: " ++ @typeName(child));
                }
            }

            return output;
        },
        else => {
            @compileError("Unsupported type for sliceToIsizeSlice: " ++ @typeName(T));
        },
    }
}

pub fn i64ToI64ArrayString(values: []const i64) ![]const u8 {
    var buffer: [20]u8 = undefined;
    var res_string = try std.mem.concat(allocator, u8, &[_][]const u8{"&[_]i64{"});
    for (values, 0..) |val, i| {
        if (i > 0) res_string = try std.mem.concat(allocator, u8, &[_][]const u8{ res_string, "," });
        const val_string = std.fmt.bufPrint(&buffer, "{}", .{val}) catch unreachable;
        res_string = try std.mem.concat(allocator, u8, &[_][]const u8{ res_string, val_string });
    }
    res_string = try std.mem.concat(allocator, u8, &[_][]const u8{ res_string, "}" });

    return res_string;
}

pub fn u32ToUsize(alloc: std.mem.Allocator, input: [*]u32, size: u32) ![]usize {
    var output = try alloc.alloc(usize, size);

    const maxUsize = std.math.maxInt(usize);

    for (0..size) |i| {
        if (input[i] < 0) {
            return error.NegativeValue;
        }
        if (input[i] > maxUsize) {
            return error.ValueTooLarge;
        }
        output[i] = @intCast(input[i]);
    }

    return output;
}

pub fn parseNumbers(input: []const u8) ![]i64 {
    var list = std.ArrayList(i64).init(allocator);
    errdefer list.deinit();

    if (input.len == 0) return list.toOwnedSlice();

    var it = std.mem.splitScalar(u8, input, ',');
    while (it.next()) |num_str| {
        const num = try std.fmt.parseInt(i64, num_str, 10);
        try list.append(num);
    }

    return list.toOwnedSlice();
}

pub fn i64SliceToUsizeArrayString(values: []const i64) ![]const u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit(); // Frees all memory

    try list.appendSlice("&[_]usize{");
    for (values, 0..) |val, i| {
        if (i > 0) try list.append(',');
        try list.writer().print("{}", .{val});
    }
    try list.append('}');

    return try list.toOwnedSlice(); // Caller must free this!
}

// ----------------- FILE MANAGEMENT -----------------
// Copy file from src to dst
pub fn copyFile(src_path: []const u8, dst_path: []const u8) !void {
    var src_file = try std.fs.cwd().openFile(src_path, .{});
    defer src_file.close();

    var dst_file = try std.fs.cwd().createFile(dst_path, .{});
    defer dst_file.close();

    // Use a buffer to copy in chunks
    var buf: [4096]u8 = undefined;
    while (true) {
        const bytes_read = try src_file.read(&buf);
        if (bytes_read == 0) break;
        _ = try dst_file.write(buf[0..bytes_read]);
    }
}

// Read the user_tests json file and return a list of test cases
pub fn loadUserTests(comptime T: type, user_tests_path: []const u8) !std.json.Parsed([]tests.UserTest(T)) {
    const user_tests_file = try std.fs.cwd().openFile(user_tests_path, .{});
    defer user_tests_file.close();

    const user_tests_content: []const u8 = try user_tests_file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(user_tests_content);

    const parsed_user_tests = try std.json.parseFromSlice([]tests.UserTest(T), allocator, user_tests_content, .{});

    return parsed_user_tests;
}

/// Parses a raw byte slice (expected to be little-endian) into an allocated slice of i64.
pub fn parseI64RawData(raw_data: []const u8) ![]i64 {
    const element_size = @sizeOf(i64);
    if (raw_data.len % element_size != 0) {
        std.log.warn("ERROR: Raw data length ({}) is not a multiple of i64 size ({})\n", .{ raw_data.len, element_size });
        return error.InvalidRawDataSize;
    }

    const num_elements = raw_data.len / element_size;
    if (num_elements == 0) {
        // Return an empty slice if raw_data is empty (and length is valid multiple of 0)
        return try allocator.alloc(i64, 0);
    }

    // Allocate the result slice.
    const result = try allocator.alloc(i64, num_elements);
    errdefer allocator.free(result);

    // Fallback: Use pointer casting to interpret raw bytes as i64 (assumes alignment and little-endian)
    // Ensure alignment (optional, might panic on some archs if unaligned)
    // if (@alignOf(i64) > @alignOf(u8) and @ptrToInt(raw_data.ptr) % @alignOf(i64) != 0) {
    //     std.log.warn("ERROR: Raw data pointer is not aligned for i64 read.\n", .{});
    //     return error.UnalignedRawData;
    // }

    // Cast the byte slice pointer to an i64 slice pointer
    const i64_ptr: [*]const i64 = @ptrCast(@alignCast(raw_data.ptr));

    // Copy the data from the cast pointer into the result slice
    @memcpy(result, i64_ptr[0..num_elements]);

    return result;
}
