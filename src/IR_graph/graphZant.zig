const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;
const NodeZant = @import("nodeZant.zig").NodeZant;

//--- proto structure
const GraphProto = zant.onnx.GraphProto;

pub const GraphZant = struct {
    name: ?[]const u8,
    nodes: std.ArrayList(*NodeZant),
    graphProto: *GraphProto,

    /// Initializes a GraphZant instance.
    pub fn init(graphProto: *GraphProto) !GraphZant {
        return GraphZant{
            .name = graphProto.name.?,
            .nodes = std.ArrayList(*NodeZant).init(allocator),
            .graphProto = graphProto,
        };
    }

    /// Deinitializes a GraphZant instance, freeing allocated resources.
    pub fn deinit(self: *GraphZant) void {
        // Deinitialize each NodeZant in the nodes list
        for (self.nodes.items) |node| {
            node.deinit();
            allocator.destroy(node); // Free the node
        }
        self.nodes.deinit();
    }

    // Adds a new node to the graph.

    pub fn build_graph(self: *GraphZant) !void {

        // create all the nodes
        for (self.graphProto.nodes) |nodeProto| {
            // allocate memory for the node
            const node = try allocator.create(NodeZant);
            node.* = try NodeZant.init(nodeProto);
            try self.nodes.append(node);
        }

        //hashmap for the outputs for the producers
        var output_map = std.StringHashMap(*NodeZant).init(allocator);
        defer output_map.deinit();

        //populate the output map with the nodes
        for (self.nodes.items) |node| {
            for (node.nodeProto.output) |output| {
                try output_map.put(output, node);
            }
        }

        // use the hashmap to find the producers of the inputs
        for (self.nodes.items) |customer| {
            for (customer.nodeProto.input) |input| {
                const producer = output_map.get(input);
                if (producer) |p| {
                    try p.add_next(customer);
                }
            }
        }
    }

    // linearize the graph
    pub fn linearize(self: *GraphZant, alloc: std.mem.Allocator) !std.ArrayList(*NodeZant) {
        var visited = std.AutoHashMap(*NodeZant, bool).init(alloc);
        var result = std.ArrayList(*NodeZant).init(alloc);
        defer visited.deinit();

        for (self.nodes.items) |node| {
            try dfs(node, &visited, &result);
        }

        GraphZant.reverseArrayList(*NodeZant, &result);

        return result;
    }

    pub fn dfs(
        node: *NodeZant,
        visited: *std.AutoHashMap(*NodeZant, bool),
        result: *std.ArrayList(*NodeZant),
    ) !void {
        if (visited.get(node)) |_| return;

        try visited.put(node, true);

        for (node.next.items) |child| {
            try dfs(child, visited, result);
        }

        try result.append(node);
    }

    pub fn reverseArrayList(comptime T: type, list: *std.ArrayList(T)) void {
        var i: usize = 0;
        var j: usize = list.items.len - 1;
        while (i < j) {
            const temp = list.items[i];
            list.items[i] = list.items[j];
            list.items[j] = temp;

            i += 1;
            j -= 1;
        }
    }

    pub fn print_beafore_linearizzation(self: *GraphZant) void {
        std.debug.print("\n\nGraphZant: {s}\n", .{self.name orelse "<unnamed>"});
        for (self.nodes.items) |node| {
            node.nodeProto.print("   ");
        }
    }
};
