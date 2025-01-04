const std = @import("std");
const tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const ActivationType = @import("activation_function").ActivationType;
const ActivLib = @import("activation_function");
const LayerError = @import("errorHandler").LayerError;

/// Represents an activation layer in a neural network, designed for use with tensors.
/// The `ActivationLayer` type is parameterized by the type `T`, which represents the data type of the tensor elements.
/// It includes the structure and utilities required to handle the layer's operations and its activation function.
///
/// @param T The data type of the tensor elements (e.g., `f32`, `f64`, etc.).
pub fn ActivationLayer(comptime T: type) type {
    return struct {

        // Layer shape --------------------
        n_inputs: usize,
        n_neurons: usize,
        input: tensor.Tensor(T), //is saved for semplicity, it can be sobstituted
        output: tensor.Tensor(T), // output = dot(input, weight.transposed) + bias

        /// Activation Function -----------------------
        /// The activation function applied to the layer's output.
        /// This is of type `ActivationType`, which determines the specific activation behavior (e.g., ReLU, Sigmoid, etc.).
        activationFunction: ActivationType,

        // Utils---------------------------
        allocator: *const std.mem.Allocator,

        const Self = @This();

        pub fn create(self: *Self) Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.ActivationLayer,
                .layer_ptr = self,
                .layer_impl = &.{
                    .init = init,
                    .deinit = deinit,
                    .forward = forward,
                    .backward = backward,
                    .printLayer = printLayer,
                    .get_n_inputs = get_n_inputs,
                    .get_n_neurons = get_n_neurons,
                    .get_input = get_input,
                    .get_output = get_output,
                },
            };
        }

        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const struct { n_inputs: usize, n_neurons: usize } = @ptrCast(@alignCast(args));
            const n_inputs = argsStruct.n_inputs;
            const n_neurons = argsStruct.n_neurons;
            std.debug.print("\nInit ActivationLayer: n_inputs = {}, n_neurons = {}, Type = {}", .{ n_inputs, n_neurons, @TypeOf(T) });

            //check on parameters
            if (n_inputs <= 0 or n_neurons <= 0) return LayerError.InvalidParameters;

            //initializing number of neurons and inputs----------------------------------
            self.n_inputs = n_inputs;
            self.n_neurons = n_neurons;
            self.allocator = alloc;
        }

        pub fn convInit(ctx: *anyopaque, input_channels: usize, output_channels: usize, kernel_size: [2]usize) !void {
            _ = ctx;
            _ = input_channels;
            _ = output_channels;
            _ = kernel_size;
            return LayerError.InvalidLayerType;
        }

        ///Deallocate the layer
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.output.data.len > 0) {
                self.output.deinit();
            }
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            std.debug.print("\nActivationLayer resources deallocated.", .{});
        }

        ///Forward pass of the layer if present it applies the activation function
        /// We can improve it removing as much as possibile all the copy operations
        pub fn forward(ctx: *anyopaque, input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // 1. copy the input both in self.input and self.output because activation.forward doesn't return a tensor
            self.input = try input.copy();
            self.output = try input.copy();

            // 2. Apply activation function
            // I was gettig crazy with this.activation initialization since ActivLib.ActivationFunction( something ) is
            //dynamic and we are trying to do everything at comptime, no excuses, you can do better than me !
            if (self.activationFunction == ActivationType.ReLU) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.ReLU);
                var activation = act_type{};
                try activation.forward(&self.output);
            } else if (self.activationFunction == ActivationType.Softmax) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Softmax);
                var activation = act_type{};
                try activation.forward(&self.output);
            } else if (self.activationFunction == ActivationType.Sigmoid) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Sigmoid);
                var activation = act_type{};
                try activation.forward(&self.output);
            }
            try self.output.isSafe();
            return self.output;
        }

        /// Backward pass of the layer It takes the dValues from the next layer and computes the gradients
        pub fn backward(ctx: *anyopaque, dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            //---- Key Steps: -----
            // 1. Apply the derivative of the activation function to dValues
            if (self.activationFunction == ActivationType.ReLU) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.ReLU);
                var activation = act_type{};
                try activation.derivate(dValues, &self.input);
            } else if (self.activationFunction == ActivationType.Softmax) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Softmax);
                var activation = act_type{};
                try activation.derivate(dValues, &self.output);
            } else if (self.activationFunction == ActivationType.Sigmoid) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Sigmoid);
                var activation = act_type{};
                try activation.derivate(dValues, &self.output);
            }

            return dValues.copy();
        }

        ///Print the layer used for debug purposes it has 2 different verbosity levels
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            std.debug.print("\n ************************Activation layer*********************", .{});
            //MENU choice:
            // 0 -> full details layer
            // 1 -> shape schema
            if (choice == 0) {
                std.debug.print("\n neurons:{}  inputs:{}", .{ self.n_neurons, self.n_inputs });
                std.debug.print("\n \n************input", .{});
                self.input.printMultidim();
                std.debug.print("\n \n************output", .{});
                self.output.printMultidim();
                std.debug.print("\n \n************activation function", .{});
                std.debug.print("\n  {any}", .{self.activationFunction});
            }
            if (choice == 1) {
                std.debug.print("\n   input         activation     output", .{});
                std.debug.print("\n [{} x {}]   ->  {any}     = [{} x {}] ", .{
                    self.input.shape[0],
                    self.input.shape[1],
                    self.activationFunction,
                    self.output.shape[0],
                    self.output.shape[1],
                });
                std.debug.print("\n ", .{});
            }
        }

        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return self.n_inputs;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return self.n_neurons;
        }

        pub fn get_input(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.output;
        }
    };
}
