---
name: 'Math Operations Template'
about: Related to TensorMath functions.
title: ''
labels: ''
assignees: ''

---

## Current Problem
Describe here your issue

## Rules
### ONNX standard
Every math method must respect the onnx representation.
You can find all the necessary details about onnx operators [here](https://onnx.ai/onnx/operators/index.html) !  

### Core Implementation (`TensorMath`)
In this first phase you have to create `src/core/Tensor/TensorMath/op_<operation_name>.zig`, a new file with all the logic behind the operations.
It must have 3 functions:

The "standard version" of a method is in the format:

```  
    pub fn opName (inputs...,args...) !Tensor(T) {
       ...
       outputShape = get_opName_output_shape(inputs...,args...)
       outputTensor = compute output tensor...
       ...
       opName_lean(input, args..., output)

       return outputTensor;
    }
```

The "lean version" of a method is in the format:

```  
    pub inline fn methodName_lean (args) void {
       ...
       ... actual computation of the output
    }
```

The "compute output shape" of a method is in the format:

```  
    pub inline fn get_opName_output_shape (inputs...,args...) void {
       ...
       ... computation of the output tensor's shape
    }
```
"lean" and "standard" are **mandatory**.
The lean version is called during the codegen and the standard version is used in testings.

### Testing 
To test your implementation of the new operation write a couple of tests in `tests/core/Tensor/TensorMath/` and then update `tests/CodeGen/Python-ONNX/onnx_gen.py` to generate the operation (that's for fuzzing)

### CodeGen Integration 
Create a dedicated file in `src/IR_graph/op_union/operators/`.

Thats how it must be structured: 
```  
    pub const opName = struct {
    input: *TensorZant,
    output: *TensorZant,
    attributes: ...,

    pub fn init(nodeProto: *NodeProto) !opName {
        ... struct initialization ...
    }

    pub fn get_output_shape(self: opName) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_output_tensor(self: opName) *TensorZant {
        return self.output_Y;
    }

    pub fn compute_output_shape(self: opName) []usize {
        ... computing output shape (using your get_opName_output_shape from TensorMath) ...
    }

    pub fn write_op(self: opName, writer: std.fs.File.Writer) !void {
        ... codegen logic ...
    }

    pub fn print(self: opName) void {
        std.debug.print("\n opName:\n {any}", .{self});
    }
};
```


> **Note**: During `write_op()`, make sure the generated code is minimal and efficient, calling only the lean function without any runtime checks.

See the [full guide][guide] for more details.

[guide]: ../../../Z-Ant/src/IR_graph/HOW_TO_ADD_MATHEMATICAL_OPERATIONS.md