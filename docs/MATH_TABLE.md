✅ : complete  
🔶 : WIP  
🔴 : missing  
  
onnx reference : \[onnx_name\]\(URL to [ONNX docs](https://onnx.ai/onnx/operators/index.html)\)  
tensor math : \[ fileName.zig \]\( path to the method \)  
tensor math tests: ✅, 🔶, 🔴  
codegen : ✅, 🔶, 🔴. You can find all the `write_op()` [here](../src/CodeGen/math_handler.zig)  
oneOp model generator: ✅ if the oneOpModel is created, remember to add the onnx name inside [available_op](all_link_here)



| math op name | onnx reference | IR_graph | tensor math | tensor math tests | codegen | oneOp model generator (.py) | notes |
| :------------: | :------------: | :---------: | :---------: | :-----------: | :-------: | :--------: | :------------- |
|Add| [Add](https://onnx.ai/onnx/operators/onnx__Add.html) | ✅ | [op_add](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_addition.zig) | ✅ | ✅ | ✅ |
|AveragePool| [AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html) | ✅ | [op_pooling](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_pooling.zig) | ✅ | ✅ | ✅ |
|BatchNormalization| [BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | ✅ | [op_add](../src/Core/Tensor/TensorMath/op_batchNormalization.zig) | 🔶 | ✅ | ✅ |
| convolution | [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html) | ✅ | [op_convolution.zig](../src/Core/Tensor/TensorMath/op_convolution.zig) | ✅ | ✅ | ✅ |
|Elu| [Elu](https://onnx.ai/onnx/operators/onnx__Elu.html) | ✅ | [op_elu](../src/Core/Tensor/TensorMath/op_elu.zig) | ✅ | ✅ | ✅ |
|Flatten| [Flatten](https://onnx.ai/onnx/operators/onnx__Flatten.html) | ✅ | [op_flatten](../src/Core/Tensor/TensorMath/lib_shape_math/op_flatten.zig) | ✅ | ✅ | ✅ | missing axis attribute
| gemm | [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html) | ✅ | [op_gemm](../src/Core/Tensor/TensorMath/op_gemm.zig) | ✅ | ✅ | ✅ |
|Div| [Div](https://onnx.ai/onnx/operators/onnx__Div.html) | ✅ | [op_div](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_division.zig) | ✅ | ✅ | ✅ |
|Concat| [Concat](https://onnx.ai/onnx/operators/onnx__Concat.html)| ✅ | [op_concat](../src/Core/Tensor/TensorMath/lib_shape_math/op_concatenate.zig) | ✅ | ✅ | ✅ |
|ReduceMean| [ReduceMen](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) | ✅ | [op_ReduceMean](../src/Core/Tensor/TensorMath/lib_reduction_math.zig) | ✅ | ✅ | ✅ |
|Ceil| [Ceil](https://onnx.ai/onnx/operators/onnx__Ceil.html) | ✅ | [op_Ceil](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_ceil.zig) | ✅ | ✅ | ✅ |
|Conv| [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html) | ✅ | [op_conv](../src/Core/Tensor/TensorMath/op_convolution.zig) | ✅ | ✅ | ✅ |
|Floor| [Floor](https://onnx.ai/onnx/operators/onnx__Floor.html) | 🔴 | [op_floor](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_floor.zig) | ✅ | ✅ | ✅ |
|Gelu| [Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html) | 🔴 | [op_gelu](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_gelu.zig) | ✅ | ✅ | ✅ |
|MaxPool| [MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html)   | ✅ | [op_maxPool](../src/Core/Tensor/TensorMath/op_pooling.zig) | ✅ | ✅ | ✅ | Missing ceil param|
| Gather| [Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)  | ✅ | [op_gather](../src/Core/Tensor/TensorMath/lib_shape_math/op_gather.zig) | ✅ | ✅ | ✅ |
| Identity | [Identity](https://onnx.ai/onnx/operators/onnx__Identity.html) | ✅ | [op_Identity](../src/Core/Tensor/TensorMath/lib_shape_math/op_identity.zig) | ✅ | ✅ | ✅ |
| LeakyRelu | [LeakyRelu](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html) | ✅ | [op_Leaky](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_leaky_reLU.zig) | ✅ | ✅ | ✅ |
| MatMul | [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)  | ✅ | [op_matMul](../src/Core/Tensor/TensorMath/op_mat_mul.zig) | ✅ | ✅ | ✅ |
| Mul| [Mul](https://onnx.ai/onnx/operators/onnx__Mul.html) | ✅ | [op_mul](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_multiplication.zig) | ✅ | ✅ | ✅ |
| Neg| [Neg](https://onnx.ai/onnx/operators/onnx__Neg.html) | ✅ | [op_neg](../src/Core/Tensor/TensorMath/lib_logical_math.zig) | ✅ | ✅ | ✅ |
|OneHot| [OneHot](https://onnx.ai/onnx/operators/onnx__OneHot.html) | 🔴 | [op_oneHot](../src/Core/Tensor/TensorMath/op_oneHot.zig) | ✅ | ✅ | 🔴 | One Hot is not included in available_operations.txt due to a bug, to solve, not urgent |
| Relu| [Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) | ✅ | [op_relu](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_reLU.zig) | ✅ | ✅ | ✅ |
| Reshape| [Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | ✅ | [op_reshape](../src/Core/Tensor/TensorMath/lib_shape_math/op_reshape.zig)  | ✅ | ✅ | ✅ |
| Resize | [Resize](https://onnx.ai/onnx/operators/onnx__Resize.html) | ✅ | [op_resize](../src/Core/Tensor/TensorMath/lib_shape_math/op_resize.zig) | ✅ | ✅ | ✅| |
| Shape| [Shape](https://onnx.ai/onnx/operators/onnx__Shape.html) | ✅ | [op_shape](../src/Core/Tensor/TensorMath/lib_shape_math/op_shape.zig)| ✅ | ✅ | ✅ | Tested but not supported by Onnx Python Generator|
| Sigmoid| [Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html) | ✅ | [op_Sigmoid](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_sigmoid.zig)  | ✅ | ✅ | ✅ |
| Slice| [Slice](https://onnx.ai/onnx/operators/onnx__Slice.html) | ✅ | [op_slice](../src/Core/Tensor/TensorMath/lib_shape_math/op_slice.zig) | ✅ | ✅ | ✅ |
| Softmax| [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html) | ✅ | [op_Softmax](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_softmax.zig) | ✅ | ✅ | ✅ |
| Split  | [Split](https://onnx.ai/onnx/operators/onnx__Split.html) | ✅ | [op_split](../src/Core/Tensor/TensorMath/lib_shape_math/op_split.zig)| ✅ | ✅ | ✅ | 
|Sqrt| [Sqrt](https://onnx.ai/onnx/operators/onnx__Sqrt.html) | 🔴 | [op_sqrt](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_sqrt.zig) | ✅ | ✅ | ✅ || Sub| [Sub](https://onnx.ai/onnx/operators/onnx__Sub.html) | ✅ | [op_Sub](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_subtraction.zig) | ✅ | ✅| ✅|
| Tanh | [Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html) | ✅ | [op_tanh](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_tanh.zig) | ✅ | ✅ | ✅ |
| Transpose| [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html)| ✅ | [op_Transose](../src/Core/Tensor/TensorMath/lib_shape_math/op_transpose.zig) | ✅ | ✅ | ✅ |
| Unsqueeze| [Unsqueeze](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) | ✅ | [op_unsqueeze](../src/Core/Tensor/TensorMath/lib_shape_math/op_unsqueeze.zig)| ✅ | ✅ | ✅| |
