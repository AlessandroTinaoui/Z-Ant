# Guide for contributors to the Z-Ant rendering and lowering pipeline.

## First of all: What's Z-Ant, and what does the Z-Ant lowering pipeline aim to do?

Z-Ant is a library written in Zig to aide developers and machine learning engineers in deploying ONNX models to embedded devices and any other sort of microcontroller, if LLVM can compile machine code to it, Z-Ant aims to support it.

What does Z-Ant do exactly to achieve its goals? As of now Z-Ant parses an onnx file, and manually generates the source files to build a custom-tailored source code just to compute that exact onnx model, as of now, we use raw, hardcoded functions, such as MatMuls, Convolutions, Gather, MaxPool etc... But as with all system, we've encountered a limit, and that was the static memory management concering tensors allocated at compile time inside the binary we were producing.

So we've decided to change plans:

## The new Z-Ant code generation pipeline:

Our new pipeline aims to achieve both a reduction in the amount of tensors allocated to serve as buffers between each operation, and to in the future allow for state of the art kernel fusion algorhithms to minimize the amount of computing power needed to evaluate models deployed using it.

The new pipeline is based around three fundamental ideas:

### IDs.

Each piece of memory, variable, array, buffer, allocator inside the codegeneration pipeline is denominated by a single, unique, `usize` number that defines it, any part of the generated code can access it using its id.

### Uops.

What is an UOp? an UOp is the smallest building block in our code generation pipeline, all higher order functions are solely built using them; a UOp is a struct composed of: 

- A tag, indicating what kind of UOp it is. 

- A data type enum, indicating the data type it will be acting on.

- A sources array, containing the id's of the memory it acts on.

- An `Any` field, `Any` is a special tagged union expandable by anyone implementing an uop, it allows for a special, one of a kind payload dedicated for any need an UOp might need in its implementation. Check `UOps.zig` for more details!

UOps can range from arithmetic expressions to memory allocations and deallocations.

### Renderers.

What is a renderer? A renderer is a function dedicated to writing the code that the UOp will generate.

When implementing a new UOp you have complete libery on how it will handle its arguments, as long as the data type is respected, and you don't change any previous underlying data structures. Code along! Take a look at how Clip handles the Any or how Range handles it.

### Views.

We are building code while not directly linting it, handling a tensor struct across multiple compilation and generation layers would be hellish, so we handle memory directly as flat arrays. How do we mantain shape then? We don't handle it in the code we generate, we hardcode them during generation. A view is a stride array tied to the id of the memory it maps, a view is malleable, can change over the course of the code generation.

To go past the word salad you just had to go through, lets go through an example together:

Lets handle an array of 10 elements:

`const var = [10]usize{};`

And let's associate a view with these strides: [5,1].

So as we've declared it, now this flat array is now a matrix with two rows and five colums.

Z-Ant has a specific UOp used to access such tensors, called GEP, looking at the already lowered functions you will see various uses for GEP, flat indexes, and view tied indexes. If we were to access an element with GEP referencing the view we just declared, the strides would be applied implicitly by the renderer.

Now, just out of spite, lets traspose the array, so now we want a matrix with five rows and two colums, all we need to do is change the view! now the view is: [1, 5], and all code rendered after the view change will see the tensor the array represents as the transposed one.

The rendered code will just see them as hardcoded parameters, this makes code faster on execution, while mantaining the same freedom you'd have in handwritten code!

## Lowering a function.

Lowering a function is probably the most complicated part of the lowering process, as you don't have a linter to guide you through the function the renderer will produce!

Use any UOp you deem fit and any argument to the lowered function you see fit, but remember to test it! Each lowered function must implement a pipeline test, where you check manually if what you wrote is satisfactory, and a kernel test, where with some test values you check if the function really does calculate what you aim to do with it.

### "I feel like I don't know where to start, what do I do?"

Give an eye to functions that have already been lowered by the contributors of the project! Convolution and MatMul are some excellent starting points, and give an eye to some renderers to know how a UOp is handled under the hood.

### Happy hacking!!
\- The Z-Ant team.