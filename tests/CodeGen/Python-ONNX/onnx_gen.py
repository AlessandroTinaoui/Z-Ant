import onnx
from onnx import helper, TensorProto
import random
import numpy as np
import argparse
import datetime
from onnx import StringStringEntryProto
import onnxruntime as ort  
import json  
import os  


def random_shape(rank, min_dim=1, max_dim=10):
    """Generates a random shape of length 'rank'."""
    return [random.randint(min_dim, max_dim) for _ in range(rank)]

def generate_fuzz_model(op_name):
    """
    Crea gli initializer, i nodi e gli output con parametri casuali per l'operatore op_name.
    In questo caso tutti gli input vengono inseriti come initializer, per cui la lista degli input
    del grafo viene restituita vuota.
    """
    initializers = []
    
    # Pre-generazione di nomi per input e output
    input_names = [f"{op_name}_param_in_{i}" for i in range(5)]
    output_names = [f"{op_name}_param_out_{i}" for i in range(5)]
    metadata = {}

    if op_name in ["Relu", "Sigmoid", "Ceil", "Tanh", "Identity", "Neg", "Shape", "Floor", "Sqrt"]:
        # Operatori a singolo input con forma casuale (rank=4)
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        # Crea dati casuali e li inserisce come initializer
        if op_name == "Sqrt":
            data = np.abs(np.random.randn(*shape)).astype(np.float32)
        else:
            data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Gelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        approximate = random.choice(["none", "tanh"])

        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(
            op_name, 
            inputs=[input_names[0]], 
            outputs=[output_names[0]], 
            approximate=approximate, 
            name=f"{op_name}node_approx{approximate}",
            )

        metadata = {"input_shapes": [shape], "output_shapes": [shape], "approximate": approximate}
        return [input_info], output_info, [node], initializers, metadata


    elif op_name == "LeakyRelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        alpha = round(random.uniform(0.001, 0.2), 3)
        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                alpha=alpha, name=f"{op_name}node_alpha{alpha}")
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "alpha": alpha}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Softmax":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(-rank, rank-1)
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                axis=axis, name=f"{op_name}node_axis{axis}")
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "axis": axis}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name in ["Add", "Sub", "Div", "Mul"]:
        # Operatori binari: due input della stessa forma
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data0 = np.random.randn(*shape).astype(np.float32)
        data1 = np.random.randn(*shape).astype(np.float32)

        init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
        init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape, data1.flatten().tolist())
        initializers.extend([init_tensor0, init_tensor1])

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, 
                                inputs=[input_names[0], 
                                input_names[1]], 
                                outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "ReduceMean":
        # Generate input with predictable shape and values
        shape = [2, 3, 4, 5]  # Fixed shape for better debugging
        
        # Use a deterministic seed for this operation to ensure reproducibility
        local_rng = np.random.RandomState(42)  
        data = local_rng.randn(*shape).astype(np.float32)
        
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Use fixed axis instead of random
        axis = 1  # Reduce along the second dimension (channel dimension in NCHW)
        
        # Create output shape based on reduction
        out_shape = shape.copy()
        out_shape[axis] = 1  # Reduced dimension becomes 1
        
        keepdims = 1
        
        # Calculate the expected output manually to verify
        expected_output = np.mean(data, axis=axis, keepdims=True)
        
        # Debug info
        print(f"ReduceMean Test Case:")
        print(f"Input shape: {shape}")
        print(f"Output shape: {out_shape}")
        print(f"Reduction axis: {axis}")
        print(f"First few input values: {data.flatten()[:5]}")
        print(f"First few expected output values: {expected_output.flatten()[:5]}")
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                               axes=[axis], keepdims=keepdims,
                               name=f"{op_name}node_axis{axis}")
        
        # Add test metadata
        metadata = {
            "input_shapes": [shape], 
            "output_shapes": [out_shape], 
            "axes": [axis], 
            "keepdims": keepdims,
            "expected_output_first_values": expected_output.flatten()[:5].tolist()
        }
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Concat":
        # Due input con forma identica eccetto per la dimensione lungo l'asse di concatenazione
        shape = [1, random.randint(2,5), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        shape2 = shape.copy()
        shape2[axis] = shape[axis] + random.randint(1,3)
        data0 = np.random.randn(*shape).astype(np.float32)
        data1 = np.random.randn(*shape2).astype(np.float32)
        init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
        init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape2, data1.flatten().tolist())
        initializers.extend([init_tensor0, init_tensor1])
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] + shape2[axis]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                axis=axis, name=f"{op_name}node_axis{axis}")
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape, shape2], "output_shapes": [out_shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "OneHot":
        # Operatore OneHot: genera indices, depth e values come initializer
        # Genera un tensore indices con rango casuale (1 o 2)
        rank = random.randint(1, 2)
        indices_shape = random_shape(rank, min_dim=2, max_dim=3)
        max_index = 3  # Limite massimo per gli indici
        indices_data = np.random.randint(0, max_index, size=indices_shape).astype(np.int64)
        indices_tensor = helper.make_tensor(input_names[0], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(indices_tensor)

        # Genera depth (scalare)
        depth_value = random.randint(3, max_index)  # Valore positivo per depth
        depth_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [], [depth_value])
        initializers.append(depth_tensor)

        # Genera values (tensore 1D di lunghezza 2, tipo float32)
        values_data = np.array([0.0, 1.0], dtype=np.float32)  # [off_value, on_value]
        values_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [2], values_data.tolist())
        initializers.append(values_tensor)

        # Scegli un axis valido
        output_rank = rank + 1
        # axis = random.randint(-output_rank, output_rank - 1)
        axis = random.randint(max(-output_rank, -1000), min(output_rank - 1, 1000))

        # Calcola la forma dell'output
        out_shape = indices_shape.copy()
        normalized_axis = axis if axis >= 0 else axis + output_rank
        out_shape.insert(normalized_axis, depth_value)

        # Crea il nodo OneHot
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(
            op_name,
            inputs=[input_names[0], input_names[1], input_names[2]],
            outputs=[output_names[0]],
            axis=axis,
            name=f"{op_name}_node_axis{axis}"
        )

        # Input info fittizio
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.INT64, indices_shape)

        # Metadati
        metadata = {
            "input_shapes": [indices_shape, [], [2]],  # Forme di indices, depth, values
            "output_shapes": [out_shape],
            "axis": axis,
            "depth": depth_value,
            "indices": indices_data.flatten().tolist()[:5],  # Solo i primi 5 per debug
            "values": values_data.tolist()
        }

        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Gather":
        # First input: data; second input: indices
        shape = [5, random.randint(5,10)]
        data = np.random.randn(*shape).astype(np.float32)
        init_data = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_data)
        
        # Pick a random axis
        axis = random.randint(0, len(shape)-1)
        
        # Ensure indices are within bounds of the chosen axis
        max_index = shape[axis] - 1
        indices_shape = [random.randint(1,3)]
        indices_data = np.random.randint(0, max_index + 1, size=indices_shape).astype(np.int64)
        init_indices = helper.make_tensor(input_names[1], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(init_indices)
        
        # Calculate output shape
        out_shape = list(shape)
        out_shape[axis] = indices_shape[0]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                              axis=axis, name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {
            "input_shapes": [shape, indices_shape], 
            "output_shapes": [out_shape],
            "axis": axis,
            "indices": indices_data.tolist()
        }
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Elu":
        # generate 1D tensor 
        shape = [random.randint(1, 10)]  
        alpha = round(random.uniform(0.5, 2.0), 3)  
        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name,
                                inputs=[input_names[0]],
                                outputs=[output_names[0]], 
                                alpha=alpha,
                                name=f"{op_name}node_alpha{alpha}"
                                )
        metadata = {"input_shapes": [shape],
                    "output_shapes": [shape],
                    "alpha": alpha
                    }
        
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Flatten":
        # Generate casual rank and shape
        rank = random.randint(0, 4)
        if rank == 0:
            shape = []  
        else:
            shape = random_shape(rank, min_dim=1, max_dim=10)

        # Generate casual data
        total_size = 1 if rank == 0 else np.prod(shape)
        data = np.random.randn(total_size).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        # choosing right axis
        axis = random.randint(-rank, rank) if rank > 0 else 0  # for scalar, axis is 0

        # Calculate output shape
        if rank == 0:
            out_shape = [1, 1]  # Scalare -> [1, 1]
        else:
            outer_dim = 1
            normalized_axis = axis if axis >= 0 else axis + rank
            for i in range(normalized_axis):
                outer_dim *= shape[i]
            inner_dim = 1
            for i in range(normalized_axis, rank):
                inner_dim *= shape[i]
            out_shape = [outer_dim, inner_dim]

        # create input and output info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

        # Create Flatten node
        node = helper.make_node(
            op_name,
            inputs=[input_names[0]],
            outputs=[output_names[0]],
            axis=axis,
            name=f"{op_name}node_axis{axis}"
        )

        # Metadati
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [out_shape],
            "axis": axis
        }

        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Squeeze":
        # Generate input shape with at least one dimension of size 1
        shape = [1, random.randint(1, 3), 1, random.randint(5, 10)]
        data = np.random.randn(*shape).astype(np.float32)

        # Create initializer for the input tensor
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        # Randomly decide whether to include axes
        use_axes = random.choice([True, False])

        if use_axes:
            # Find which axes in the shape are 1
            squeezable_axes = [i for i, dim in enumerate(shape) if dim == 1]
            if not squeezable_axes:
                # No valid axes to squeeze; force at least one
                squeezable_axes = [0]
            # Pick a random subset of the squeezable axes
            num_axes = random.randint(1, len(squeezable_axes))
            selected_axes = random.sample(squeezable_axes, num_axes)

            # Create an INT64 tensor with selected_axes
            axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(selected_axes)], selected_axes)
            initializers.append(axes_tensor)

            # Node takes 2 inputs: data and axes
            node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]],
                                    outputs=[output_names[0]], name=f"{op_name}_node")

            # Compute output shape manually
            out_shape = [dim for i, dim in enumerate(shape) if i not in selected_axes]

        else:
            # No axes specified: remove all dims of size 1
            node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                    name=f"{op_name}_node")
            out_shape = [dim for dim in shape if dim != 1]

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

        metadata = {
            "input_shapes": [shape],
            "output_shapes": [out_shape],
            "axes": selected_axes if use_axes else None
        }

        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Pad":
        # Operatore Pad: genera dati, pads e constant_value come initializer
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        
        # Generate pads tensor
        pads = [random.randint(0,2) for _ in range(2*rank)]
        out_shape = [shape[i] + pads[i] + pads[i+rank] for i in range(rank)]
        pads_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(pads)], pads)
        initializers.append(pads_tensor)
        
        # Choose a mode
        mode = random.choice(["constant", "reflect", "edge"])
        node_inputs = [input_names[0], input_names[1]]
        constant_value = None # Initialize to None

        if mode == "constant":
            constant_value = round(random.uniform(-1.0, 1.0), 2) # Random constant value
            constant_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [constant_value])
            initializers.append(constant_tensor)
            node_inputs.append(input_names[2]) # Add constant_value input only for 'constant' mode
        
        # Create the Pad node with mode as attribute
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(
            op_name, 
            inputs=node_inputs, 
            outputs=[output_names[0]],
            mode=mode, # Pass mode as an attribute
            name=f"{op_name}_node_mode_{mode}"
        )
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)

        metadata = {
            "input_shapes": [shape], 
            "output_shapes": [out_shape], 
            "pads": pads, 
            "mode": mode
        }
        # Only add constant_value to metadata if it was used
        if constant_value is not None:
            metadata["constant_value"] = constant_value
            
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Reshape":
        # Primo input: dati; secondo input: nuovo shape (initializer)
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        new_shape = shape.copy()
        random.shuffle(new_shape)
        shape_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(new_shape)], new_shape)
        initializers.append(shape_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, new_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        metadata = {"input_shapes": [shape, new_shape], "output_shapes": [new_shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Resize":

        # Generate a random input tensor shape: (N=1, C=random, H=random, W=random)
        shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.ravel())
        initializers.append(init_tensor)

        # Empty ROI tensor (optional in ONNX, but included for compatibility)
        roi_name = input_names[1] + "_roi"
        roi_tensor = helper.make_tensor(roi_name, TensorProto.FLOAT, [0], [])
        initializers.append(roi_tensor)

        # Empty Scales tensor (using Sizes instead)
        scales_name = input_names[2] + "_scales"
        scales_tensor = helper.make_tensor(scales_name, TensorProto.FLOAT, [0], [])
        initializers.append(scales_tensor)

        # Compute new sizes (scaled spatial dimensions)
        scale_factor = round(random.uniform(0.5, 2.0), 2)
        new_height = int(shape[2] * scale_factor)
        new_width = int(shape[3] * scale_factor)

        sizes = [shape[0], shape[1], new_height, new_width]
        sizes_name = input_names[3] + "_sizes"
        sizes_tensor = helper.make_tensor(sizes_name, TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)

        # Output tensor info
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, sizes)

        # Choose a resize mode (explicit selection)
        mode = random.choice(["nearest", "linear"])

        # Create ONNX Resize node
        node = helper.make_node(
            op_name,
            inputs=[input_names[0], roi_name, scales_name, sizes_name],
            outputs=[output_names[0]],
            mode=mode,
            name=f"{op_name}_mode_{mode}"
        )

        # Input info placeholder (not actually used)
        input_info = helper.make_tensor_value_info("unused_input", TensorProto.FLOAT, shape)

        # Metadata dictionary
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [sizes],
            "mode": mode,
            "scale_factor": scale_factor,
        }

        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Slice":
        # Primo input: dati; gli altri due (starts ed ends) come initializer
        shape = [random.randint(5,10) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        starts, ends = [], []
        for d in shape:
            start = random.randint(0, d-1)
            end = random.randint(start+1, d)
            starts.append(start)
            ends.append(end)
        starts_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(starts)], starts)
        ends_tensor = helper.make_tensor(input_names[2], TensorProto.INT64, [len(ends)], ends)
        initializers.extend([starts_tensor, ends_tensor])
        out_shape = [ends[i] - starts[i] for i in range(rank)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]],
                                outputs=[output_names[0]], name=f"{op_name}_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "starts": starts, "ends": ends}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Split":
        # Create a more realistic neural network test for Split
        shape = [2, 7, 28, 26]  # Fixed shape to match API checks
        axis = 0  # Split along the first dimension
        
        # Create input data
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Calculate split output shapes
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] // 2
        
        # Create intermediate output names
        split_output1 = "split_output1"
        split_output2 = "split_output2"
        processed_output1 = "processed_output1"
        processed_output2 = "processed_output2"
        
        # Create the Split node
        split_node = helper.make_node(
            "Split", 
            inputs=[input_names[0]], 
            outputs=[split_output1, split_output2],
            axis=axis, 
            name=f"{op_name}_split_node",
            num_outputs=2  # Add this line
        )
        
        # Process the first split part with Relu
        relu_node = helper.make_node(
            "Relu",
            inputs=[split_output1],
            outputs=[processed_output1],
            name="Relu_after_split"
        )
        
        # Process the second split part with Sigmoid
        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=[split_output2],
            outputs=[processed_output2],
            name="Sigmoid_after_split"
        )
        
        # Combine the processed outputs with Add
        add_node = helper.make_node(
            "Add",
            inputs=[processed_output1, processed_output2],
            outputs=[output_names[0]],
            name="Add_after_processing"
        )
        
        # All nodes needed for this model
        node = [split_node, relu_node, sigmoid_node, add_node]
        
        # Create the input tensor info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        metadata = {
            "input_shapes": [shape], 
            "output_shapes": [out_shape],
            "axis": axis,
            "note": "This model splits the input, applies Relu to first part and Sigmoid to second part, then adds them together"
        }
        
        return [input_info], helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape), node, initializers, metadata

    elif op_name == "Transpose":
        # Genera una permutazione casuale per Transpose
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        perm = list(range(rank))
        random.shuffle(perm)
        out_shape = [shape[i] for i in perm]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                perm=perm, name=f"{op_name}node_perm{perm}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "perm": perm}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Unsqueeze":
        # Inserisce una dimensione in un asse casuale
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        axis = random.randint(0, rank)
        axes = [axis]
        axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        out_shape = shape.copy()
        out_shape.insert(axis, 1)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "axes": axes}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Conv":
        # Operatore Conv: genera input e pesi come initializer
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        data = np.random.randn(*input_shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        kH = random.randint(2, max(2, H//2))
        kW = random.randint(2, max(2, W//2))
        kernel_shape = [kH, kW]
        M = random.randint(1,4)
        weight_shape = [M, C, kH, kW]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        init_weight = helper.make_tensor(input_names[1], TensorProto.FLOAT, weight_shape, weight_data.flatten().tolist())
        initializers.append(init_weight)
        
        # Add strides parameter
        strides = [random.randint(1, 3), random.randint(1, 3)]
        
        # Add dilations parameter
        dilations = [random.randint(1, 2), random.randint(1, 2)]
        
        # Add padding parameter (padding for begin and end of each spatial axis)
        # Format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        pad_h = random.randint(0, 2)
        pad_w = random.randint(0, 2)
        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]  # Padding for 4D tensor [N,C,H,W]
        
        # Calculate output dimensions with strides, dilations and padding
        # Formula: output_size = (input_size + pad_begin + pad_end - (kernel_size-1)*dilation - 1) / stride + 1
        H_out = (H + 2*pad_h - (kH-1)*dilations[0] - 1) // strides[0] + 1
        W_out = (W + 2*pad_w - (kW-1)*dilations[1] - 1) // strides[1] + 1
        
        # Ensure valid output dimensions
        if H_out <= 0 or W_out <= 0:
            # Fallback to simpler case if dimensions don't work out
            dilations = [1, 1]
            pad_h = pad_w = 1
            pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
            H_out = (H + 2*pad_h - kH) // strides[0] + 1
            W_out = (W + 2*pad_w - kW) // strides[1] + 1
        
        output_shape = [N, M, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        
        # Use pads attribute in the node (simplified to just the 4 values for H,W dimensions)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                kernel_shape=kernel_shape, strides=strides, dilations=dilations,
                                pads=[pad_h, pad_w, pad_h, pad_w],
                                name=f"{op_name}node_k{kernel_shape}_s{strides}_d{dilations}_p{[pad_h, pad_w]}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        metadata = {"input_shapes": [input_shape, weight_shape], "output_shapes": [output_shape],
                    "kernel_shape": kernel_shape, "strides": strides, "dilations": dilations, 
                    "pads": [pad_h, pad_w, pad_h, pad_w]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "MatMul":
        # Genera due matrici 2D compatibili
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        A_data = np.random.randn(*A_shape).astype(np.float32)
        B_data = np.random.randn(*B_shape).astype(np.float32)
        init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
        init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
        initializers.extend([init_tensor_A, init_tensor_B])
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
        metadata = {"input_shapes": [A_shape, B_shape], "output_shapes": [[M_val, N_val]]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Gemm":
        # Gemm: esegue A * B + C
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        C_shape = [M_val, N_val]  # C must be broadcastable to (M,N)
        
        A_data = np.random.randn(*A_shape).astype(np.float32)
        B_data = np.random.randn(*B_shape).astype(np.float32)
        C_data = np.random.randn(*C_shape).astype(np.float32)
        
        init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
        init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
        init_tensor_C = helper.make_tensor(input_names[2], TensorProto.FLOAT, C_shape, C_data.flatten().tolist())
        initializers.extend([init_tensor_A, init_tensor_B, init_tensor_C])
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
        
        alpha = round(random.uniform(0.5, 2.0), 2)
        beta = round(random.uniform(0.5, 2.0), 2)
        
        # Fix: Don't use transA/transB to avoid dimension mismatches
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                                alpha=alpha, beta=beta, 
                                name=f"{op_name}node_alpha{alpha}beta{beta}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
        metadata = {"input_shapes": [A_shape, B_shape, C_shape], "output_shapes": [[M_val, N_val]],
                    "alpha": alpha, "beta": beta}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "MaxPool":
        # Create a simple MaxPool test with predictable dimensions and parameters
        # Fixed input dimensions
        N = 1
        C = 1  # Single channel for simplicity
        H = 4  # Small height
        W = 4  # Small width
        input_shape = [N, C, H, W]
        
        # Create a simple input pattern with predictable values
        data = np.zeros(input_shape, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                data[0, 0, i, j] = float(i * W + j + 1)  # Simple increasing values
        
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Use simple 2x2 kernel
        kernel_shape = [2, 2]
        
        # Use stride of 1
        strides = [1, 1]
        
        # No padding
        pads = [0, 0, 0, 0]  # [pad_top, pad_left, pad_bottom, pad_right]
        
        # Calculate output dimensions
        H_out = ((H - kernel_shape[0]) // strides[0]) + 1
        W_out = ((W - kernel_shape[1]) // strides[1]) + 1
        
        output_shape = [N, C, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                              kernel_shape=kernel_shape, 
                              strides=strides, 
                              pads=pads,
                              auto_pad="NOTSET",
                              name=f"{op_name}node_k{kernel_shape}_s{strides}_p{pads}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        metadata = {
            "input_shapes": [input_shape], 
            "output_shapes": [output_shape],
            "kernel_shape": kernel_shape, 
            "strides": strides, 
            "pads": pads,
            "auto_pad": "NOTSET"
        }
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "AveragePool":
        N, C = 1, 1
        H = 4
        W = 4
        input_shape = [N, C, H, W]

        # Create input data with predictable values
        data = np.zeros(input_shape, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                data[0, 0, i, j] = float(i * W + j + 1)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)

        # Randomized parameters
        kernel_shape = [random.randint(2, 3), random.randint(2, 3)]
        strides = [random.randint(1, 2), random.randint(1, 2)]
        count_include_pad = random.choice([0, 1])  # 0 = False, 1 = True
        auto_pad = random.choice(["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"])  # Random auto_pad
        if auto_pad == "NOTSET":
            pads = [random.randint(0, 1) for _ in range(4)]
        else:
            pads = [0, 0, 0, 0]  # Default

        # Output dimensions calculation
        if auto_pad == "NOTSET":
            # Standard calculation with explicit pads
            H_out = ((H + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
            W_out = ((W + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1
        elif auto_pad == "VALID":
            # No padding
            H_out = ((H - kernel_shape[0]) // strides[0]) + 1
            W_out = ((W - kernel_shape[1]) // strides[1]) + 1
        else:  # SAME_UPPER or SAME_LOWER
            # Maintain output dimensions similar to input
            H_out = int(np.ceil(H / strides[0]))
            W_out = int(np.ceil(W / strides[1]))
            # Calculate total padding (for reference, not used in the node)
            pad_h = (H_out - 1) * strides[0] + kernel_shape[0] - H
            pad_w = (W_out - 1) * strides[1] + kernel_shape[1] - W
            pads = [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]  # Only for metadata

        output_shape = [N, C, H_out, W_out]
        if H_out <= 0 or W_out <= 0:
            # Avoid invalid shapes
            kernel_shape = [2, 2]  # Fallback
            strides = [1, 1]
            auto_pad = "NOTSET"
            pads = [0, 0, 0, 0]
            H_out = ((H + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
            W_out = ((W + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1
            output_shape = [N, C, H_out, W_out]

        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                kernel_shape=kernel_shape,
                                strides=strides,
                                pads=pads,
                                count_include_pad=count_include_pad,
                                auto_pad=auto_pad,
                                name=f"{op_name}node_k{kernel_shape}_s{strides}_p{pads}_c{count_include_pad}_ap{auto_pad}")

        metadata = {
            "input_shapes": [input_shape],
            "output_shapes": [output_shape],
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "auto_pad": auto_pad,
            "count_include_pad": count_include_pad
        }

        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Mean":
        num_inputs = random.randint(1, 5)
        max_dims = 3 
        
        # 1. Generate a potential "output" shape first
        output_shape = [random.randint(1, 4) for _ in range(max_dims)]

        shapes = []
        initializers = [] # Ensure initializers is defined here
        
        for i in range(num_inputs):
            # 2. Derive compatible input shape from the output shape
            current_shape = []
            for dim_size in output_shape:
                # Each dimension is either the same as output_shape or 1
                current_shape.append(random.choice([1, dim_size]))
            shapes.append(current_shape)

            # data generation for each input tensor
            data = np.random.randn(*current_shape).astype(np.float32)
            tensor_name = input_names[i]
            init_tensor = helper.make_tensor(tensor_name, TensorProto.FLOAT, current_shape, data.flatten().tolist())
            initializers.append(init_tensor) # Now append to the locally defined list
        
        # The actual output shape is already determined by output_shape list
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        node = helper.make_node(
            op_name,
            inputs=[input_names[i] for i in range(num_inputs)],
            outputs=[output_names[0]],
            name=f"{op_name}_node_{num_inputs}_inputs"
        )
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shapes[0])
        metadata = {
            "input_shapes": shapes,
            "output_shapes": [output_shape]
        }
        
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Clip":
        # Clip operator: clips values between min and max
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32) * 10 # Scale data to have values outside typical clip range
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        node_inputs = [input_names[0]]
        min_val = None
        max_val = None
        clip_metadata = {}

        # Randomly include min value
        if random.choice([True, False]):
            min_val = round(random.uniform(-5.0, 0.0), 2)
            min_tensor = helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [min_val])
            initializers.append(min_tensor)
            node_inputs.append(input_names[1])
            clip_metadata["min_value"] = min_val
            
        # Randomly include max value
        if random.choice([True, False]):
            # Ensure max_val is greater than min_val if min_val exists
            lower_bound_for_max = min_val if min_val is not None else 0.1
            max_val = round(random.uniform(lower_bound_for_max, 5.0), 2) 
            max_tensor_input_index = len(node_inputs) # Determine correct input index for max
            max_tensor = helper.make_tensor(input_names[max_tensor_input_index], TensorProto.FLOAT, [], [max_val])
            initializers.append(max_tensor)
            node_inputs.append(input_names[max_tensor_input_index])
            clip_metadata["max_value"] = max_val
            
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=node_inputs, outputs=[output_names[0]], 
                                name=f"{op_name}_node")

        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape], **clip_metadata} # Merge clip specific metadata
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "BatchNormalization":

        # BatchNorm has 5 inputs: X, scale, B, mean, var
        # Output shape is the same as input
        shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
        C = shape[1]  # Number of channels (second dimension)

        # Create input tensor
        data_X = np.random.randn(*shape).astype(np.float32)
        init_X = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data_X.flatten().tolist())

        # scale (gamma), bias (beta), mean, var — all of shape [C]
        scale = np.random.randn(C).astype(np.float32)
        B = np.random.randn(C).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        var = np.abs(np.random.randn(C)).astype(np.float32)  # ensure variance is non-negative

        init_scale = helper.make_tensor(input_names[1], TensorProto.FLOAT, [C], scale.tolist())
        init_B = helper.make_tensor(input_names[2], TensorProto.FLOAT, [C], B.tolist())
        init_mean = helper.make_tensor(input_names[3], TensorProto.FLOAT, [C], mean.tolist())
        init_var = helper.make_tensor(input_names[4], TensorProto.FLOAT, [C], var.tolist())

        initializers.extend([init_X, init_scale, init_B, init_mean, init_var])

        epsilon = round(random.uniform(1e-5, 1e-2), 6)
        momentum = round(random.uniform(0.8, 0.99), 3)

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(
            "BatchNormalization",
            inputs=input_names[:5],
            outputs=[output_names[0]],
            epsilon=epsilon,
            momentum=momentum,
            name=f"{op_name}_node"
        )

        metadata = {
            "input_shapes": [shape, [C], [C], [C], [C]],
            "output_shapes": [shape],
            "epsilon": epsilon,
            "momentum": momentum
        }

        return [input_info], output_info, [node], initializers, metadata
    else:
        # Caso di fallback per operatori non gestiti esplicitamente
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                name=f"{op_name}_generic_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata

def generate_model(op_name, filename, model_id=0):
    """Crea e salva un modello ONNX."""
    input_info, output_info, nodes, initializers, metadata = generate_fuzz_model(op_name)
    
    # Gli output vanno sempre definiti
    graph_outputs = output_info if isinstance(output_info, list) else [output_info]
    
    for node in nodes:
        node.doc_string = f"Test node for {op_name} operation with ID {model_id}"
    
    graph = helper.make_graph(
        nodes,
        name=f"{op_name}test_graph{model_id}",
        inputs=input_info,  # In questo caso vuota, poiché tutti gli input sono initializer
        outputs=graph_outputs,
        initializer=initializers,
        doc_string=f"Test graph for {op_name} operation with configuration: {metadata}"
    )
    
    opset_imports = [
        helper.make_opsetid("", 13),  # Standard ONNX opset
        helper.make_opsetid("", 20)
    ]
    
    model = helper.make_model(
        graph, 
        producer_name='zant_test_generator',
        producer_version='1.0',
        domain='ai.zant.test',
        model_version=model_id,
        doc_string=f"Test model for {op_name} operation. Generated on {datetime.datetime.now().isoformat()}",
        opset_imports=opset_imports,
        ir_version=6  # Explicitly set IR version 6 which corresponds to opset 10
    )
    model = onnx.shape_inference.infer_shapes(model)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_metadata"
    meta_prop.value = str(metadata)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_op"
    meta_prop.value = op_name
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_id"
    meta_prop.value = str(model_id)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "generator_version"
    meta_prop.value = "1.0"
    model.metadata_props.append(meta_prop)
    
    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"Fuzzed model for {op_name} (ID: {model_id}) saved to: {filename}")
    return metadata

def run_model(filename):
    """
    Esegue il modello ONNX.
    Poiché tutti gli input sono definiti come initializer, il feed del runtime sarà vuoto.
    """
    model = onnx.load(filename)
    graph = model.graph
    # Ottieni i nomi degli initializer (gli input runtime saranno vuoti)
    initializer_names = [init.name for init in graph.initializer]
    runtime_inputs = [inp for inp in graph.input if inp.name not in initializer_names]
    
    input_data = {}
    for inp in runtime_inputs:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT:
            # Fix: Use astype(np.float32) instead of default np.float64
            data = np.random.randn(*shape).astype(np.float32)
        elif elem_type == TensorProto.INT64:
            data = np.random.randint(0, 10, size=shape, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported input type: {elem_type}")
        input_data[inp.name] = data
    
    session = ort.InferenceSession(filename)
    output_names = [out.name for out in graph.output]
    outputs = session.run(output_names, input_data)
    
    # Convert numpy arrays to lists for JSON serialization
    outputs_dict = {name: output.tolist() for name, output in zip(output_names, outputs)}
    input_data_dict = {name: data.tolist() for name, data in input_data.items()}
    
    return {"inputs": input_data_dict, "outputs": outputs_dict}

def load_supported_ops(filename="tests/CodeGen/Python-ONNX/available_operations.txt"):
    """Carica le operazioni supportate da un file oppure restituisce una lista di default."""
    try:
        with open(filename, "r") as file:
            ops = [line.strip() for line in file if line.strip()]
            # Remove the problematic Shape operator for now
            if "Shape" in ops:
                ops.remove("Shape")
            return ops
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return [
            "LeakyRelu", "Relu", "Sigmoid", "Softmax", "Add", "Ceil", "Div", "Mul", "Sub", "Tanh",
            "Concat", "Gather", "Identity", "Neg", "Reshape", "Resize", "Slice", 
            "Split", "Transpose", "Unsqueeze", "ReduceMean", "Conv", "MatMul", "Gemm", "MaxPool",
            "Clip"
            # "Shape" removed from the list
        ]


def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description="Generate fuzzed ONNX models and save execution data in JSON.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of models to generate for each operation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random generation (for reproducibility).")
    parser.add_argument("--output-dir", type=str, default="datasets/oneOpModels",
                        help="Directory to save generated models.")
    parser.add_argument("--metadata-file", type=str, default="datasets/oneOpModels/results.json",
                        help="File to save metadata and execution data.")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    output_dir = args.output_dir
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
    os.makedirs(output_dir, exist_ok=True)
    
    supported_ops = load_supported_ops()
    
    print(f"\n supported_ops : {supported_ops}")
    
    all_models = []
    
    for op in supported_ops:
        for i in range(args.iterations):
            print("Saving model to " + output_dir) 
            filename = f"{output_dir}{op}_{i}.onnx"
            try: 
                metadata = generate_model(op, filename, i)
                print(f"Successfully generated model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error generating model for {op} (ID: {i}): {e}")

            try:
                data = run_model(filename)
                model_info = {
                    "operation": op,
                    "model_id": i,
                    "inputs": data["inputs"],
                    "outputs": data["outputs"],
                    "metadata": metadata
                }
                
                test_file_name = f"{output_dir}{op}_{i}_user_tests.json"
                print(f"Saving user tests to {test_file_name}")
                
                user_tests = []

                for (in_key, out_key) in zip(data["inputs"].keys(), data["outputs"].keys()):
                    in_array = np.array(data["inputs"][in_key]).flatten().tolist()
                    out_array = np.array(data["outputs"][out_key]).flatten().tolist()
                    
                    test_model_info = {
                        "name": op,
                        "type": "exact",
                        "input": in_array,
                        "output": out_array,
                        "expected_class": 0
                    }
                    user_tests.append(test_model_info)

                with open(test_file_name, 'w') as f:
                    json.dump(user_tests, f, indent=2)
                print(f"Execution data saved to {test_file_name}")
                    
                
                all_models.append(model_info)
                print(f"Successfully ran model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error running model for {op} (ID: {i}): {e}")
    
    with open(args.metadata_file, 'w') as f:
        json.dump(all_models, f, indent=2)
    print(f"Execution data saved to {args.metadata_file}")

if __name__ == "__main__":
    main()