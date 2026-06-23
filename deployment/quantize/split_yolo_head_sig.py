"""Split YOLO ONNX: remove NMS, output boxes (raw LTRB) + sigmoid(scores).

Differs from split_yolo_head.py: adds Sigmoid to the class score output so the
quantized NB outputs scores in [0,1] instead of raw logits in [-131, 3.2].
This prevents uint8 quantization from corrupting per-channel class scores.

Input:  Ultralytics YOLO11 export with NMS (output0: [1,300,6])
Output: boxes [1,4,N] (raw LTRB, NO sigmoid) + scores [1,nc,N] (WITH sigmoid)
"""

import argparse, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import onnx
from onnx import helper, checker, shape_inference
import numpy as np


def split_head_sigmoid(input_path, output_path):
    model = onnx.load(input_path)

    concat_boxes = None
    concat_scores = None
    sigmoid_node = None

    for node in model.graph.node:
        if node.op_type != 'Concat':
            continue
        name = node.name
        if name == '/model.23/Concat' and name != '/model.23/Concat_1':
            inputs = list(node.input)
            if len(inputs) == 3 and all('Reshape' in inp for inp in inputs):
                concat_boxes = node
        elif name == '/model.23/Concat_1':
            inputs = list(node.input)
            if len(inputs) == 3 and all('Reshape' in inp for inp in inputs):
                concat_scores = node

    # Find the Sigmoid applied to scores (after Concat_1)
    for node in model.graph.node:
        if node.op_type == 'Sigmoid' and concat_scores.output[0] in node.input:
            sigmoid_node = node
            break

    if concat_boxes is None:
        raise ValueError("Could not find box Concat node")
    if concat_scores is None:
        raise ValueError("Could not find score Concat node")

    print(f"Boxes concat: {concat_boxes.name} -> {concat_boxes.output}")
    print(f"Scores concat: {concat_scores.name} -> {concat_scores.output}")
    print(f"Sigmoid after scores: {sigmoid_node.name if sigmoid_node else 'NOT FOUND'}")

    boxes_output_name = concat_boxes.output[0]
    scores_concat_output_name = concat_scores.output[0]

    # Determine nc from inferred concat output shape, then fall back to the
    # score reshape constant. BSD defaults to 4 classes.
    nc = 4
    try:
        inferred = shape_inference.infer_shapes(model)
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
            if vi.name == scores_concat_output_name:
                dims = vi.type.tensor_type.shape.dim
                if len(dims) >= 2 and dims[1].dim_value > 0:
                    nc = int(dims[1].dim_value)
                    break
    except Exception as e:
        print(f"Shape inference warning: {e}")

    for node in model.graph.node:
        if node.op_type == 'Reshape' and node.output[0] in concat_scores.input:
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    shape = onnx.numpy_helper.to_array(init)
                    if len(shape) >= 2 and int(shape[1]) <= 16:
                        nc = int(shape[1])
                    break
            break
    print(f"Detected nc={nc}")

    # Insert Sigmoid node after scores Concat
    sigmoid_output_name = 'scores'
    sig_node = helper.make_node(
        'Sigmoid',
        inputs=[scores_concat_output_name],
        outputs=[sigmoid_output_name],
        name='/model.23/sigmoid_for_scores'
    )
    model.graph.node.append(sig_node)
    print(f"Inserted Sigmoid: {scores_concat_output_name} -> {sigmoid_output_name}")

    # Create output value_info
    boxes_vi = helper.make_tensor_value_info('boxes', 1, ['batch', 4, 'anchors'])
    scores_vi = helper.make_tensor_value_info('scores', 1, ['batch', nc, 'anchors'])

    del model.graph.output[:]
    model.graph.output.extend([boxes_vi, scores_vi])

    # Collect all nodes needed (BFS from outputs)
    keep_outputs = {boxes_output_name, scores_concat_output_name}

    node_by_output = {}
    for node in model.graph.node:
        for out in node.output:
            node_by_output[out] = node

    init_names = {i.name for i in model.graph.initializer}
    input_names = {i.name for i in model.graph.input}

    nodes_to_keep = set()
    outputs_to_process = list(keep_outputs)

    while outputs_to_process:
        out_name = outputs_to_process.pop(0)
        if out_name in node_by_output:
            node = node_by_output[out_name]
            if node.name not in nodes_to_keep:
                nodes_to_keep.add(node.name)
                for inp in node.input:
                    if inp not in init_names and inp not in input_names:
                        outputs_to_process.append(inp)

    # Also keep the newly added sigmoid node
    nodes_to_keep.add(sig_node.name)

    new_nodes = [n for n in model.graph.node if n.name in nodes_to_keep]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Rename box concat output to 'boxes'
    for node in model.graph.node:
        for k, out in enumerate(node.output):
            if out == boxes_output_name:
                node.output[k] = 'boxes'
        for k, inp in enumerate(node.input):
            if inp == boxes_output_name:
                node.input[k] = 'boxes'

    # Clean up unused initializers
    used_names = set()
    for node in model.graph.node:
        for name in node.input:
            used_names.add(name)
        for name in node.output:
            used_names.add(name)
    for inp in model.graph.input:
        used_names.add(inp.name)
    for out in model.graph.output:
        used_names.add(out.name)

    new_initializers = [init for init in model.graph.initializer if init.name in used_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    # Check and save
    try:
        checker.check_model(model)
        print("Model check passed")
    except Exception as e:
        print(f"Model check warning: {e}")

    onnx.save(model, output_path)
    print(f"Saved: {output_path}")
    print(f"Outputs: boxes (raw LTRB, no sigmoid), scores (sigmoid'ed, [0,1])")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input ONNX with NMS')
    parser.add_argument('--output', required=True, help='Output ONNX (boxes raw + scores sigmoid)')
    args = parser.parse_args()
    split_head_sigmoid(args.input, args.output)
