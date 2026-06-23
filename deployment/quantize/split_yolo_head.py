"""Split YOLO ONNX: remove NMS, output boxes + scores separately.

Input:  Ultralytics YOLO11 export with NMS (output0: [1,300,6])
Output: boxes [1,4,N] + scores [1,nc,N] (raw logit, before Sigmoid)
"""

import argparse
import onnx
from onnx import helper, checker, shape_inference


def split_head(input_path, output_path):
    model = onnx.load(input_path)

    # Find key nodes
    concat_boxes = None
    concat_scores = None
    sigmoid_node = None
    nms_concat = None

    for node in model.graph.node:
        if node.op_type != 'Concat':
            continue
        # Box concat: /model.23/Concat combines 3 reshape outputs (boxes)
        # Score concat: /model.23/Concat_1 combines 3 reshape outputs (scores before sigmoid)
        name = node.name
        if name.endswith('/Concat') and not name.endswith('/Concat_1') and not name.endswith('/Concat_2') and not name.endswith('/Concat_3') and not name.endswith('/Concat_6'):
            # The main Concat that merges 3 detection head outputs
            inputs = list(node.input)
            if len(inputs) == 3 and all('Reshape' in inp for inp in inputs):
                concat_boxes = node
        elif name.endswith('/Concat_1'):
            inputs = list(node.input)
            if len(inputs) == 3 and all('Reshape' in inp for inp in inputs):
                concat_scores = node

    # Find the sigmoid that's applied to scores
    for node in model.graph.node:
        if node.op_type == 'Sigmoid' and concat_scores and node.input[0] == concat_scores.output[0]:
            sigmoid_node = node
            break

    if concat_boxes is None or concat_scores is None:
        # Fallback: search by pattern
        for node in model.graph.node:
            if node.name == '/model.23/Concat':
                concat_boxes = node
            if node.name == '/model.23/Concat_1':
                concat_scores = node

    if concat_boxes is None:
        raise ValueError("Could not find box concat node")
    if concat_scores is None:
        raise ValueError("Could not find score concat node")

    print(f"Boxes node: {concat_boxes.name} -> {concat_boxes.output}")
    print(f"Scores node: {concat_scores.name} -> {concat_scores.output}")

    boxes_output_name = concat_boxes.output[0]
    scores_output_name = concat_scores.output[0]

    # Determine shapes from the reshape constants
    # Boxes: [1, 4, 8400], Scores: [1, nc, 8400]

    # Create new graph outputs
    # Find existing shapes by looking at the constant shape tensors
    box_reshape = None
    score_reshape = None
    for node in model.graph.node:
        if node.op_type == 'Reshape' and node.output[0] in concat_boxes.input:
            box_reshape = node
        if node.op_type == 'Reshape' and node.output[0] in concat_scores.input:
            score_reshape = node

    # Determine number of classes from inferred concat output shape.
    nc = 4  # BSD default
    try:
        inferred = shape_inference.infer_shapes(model)
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
            if vi.name == scores_output_name:
                dims = vi.type.tensor_type.shape.dim
                if len(dims) >= 2 and dims[1].dim_value > 0:
                    nc = int(dims[1].dim_value)
                    break
    except Exception as e:
        print(f"Shape inference warning: {e}")

    if score_reshape:
        for init in model.graph.initializer:
            if init.name == score_reshape.input[1]:
                import numpy as np
                shape = onnx.numpy_helper.to_array(init)
                if len(shape) >= 2 and int(shape[1]) <= 16:
                    nc = int(shape[1])
                break

    print(f"Detected nc={nc}")

    # Create value_info for outputs
    boxes_vi = helper.make_tensor_value_info('boxes', 1, ['batch', 4, 'anchors'])
    scores_vi = helper.make_tensor_value_info('scores', 1, ['batch', nc, 'anchors'])

    # Replace graph outputs
    del model.graph.output[:]
    model.graph.output.extend([boxes_vi, scores_vi])

    # Collect nodes to keep (everything that feeds into boxes or scores outputs)
    keep_outputs = {boxes_output_name, scores_output_name}

    # BFS to collect all nodes needed
    nodes_to_keep = set()
    outputs_to_process = list(keep_outputs)

    node_by_output = {}
    for node in model.graph.node:
        for out in node.output:
            node_by_output[out] = node

    init_names = {i.name for i in model.graph.initializer}
    input_names = {i.name for i in model.graph.input}

    while outputs_to_process:
        out_name = outputs_to_process.pop(0)
        if out_name in node_by_output:
            node = node_by_output[out_name]
            if node.name not in nodes_to_keep:
                nodes_to_keep.add(node.name)
                for inp in node.input:
                    if inp not in init_names and inp not in input_names:
                        outputs_to_process.append(inp)

    # Keep only needed nodes
    new_nodes = [n for n in model.graph.node if n.name in nodes_to_keep]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Rename concat outputs to match graph output names
    old_box_name = boxes_output_name
    old_score_name = scores_output_name
    for node in model.graph.node:
        for k, out in enumerate(node.output):
            if out == old_box_name:
                node.output[k] = 'boxes'
            elif out == old_score_name:
                node.output[k] = 'scores'
        for k, inp in enumerate(node.input):
            if inp == old_box_name:
                node.input[k] = 'boxes'
            elif inp == old_score_name:
                node.input[k] = 'scores'

    # Clean up unused initializers and value_info
    used_names = set()
    for node in model.graph.node:
        for name in node.input:
            used_names.add(name)
        for name in node.output:
            used_names.add(name)

    # Keep input and output names
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input ONNX with NMS')
    parser.add_argument('--output', required=True, help='Output ONNX (boxes + scores)')
    args = parser.parse_args()
    split_head(args.input, args.output)
