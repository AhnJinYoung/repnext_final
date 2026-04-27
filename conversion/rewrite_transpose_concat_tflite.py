#!/usr/bin/env python3
"""Remove exact transpose layout islands around Concat and Split from TFLite.

This rewrite is mathematically exact for patterns like:

    concat(transpose_p(x0), transpose_p(x1), ..., axis=a) -> transpose_inv_p

which becomes:

    concat(x0, x1, ..., axis=p[a])

The concat output is changed to the inverse-transpose tensor, preserving the
downstream shape/layout. Only single-consumer transpose inputs and a single-consumer
concat output transpose are rewritten.

It also rewrites:

    split(transpose_p(x), axis=a) -> transpose_inv_p(each output)

into:

    split(x, axis=p[a])
"""
import argparse
from collections import defaultdict
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


TRANSPOSE = schema.BuiltinOperator.TRANSPOSE
CONCAT = schema.BuiltinOperator.CONCATENATION
SPLIT = schema.BuiltinOperator.SPLIT


def read_int32_buffer(model, tensor):
    buf = model.buffers[tensor.buffer]
    if buf.data is None or len(buf.data) == 0:
        return None
    raw = np.array(buf.data, dtype=np.uint8).tobytes()
    return tuple(int(x) for x in np.frombuffer(raw, dtype=np.int32).tolist())


def write_int32_buffer(model, tensor, values):
    arr = np.array(values, dtype=np.int32)
    model.buffers[tensor.buffer].data = np.frombuffer(arr.tobytes(), dtype=np.uint8)


def is_inverse(p, q):
    if p is None or q is None or len(p) != len(q):
        return False
    return tuple(p[i] for i in q) == tuple(range(len(p)))


def patch_model(src_path, dst_path):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(src_path.read_bytes(), 0))
    sg = model.subgraphs[0]
    opcodes = [model.operatorCodes[op.opcodeIndex].builtinCode for op in sg.operators]

    consumers = defaultdict(list)
    producer = {}
    for idx, op in enumerate(sg.operators):
        for tensor_idx in op.inputs:
            if tensor_idx is not None and tensor_idx >= 0:
                consumers[tensor_idx].append(idx)
        for tensor_idx in op.outputs:
            if tensor_idx is not None and tensor_idx >= 0:
                producer[tensor_idx] = idx

    remove_ops = set()
    rewritten_concat = 0
    rewritten_split = 0
    skipped = 0

    for concat_idx, concat_op in enumerate(sg.operators):
        if opcodes[concat_idx] != CONCAT or concat_idx in remove_ops:
            continue
        if len(concat_op.outputs) != 1:
            skipped += 1
            continue

        concat_out = concat_op.outputs[0]
        concat_out_consumers = consumers.get(concat_out, [])
        if len(concat_out_consumers) != 1:
            skipped += 1
            continue
        out_transpose_idx = concat_out_consumers[0]
        if opcodes[out_transpose_idx] != TRANSPOSE:
            skipped += 1
            continue
        out_transpose = sg.operators[out_transpose_idx]

        input_transposes = []
        input_tensors = []
        perm = None
        ok = True
        for tensor_idx in concat_op.inputs:
            prod_idx = producer.get(tensor_idx)
            if prod_idx is None or opcodes[prod_idx] != TRANSPOSE:
                ok = False
                break
            if len(consumers.get(tensor_idx, [])) != 1:
                ok = False
                break
            trans = sg.operators[prod_idx]
            this_perm = read_int32_buffer(model, sg.tensors[trans.inputs[1]])
            if perm is None:
                perm = this_perm
            elif this_perm != perm:
                ok = False
                break
            input_transposes.append(prod_idx)
            input_tensors.append(trans.inputs[0])
        if not ok or perm is None:
            skipped += 1
            continue

        out_perm = read_int32_buffer(model, sg.tensors[out_transpose.inputs[1]])
        if not is_inverse(perm, out_perm):
            skipped += 1
            continue

        options = concat_op.builtinOptions
        old_axis = int(options.axis)
        if old_axis < 0:
            old_axis += len(perm)
        if old_axis < 0 or old_axis >= len(perm):
            skipped += 1
            continue
        new_axis = int(perm[old_axis])

        new_options = schema.ConcatenationOptionsT()
        new_options.axis = new_axis
        new_options.fusedActivationFunction = options.fusedActivationFunction

        concat_op.inputs = np.array(input_tensors, dtype=np.int32)
        concat_op.outputs = np.array([out_transpose.outputs[0]], dtype=np.int32)
        concat_op.builtinOptions = new_options

        remove_ops.add(out_transpose_idx)
        remove_ops.update(input_transposes)
        rewritten_concat += 1

    # Recompute graph after concat rewrites before looking for split islands.
    total_removed = len(remove_ops)
    if remove_ops:
        active_ops = [op for idx, op in enumerate(sg.operators) if idx not in remove_ops]
    else:
        active_ops = list(sg.operators)
    sg.operators = active_ops
    opcodes = [model.operatorCodes[op.opcodeIndex].builtinCode for op in sg.operators]
    consumers = defaultdict(list)
    producer = {}
    for idx, op in enumerate(sg.operators):
        for tensor_idx in op.inputs:
            if tensor_idx is not None and tensor_idx >= 0:
                consumers[tensor_idx].append(idx)
        for tensor_idx in op.outputs:
            if tensor_idx is not None and tensor_idx >= 0:
                producer[tensor_idx] = idx

    remove_ops = set()
    for split_idx, split_op in enumerate(sg.operators):
        if opcodes[split_idx] != SPLIT or len(split_op.inputs) < 2:
            continue
        axis_tensor_idx = split_op.inputs[0]
        value_tensor_idx = split_op.inputs[1]
        value_prod_idx = producer.get(value_tensor_idx)
        if value_prod_idx is None or opcodes[value_prod_idx] != TRANSPOSE:
            continue
        if len(consumers.get(value_tensor_idx, [])) != 1:
            continue
        in_transpose = sg.operators[value_prod_idx]
        perm = read_int32_buffer(model, sg.tensors[in_transpose.inputs[1]])
        if perm is None:
            continue
        axis_values = read_int32_buffer(model, sg.tensors[axis_tensor_idx])
        if axis_values is None or len(axis_values) != 1:
            continue
        old_axis = int(axis_values[0])
        if old_axis < 0:
            old_axis += len(perm)
        if old_axis < 0 or old_axis >= len(perm):
            continue
        new_axis = int(perm[old_axis])

        new_outputs = []
        output_transposes = []
        ok = True
        for out_tensor_idx in split_op.outputs:
            out_consumers = consumers.get(out_tensor_idx, [])
            if len(out_consumers) != 1:
                ok = False
                break
            trans_idx = out_consumers[0]
            if opcodes[trans_idx] != TRANSPOSE:
                ok = False
                break
            trans = sg.operators[trans_idx]
            out_perm = read_int32_buffer(model, sg.tensors[trans.inputs[1]])
            if not is_inverse(perm, out_perm):
                ok = False
                break
            output_transposes.append(trans_idx)
            new_outputs.append(trans.outputs[0])
        if not ok:
            continue

        split_op.inputs[1] = in_transpose.inputs[0]
        split_op.outputs = np.array(new_outputs, dtype=np.int32)
        write_int32_buffer(model, sg.tensors[axis_tensor_idx], [new_axis])
        remove_ops.add(value_prod_idx)
        remove_ops.update(output_transposes)
        rewritten_split += 1

    if remove_ops:
        sg.operators = [op for idx, op in enumerate(sg.operators) if idx not in remove_ops]
        total_removed += len(remove_ops)

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), b"TFL3")
    dst_path.write_bytes(bytes(builder.Output()))
    return rewritten_concat, rewritten_split, total_removed, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    rewritten_concat, rewritten_split, removed, skipped = patch_model(args.src, args.dst)
    print(f"rewritten_concat_islands={rewritten_concat}")
    print(f"rewritten_split_islands={rewritten_split}")
    print(f"removed_transpose_ops={removed}")
    print(f"skipped_concat_candidates={skipped}")
    print(f"wrote={args.dst}")


if __name__ == "__main__":
    main()
