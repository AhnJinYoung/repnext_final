"""Patch depthwise-shaped Conv2D ops in a TFLite file.

onnx2tf can encode depthwise grouped convolutions as CONV_2D version 6 with
filters shaped [out_channels, height, width, 1]. EdgeTPU compiler v16 rejects
that opcode version. This script rewrites those ops to DEPTHWISE_CONV_2D and
then lowers the remaining Conv2D opcode version for compiler compatibility.
"""
import argparse
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


CONV_2D = schema.BuiltinOperator.CONV_2D
DEPTHWISE_CONV_2D = schema.BuiltinOperator.DEPTHWISE_CONV_2D


def tensor_shape(tensor):
    return [int(v) for v in tensor.shape]


def patch_model(src_path, dst_path, conv_version):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(src_path.read_bytes(), 0))
    subgraph = model.subgraphs[0]

    depthwise_opcode_index = None
    conv_opcode_index = None
    for idx, opcode in enumerate(model.operatorCodes):
        if opcode.builtinCode == DEPTHWISE_CONV_2D:
            depthwise_opcode_index = idx
        if opcode.builtinCode == CONV_2D:
            conv_opcode_index = idx
            opcode.version = conv_version

    if depthwise_opcode_index is None:
        depthwise_opcode = schema.OperatorCodeT()
        depthwise_opcode.builtinCode = DEPTHWISE_CONV_2D
        depthwise_opcode.version = 3
        model.operatorCodes.append(depthwise_opcode)
        depthwise_opcode_index = len(model.operatorCodes) - 1

    patched = 0
    skipped = 0
    for op in subgraph.operators:
        if op.opcodeIndex != conv_opcode_index:
            continue
        if len(op.inputs) < 3 or len(op.outputs) < 1:
            skipped += 1
            continue

        input_tensor = subgraph.tensors[op.inputs[0]]
        filter_tensor = subgraph.tensors[op.inputs[1]]
        output_tensor = subgraph.tensors[op.outputs[0]]
        input_shape = tensor_shape(input_tensor)
        filter_shape = tensor_shape(filter_tensor)
        output_shape = tensor_shape(output_tensor)

        is_depthwise_shape = (
            len(input_shape) == 4
            and len(filter_shape) == 4
            and len(output_shape) == 4
            and filter_shape[3] == 1
            and input_shape[3] == filter_shape[0]
            and output_shape[3] == filter_shape[0]
        )
        if not is_depthwise_shape:
            continue

        conv_options = op.builtinOptions
        depthwise_options = schema.DepthwiseConv2DOptionsT()
        depthwise_options.padding = conv_options.padding
        depthwise_options.strideW = conv_options.strideW
        depthwise_options.strideH = conv_options.strideH
        depthwise_options.depthMultiplier = 1
        depthwise_options.fusedActivationFunction = conv_options.fusedActivationFunction
        depthwise_options.dilationWFactor = conv_options.dilationWFactor
        depthwise_options.dilationHFactor = conv_options.dilationHFactor

        buffer = model.buffers[filter_tensor.buffer]
        weights = np.asarray(buffer.data, dtype=np.int8).reshape(filter_shape)
        buffer.data = np.transpose(weights, (3, 1, 2, 0)).reshape(-1)
        filter_tensor.shape = np.array([1, filter_shape[1], filter_shape[2], filter_shape[0]], dtype=np.int32)
        if filter_tensor.quantization is not None:
            filter_tensor.quantization.quantizedDimension = 3

        op.opcodeIndex = depthwise_opcode_index
        op.builtinOptionsType = schema.BuiltinOptions.DepthwiseConv2DOptions
        op.builtinOptions = depthwise_options
        patched += 1

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), b"TFL3")
    dst_path.write_bytes(bytes(builder.Output()))
    return patched, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--conv-version", type=int, default=3)
    args = parser.parse_args()

    patched, skipped = patch_model(args.src, args.dst, args.conv_version)
    print(f"patched_depthwise_conv2d={patched}")
    print(f"skipped_conv2d={skipped}")
    print(f"wrote={args.dst}")


if __name__ == "__main__":
    main()
