"""Convert RepNeXt ONNX to INT8 TFLite and compile it for Coral EdgeTPU."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def run(cmd, cwd=None, log_path=None):
    print("[run]", " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout)
    if log_path:
        Path(log_path).write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd[0]}")
    return proc.stdout


def load_image(path, size):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required when --calib-dir is used") from exc
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = (arr - np.array(DEFAULT_MEAN, dtype="float32")) / np.array(DEFAULT_STD, dtype="float32")
    return arr[np.newaxis, ...]


def representative_dataset(calib_dir, size, samples, input_layout):
    image_paths = []
    if calib_dir:
        root = Path(calib_dir)
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(root.rglob(suffix))
        image_paths = sorted(image_paths)[:samples]

    def gen():
        if image_paths:
            for path in image_paths:
                arr = load_image(path, size)
                if input_layout == "nchw":
                    arr = np.transpose(arr, (0, 3, 1, 2))
                yield [arr.astype("float32")]
            return
        rng = np.random.default_rng(0)
        for _ in range(samples):
            shape = (1, 3, size, size) if input_layout == "nchw" else (1, size, size, 3)
            yield [rng.normal(0, 1, shape).astype("float32")]

    return gen


def convert_saved_model_to_int8(saved_model, tflite_path, calib_dir, size, samples, input_layout):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(calib_dir, size, samples, input_layout)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    Path(tflite_path).write_bytes(tflite_model)
    print(f"[done] wrote {tflite_path} ({Path(tflite_path).stat().st_size / 1e6:.1f} MB)")


def make_calibration_npy(path, size, samples):
    path = Path(path)
    if path.exists():
        return path
    rng = np.random.default_rng(0)
    data = rng.random((samples, size, size, 3), dtype=np.float32)
    np.save(path, data)
    print(f"[calib] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def compile_edgetpu(tflite_path, out_dir, log_path, compiler):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [compiler, "-s", "-a", "-o", str(out_dir), str(tflite_path)]
    run(cmd, log_path=log_path)


def patch_conv_kernel_shapes(onnx_path):
    import onnx

    src = Path(onnx_path)
    dst = src.with_name(src.stem + "_kernelshape.onnx")
    model = onnx.load(src, load_external_data=True)
    initializers = {item.name: item for item in model.graph.initializer}
    patched = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            continue
        dims = list(initializers[node.input[1]].dims)
        if len(dims) >= 3:
            node.attribute.append(onnx.helper.make_attribute("kernel_shape", dims[2:]))
            patched += 1
    if patched:
        onnx.save_model(model, dst, save_as_external_data=False)
        print(f"[onnx] patched kernel_shape on {patched} Conv nodes -> {dst.name}")
        return str(dst)
    return str(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="repnext_m5_ade20k.onnx")
    ap.add_argument("--saved-model", default="saved_model_repnext")
    ap.add_argument("--tflite", default="repnext_m5_ade20k_int8.tflite")
    ap.add_argument("--edgetpu-log", default="repnext_m5_ade20k_int8_edgetpu.log")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--calib-dir", default="")
    ap.add_argument("--input-layout", choices=["nhwc", "nchw"], default="nhwc")
    ap.add_argument("--skip-onnx2tf", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--onnx2tf-int8", action="store_true", help="Let onnx2tf emit INT8 TFLite directly with -oiqt")
    ap.add_argument("--disable-group-conv", action="store_true", help="Pass -dgc to onnx2tf for SavedModel-compatible grouped conv rewrite")
    ap.add_argument("--compiler", default="edgetpu_compiler")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)

    meta = {
        "model": "RepNeXt-M5 ADE20K",
        "onnx": args.onnx,
        "tflite": args.tflite,
        "size": args.size,
        "samples": args.samples,
        "calib_dir": args.calib_dir or None,
        "input_layout": args.input_layout,
    }
    Path("conversion_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    onnx_for_conversion = patch_conv_kernel_shapes(args.onnx)
    if not args.skip_onnx2tf:
        cmd = ["onnx2tf", "-i", onnx_for_conversion, "-o", args.saved_model, "-dsm", "-n"]
        if args.disable_group_conv:
            cmd.append("-dgc")
        if args.onnx2tf_int8:
            calib = make_calibration_npy(f"calib_{args.size}_{args.samples}_nhwc_float32.npy", args.size, args.samples)
            cmd.extend([
                "-oiqt", "-iqd", "int8", "-oqd", "int8",
                "-cind", "input", str(calib),
                "[[[[0.485,0.456,0.406]]]]",
                "[[[[0.229,0.224,0.225]]]]",
            ])
        run(cmd)
    if not args.onnx2tf_int8:
        convert_saved_model_to_int8(args.saved_model, args.tflite, args.calib_dir, args.size, args.samples, args.input_layout)
    if not args.skip_compile:
        compile_edgetpu(args.tflite, ".", args.edgetpu_log, args.compiler)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
