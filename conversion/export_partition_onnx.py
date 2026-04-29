"""Export exact RepNeXt-M5 prefix partitions to ONNX."""
import argparse
import os

import torch
import torch.nn as nn

import export_onnx


class RepNeXtPrefix(nn.Module):
    def __init__(self, full_model, stop_at, stage2_blocks):
        super().__init__()
        self.backbone = full_model.backbone
        self.stop_at = stop_at
        self.stage2_blocks = stage2_blocks

    def forward(self, x):
        x = self.backbone.stem(x)
        if self.stop_at == "stem":
            return x
        for idx, stage in enumerate(self.backbone.stages):
            if idx == 2 and self.stop_at in ("stage2_downsample", "stage2_blocks"):
                x = stage.downsample(x)
                if self.stop_at == "stage2_downsample":
                    return x
                for block_idx, block in enumerate(stage.blocks):
                    if block_idx >= self.stage2_blocks:
                        break
                    x = block(x)
                return x
            x = stage(x)
            if self.stop_at == f"stage{idx}":
                return x
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="repnext_m5_ade20k.pth")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--stop-at",
        choices=["stem", "stage0", "stage1", "stage2", "stage2_downsample", "stage2_blocks", "stage3"],
        default="stem",
    )
    parser.add_argument("--stage2-blocks", type=int, default=0)
    parser.add_argument("--activation", choices=["relu", "gelu", "tanh-gelu"], default="tanh-gelu")
    parser.add_argument("--tpu-friendly-downsample", action="store_true")
    parser.add_argument("--sparse-equiv-downsample", action="store_true")
    args = parser.parse_args()

    if args.tpu_friendly_downsample and args.sparse_equiv_downsample:
        raise ValueError("--tpu-friendly-downsample and --sparse-equiv-downsample are mutually exclusive")
    export_onnx.TPU_FRIENDLY_DOWNSAMPLE = args.tpu_friendly_downsample
    export_onnx.SPARSE_EQUIV_DOWNSAMPLE = args.sparse_equiv_downsample
    if args.activation == "relu":
        act_cls = nn.ReLU
    elif args.activation == "tanh-gelu":
        act_cls = lambda: nn.GELU(approximate="tanh")
    else:
        act_cls = nn.GELU

    full = export_onnx.RepNeXtSeg(act=act_cls)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if args.sparse_equiv_downsample:
        state_dict, rewritten_count = export_onnx.rewrite_sparse_downsample_weights(state_dict)
        print(f"[load] sparse downsample weight rewrites={rewritten_count}")
    missing, unexpected = full.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model = RepNeXtPrefix(full, args.stop_at, args.stage2_blocks).eval()
    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        y = model(x)
    print(f"[forward] stop_at={args.stop_at} input={tuple(x.shape)} output={tuple(y.shape)}")

    torch.onnx.export(
        model,
        x,
        args.out,
        input_names=["input"],
        output_names=[f"{args.stop_at}_out"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[done] {args.out} {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
