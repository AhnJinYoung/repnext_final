"""Export full RepNeXt tile graph without the final image-size upsample."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

import export_onnx


class RepNeXtTileLogits(nn.Module):
    def __init__(self, full_model: export_onnx.RepNeXtSeg):
        super().__init__()
        self.backbone = full_model.backbone
        self.neck = full_model.neck
        self.decode_head = full_model.decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.decode_head(feats)


def activation_class(name: str):
    if name == "relu":
        return nn.ReLU
    if name == "tanh-gelu":
        return lambda: nn.GELU(approximate="tanh")
    return nn.GELU


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="repnext_m5_ade20k.pth")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--activation", choices=["gelu", "relu", "tanh-gelu"], default="relu")
    parser.add_argument("--sparse-equiv-downsample", action="store_true")
    args = parser.parse_args()

    export_onnx.SPARSE_EQUIV_DOWNSAMPLE = args.sparse_equiv_downsample
    export_onnx.TPU_FRIENDLY_DOWNSAMPLE = False

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # A pruned/shrunk checkpoint bundles its architecture config; rebuild that exact net.
    config = ckpt.get("config") if isinstance(ckpt, dict) else None
    if config is not None:
        print(f"[config] {config}")
        full = export_onnx.RepNeXtSeg(
            embed_dim=tuple(config["embed_dim"]),
            depth=tuple(config["depth"]),
            fpn_out=config["fpn_out"],
            head_ch=config["head_ch"],
            num_classes=config["num_classes"],
            act=activation_class(args.activation),
        )
    else:
        full = export_onnx.RepNeXtSeg(act=activation_class(args.activation))
    if args.sparse_equiv_downsample:
        state_dict, rewritten_count = export_onnx.rewrite_sparse_downsample_weights(state_dict)
        print(f"[load] sparse downsample weight rewrites={rewritten_count}")
    missing, unexpected = full.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model = RepNeXtTileLogits(full).eval()
    x = torch.randn(1, 3, args.size, args.size)
    with torch.inference_mode():
        y = model(x)
    print(f"[forward] input={tuple(x.shape)} logits={tuple(y.shape)}")

    torch.onnx.export(
        model,
        x,
        args.out,
        input_names=["input"],
        output_names=["tile_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[done] {args.out} {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
