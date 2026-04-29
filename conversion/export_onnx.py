"""
Export RepNeXt-M5 ADE20K (backbone + FPN + SemanticFPNHead) to ONNX for
EdgeTPU op-coverage analysis. Self-contained: strips mmseg/mmcv dependencies.

Usage:
    python export_onnx.py --ckpt repnext_m5_ade20k.pth --out repnext_m5_ade20k.onnx
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CONVERSION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_MODEL_PATH = os.path.join(CONVERSION_DIR, "RepNeXt", "model")
if not os.path.isdir(REPO_MODEL_PATH):
    REPO_MODEL_PATH = os.path.abspath(
        os.path.join(CONVERSION_DIR, "..", "..", "RepNeXt-tpu", "RepNeXt", "model")
    )
sys.path.insert(0, REPO_MODEL_PATH)

TPU_FRIENDLY_DOWNSAMPLE = False
SPARSE_EQUIV_DOWNSAMPLE = False


class ConvNorm(nn.Sequential):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, d=1, g=1, bias=False):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_c, out_c, k, s, p, d, g, bias=bias))
        self.add_module("norm", nn.BatchNorm2d(out_c))


def mlp(in_c, hid_c, act=nn.GELU):
    return nn.Sequential(ConvNorm(in_c, hid_c, k=1), act(), ConvNorm(hid_c, in_c, k=1))


class RepDWConvS(nn.Module):
    def __init__(self, in_c, stride=1, bias=True):
        super().__init__()
        self.stride = stride
        kw = dict(in_channels=in_c, out_channels=in_c, groups=in_c)
        self.conv_3_3 = nn.Conv2d(bias=bias, kernel_size=3, stride=stride, dilation=1, padding=1, **kw)
        self.conv_3_w = nn.Conv2d(bias=bias and stride == 1, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), **kw)
        self.conv_3_h = nn.Conv2d(bias=bias and stride == 1, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), **kw)
        if SPARSE_EQUIV_DOWNSAMPLE and stride == 2:
            self.conv_2_2 = nn.Conv2d(bias=bias, kernel_size=3, stride=stride, dilation=1, padding=1, **kw)
        elif TPU_FRIENDLY_DOWNSAMPLE and stride == 2:
            self.conv_2_2 = nn.Conv2d(bias=bias, kernel_size=2, stride=stride, dilation=1, padding=0, **kw)
        else:
            self.conv_2_2 = nn.Conv2d(bias=bias, kernel_size=2, stride=stride, dilation=2, padding=1, **kw)

    def forward(self, x):
        if self.stride == 1:
            return self.conv_3_3(x) + self.conv_3_h(x) + self.conv_3_w(x) + self.conv_2_2(x)
        return self.conv_3_3(x) + self.conv_3_h(self.conv_3_w(x)) + self.conv_2_2(x)


class RepDWConvM(nn.Module):
    def __init__(self, in_c, stride=1, bias=True):
        super().__init__()
        kw = dict(in_channels=in_c, out_channels=in_c, groups=in_c)
        self.conv_7_7 = nn.Conv2d(bias=bias, kernel_size=(7, 7), stride=stride, padding=3, **kw)
        self.conv_5_3 = nn.Conv2d(bias=bias, kernel_size=(5, 3), stride=stride, padding=(2, 1), **kw)
        self.conv_3_5 = nn.Conv2d(bias=bias, kernel_size=(3, 5), stride=stride, padding=(1, 2), **kw)
        self.conv_7_w = nn.Conv2d(bias=False, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), **kw)
        self.conv_7_h = nn.Conv2d(bias=False, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), **kw)
        self.conv_5_w = nn.Conv2d(bias=False, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2), **kw)
        self.conv_5_h = nn.Conv2d(bias=False, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0), **kw)

    def forward(self, x):
        return (self.conv_7_7(x) + self.conv_5_3(x) + self.conv_3_5(x)
                + self.conv_7_h(self.conv_7_w(x)) + self.conv_5_h(self.conv_5_w(x)))


class ChunkConv(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        h = in_c // 4
        self.conv_s = RepDWConvS(h)
        self.conv_m = RepDWConvM(h)
        self.conv_l = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=(1, 11), padding=(0, 5), groups=h),
            nn.Conv2d(h, h, kernel_size=(11, 1), padding=(5, 0), groups=h),
        )

    def forward(self, x):
        i, s, m, l = torch.chunk(x, 4, dim=1)
        return torch.cat((i, self.conv_s(s), self.conv_m(m), self.conv_l(l)), dim=1)


class CopyConv(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv_s = RepDWConvS(in_c, stride=2)
        self.conv_m = RepDWConvM(in_c, stride=2)

    def forward(self, x):
        return torch.cat((self.conv_s(x), self.conv_m(x)), dim=1)


class RepNextStem(nn.Module):
    def __init__(self, in_c, out_c, act=nn.GELU):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNorm(in_c, out_c // 2, k=3, s=2, p=1),
            act(),
            ConvNorm(out_c // 2, out_c, k=3, s=2, p=1),
        )

    def forward(self, x):
        return self.stem(x)


class MetaNeXtBlock(nn.Module):
    def __init__(self, in_c, mlp_ratio, act=nn.GELU):
        super().__init__()
        self.token_mixer = ChunkConv(in_c)
        self.norm = nn.BatchNorm2d(in_c)
        self.channel_mixer = mlp(in_c, in_c * mlp_ratio, act=act)

    def forward(self, x):
        return x + self.channel_mixer(self.norm(self.token_mixer(x)))


class Downsample(nn.Module):
    def __init__(self, in_c, mlp_ratio, act=nn.GELU):
        super().__init__()
        out_c = in_c * 2
        self.token_mixer = CopyConv(in_c)
        self.norm = nn.BatchNorm2d(out_c)
        self.channel_mixer = mlp(out_c, out_c * mlp_ratio, act=act)

    def forward(self, x):
        x = self.norm(self.token_mixer(x))
        return x + self.channel_mixer(x)


class RepNextStage(nn.Module):
    def __init__(self, in_c, out_c, depth, mlp_ratio, act=nn.GELU, downsample=True):
        super().__init__()
        self.downsample = Downsample(in_c, mlp_ratio, act=act) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[MetaNeXtBlock(out_c, mlp_ratio, act=act) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class RepNext(nn.Module):
    def __init__(self, embed_dim=(80, 160, 320, 640), depth=(7, 7, 35, 2),
                 mlp_ratio=2, in_chans=3, act=nn.GELU):
        super().__init__()
        self.stem = RepNextStem(in_chans, embed_dim[0], act=act)
        stages = []
        in_c = embed_dim[0]
        for i, (dim, d) in enumerate(zip(embed_dim, depth)):
            stages.append(RepNextStage(in_c, dim, d, mlp_ratio, act=act, downsample=(i != 0)))
            in_c = dim
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for f in self.stages:
            x = f(x)
            outs.append(x)
        return outs


class FPN(nn.Module):
    """mmseg-style FPN: 1x1 lateral + 3x3 output conv + nearest upsample + add."""
    def __init__(self, in_channels=(80, 160, 320, 640), out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential() for _ in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential() for _ in in_channels
        ])
        for i, c in enumerate(in_channels):
            self.lateral_convs[i].add_module("conv", nn.Conv2d(c, out_channels, 1))
            self.fpn_convs[i].add_module("conv", nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, feats):
        laterals = [lc(f) for lc, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
        outs = [fc(lat) for fc, lat in zip(self.fpn_convs, laterals)]
        return outs


class SemanticFPNHead(nn.Module):
    """mmseg FPNHead: scale_heads (Conv+BN+ReLU + Upsample pairs) + summation + conv_seg."""
    def __init__(self, feature_strides=(4, 8, 16, 32), in_channels=256, channels=128, num_classes=150):
        super().__init__()
        self.scale_heads = nn.ModuleList()
        self.feature_strides = feature_strides
        for i, stride in enumerate(feature_strides):
            n_ups = max(1, int((stride // feature_strides[0]).bit_length() - 1)) if stride > feature_strides[0] else 0
            layers = []
            # first conv
            layers.append(self._make_cbr(in_channels, channels))
            if stride > feature_strides[0]:
                layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                cur = stride // 2
                while cur > feature_strides[0]:
                    layers.append(self._make_cbr(channels, channels))
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                    cur //= 2
            self.scale_heads.append(nn.Sequential(*layers))
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def _make_cbr(self, in_c, out_c):
        m = nn.Sequential()
        m.add_module("conv", nn.Conv2d(in_c, out_c, 3, padding=1, bias=False))
        m.add_module("bn", nn.BatchNorm2d(out_c))
        m.add_module("act", nn.ReLU(inplace=True))
        return m

    def forward(self, feats):
        out = self.scale_heads[0](feats[0])
        for i in range(1, len(feats)):
            out = out + self.scale_heads[i](feats[i])
        return self.conv_seg(out)


class RepNeXtSeg(nn.Module):
    def __init__(self, embed_dim=(80, 160, 320, 640), depth=(7, 7, 35, 2),
                 fpn_out=256, head_ch=128, num_classes=150, act=nn.GELU):
        super().__init__()
        self.backbone = RepNext(embed_dim=embed_dim, depth=depth, act=act)
        self.neck = FPN(in_channels=embed_dim, out_channels=fpn_out)
        self.decode_head = SemanticFPNHead(
            feature_strides=(4, 8, 16, 32), in_channels=fpn_out,
            channels=head_ch, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        logits = self.decode_head(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def rewrite_sparse_downsample_weights(state_dict):
    rewritten = dict(state_dict)
    rewritten_count = 0
    for key, value in list(state_dict.items()):
        if not key.endswith("downsample.token_mixer.conv_s.conv_2_2.weight"):
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape[-2:]) != (2, 2):
            continue
        sparse = torch.zeros(
            (value.shape[0], value.shape[1], 3, 3),
            dtype=value.dtype,
            device=value.device,
        )
        sparse[:, :, 0, 0] = value[:, :, 0, 0]
        sparse[:, :, 0, 2] = value[:, :, 0, 1]
        sparse[:, :, 2, 0] = value[:, :, 1, 0]
        sparse[:, :, 2, 2] = value[:, :, 1, 1]
        rewritten[key] = sparse
        rewritten_count += 1
    return rewritten, rewritten_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="repnext_m5_ade20k.pth")
    ap.add_argument("--out", default="repnext_m5_ade20k.onnx")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--tpu-friendly-downsample", action="store_true",
                    help="Remove stride=2+dilation=2 depthwise convs that TensorFlow/TFLite cannot lower.")
    ap.add_argument("--sparse-equiv-downsample", action="store_true",
                    help="Rewrite the 3 problematic stride=2,dilation=2 depthwise branches as equivalent sparse 3x3 depthwise convs.")
    ap.add_argument("--activation", choices=["gelu", "relu", "tanh-gelu"], default="gelu")
    args = ap.parse_args()

    global TPU_FRIENDLY_DOWNSAMPLE, SPARSE_EQUIV_DOWNSAMPLE
    if args.tpu_friendly_downsample and args.sparse_equiv_downsample:
        raise ValueError("--tpu-friendly-downsample and --sparse-equiv-downsample are mutually exclusive")
    TPU_FRIENDLY_DOWNSAMPLE = args.tpu_friendly_downsample
    SPARSE_EQUIV_DOWNSAMPLE = args.sparse_equiv_downsample

    if args.activation == "relu":
        act_cls = nn.ReLU
    elif args.activation == "tanh-gelu":
        act_cls = lambda: nn.GELU(approximate="tanh")
    else:
        act_cls = nn.GELU

    model = RepNeXtSeg(act=act_cls)
    model.eval()

    print(f"[load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if args.sparse_equiv_downsample:
        sd, rewritten_count = rewrite_sparse_downsample_weights(sd)
        print(f"[load] sparse downsample weight rewrites={rewritten_count}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  first 5 missing:", missing[:5])
    if unexpected:
        print("  first 5 unexpected:", unexpected[:5])

    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        y = model(x)
    print(f"[forward] input={tuple(x.shape)} output={tuple(y.shape)}")

    print(f"[onnx] exporting → {args.out} (opset={args.opset})")
    torch.onnx.export(
        model, x, args.out,
        input_names=["input"], output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[done] {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
