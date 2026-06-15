#!/usr/bin/env python3
"""GPU-server distillation + QAT training for the Coral TPU RepNeXt student.

This script is intentionally self-contained for a fresh GPU machine:

1. Build a small RepNeXt-style student that is known to compile well for Coral
   Edge TPU (default: w48, input 192).
2. Warm-start it from the original RepNeXt-M5 ADE20K checkpoint by channel
   slicing.
3. Train with ground-truth CE + teacher KL distillation.
4. Enable activation fake-quant in the final epochs so the student becomes more
   robust to full INT8 TFLite quantization.
5. Save a config-bundled checkpoint that the existing export/compile pipeline
   can consume directly.

The script trains logits at stride 4. For input 192, the student output is
48x48x150 logits, matching the compiler-clean Coral artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conversion"))

import export_onnx  # noqa: E402

NUM_CLASSES = 150
IGNORE_INDEX = 255


@dataclass
class StudentConfig:
    embed_dim: list[int]
    depth: list[int]
    fpn_out: int
    head_ch: int
    num_classes: int = NUM_CLASSES


class LogitsModel(nn.Module):
    """RepNeXtSeg without final upsample; returns stride-4 logits."""

    def __init__(self, seg: export_onnx.RepNeXtSeg) -> None:
        super().__init__()
        self.backbone = seg.backbone
        self.neck = seg.neck
        self.decode_head = seg.decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.neck(self.backbone(x)))


class ActFakeQuant(nn.Module):
    """Per-tensor affine INT8 fake quantization with straight-through gradients."""

    def __init__(self, momentum: float = 0.01) -> None:
        super().__init__()
        self.momentum = momentum
        self.enabled = False
        self.register_buffer("min_val", torch.zeros(1))
        self.register_buffer("max_val", torch.zeros(1))
        self.register_buffer("inited", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.training:
            cur_min = x.detach().amin()
            cur_max = x.detach().amax()
            if self.inited.item() == 0:
                self.min_val.copy_(cur_min.reshape(1))
                self.max_val.copy_(cur_max.reshape(1))
                self.inited.fill_(1)
            else:
                self.min_val.mul_(1.0 - self.momentum).add_(self.momentum * cur_min)
                self.max_val.mul_(1.0 - self.momentum).add_(self.momentum * cur_max)

        qmin, qmax = -128.0, 127.0
        min_v = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_v = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        scale = (max_v - min_v).clamp_min(1e-8) / (qmax - qmin)
        zero_point = torch.round(qmin - min_v / scale)
        q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        dq = (q - zero_point) * scale
        return x + (dq - x).detach()


def attach_fake_quant(model: nn.Module) -> nn.ModuleList:
    """Attach fake-quant hooks after BN and activation tensors."""

    quantizers = nn.ModuleList()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.GELU)):
            fq = ActFakeQuant()
            quantizers.append(fq)
            module.register_forward_hook(lambda _m, _i, out, _fq=fq: _fq(out))
    return quantizers


def set_fake_quant(quantizers: nn.ModuleList, enabled: bool) -> None:
    for q in quantizers:
        q.enabled = enabled


def build_student_from_config(config: StudentConfig) -> export_onnx.RepNeXtSeg:
    export_onnx.SPARSE_EQUIV_DOWNSAMPLE = True
    export_onnx.TPU_FRIENDLY_DOWNSAMPLE = False
    return export_onnx.RepNeXtSeg(
        embed_dim=tuple(config.embed_dim),
        depth=tuple(config.depth),
        fpn_out=config.fpn_out,
        head_ch=config.head_ch,
        num_classes=config.num_classes,
        act=lambda: nn.GELU(approximate="tanh"),
    )


def build_teacher(weights: Path, device: torch.device) -> LogitsModel:
    export_onnx.SPARSE_EQUIV_DOWNSAMPLE = False
    export_onnx.TPU_FRIENDLY_DOWNSAMPLE = False
    teacher = export_onnx.RepNeXtSeg(act=nn.GELU)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    teacher = LogitsModel(teacher).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _remap_block_keys(src_sd: dict, depth_new: list[int], depth_old=(7, 7, 35, 2)) -> dict:
    keep = dict(src_sd)
    for stage, d_old in enumerate(depth_old):
        d_new = depth_new[stage]
        for blk in range(d_new, d_old):
            prefix = f"backbone.stages.{stage}.blocks.{blk}."
            for key in [k for k in keep if k.startswith(prefix)]:
                del keep[key]
    return keep


def warm_start_student(model: nn.Module, src_sd: dict) -> tuple[int, int]:
    """Copy pretrained tensors by name, slicing leading channels for the w48 student."""

    src_sd = export_onnx.rewrite_sparse_downsample_weights(src_sd)[0]
    depth_new = []
    keys = list(model.state_dict())
    for stage in range(4):
        blocks = {
            key.split(".blocks.")[1].split(".")[0]
            for key in keys
            if f"backbone.stages.{stage}.blocks." in key
        }
        depth_new.append(len(blocks))
    src_sd = _remap_block_keys(src_sd, depth_new)

    dst_sd = model.state_dict()
    copied = skipped = 0
    for name, dst in dst_sd.items():
        src = src_sd.get(name)
        if src is None or src.dim() != dst.dim():
            skipped += 1
            continue
        slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
        try:
            dst[slices].copy_(src[slices])
            copied += 1
        except RuntimeError:
            skipped += 1
    model.load_state_dict(dst_sd)
    return copied, skipped


def build_or_load_student(args: argparse.Namespace) -> tuple[export_onnx.RepNeXtSeg, dict]:
    if args.student:
        ckpt = torch.load(args.student, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        config = StudentConfig(
            embed_dim=list(cfg["embed_dim"]),
            depth=list(cfg["depth"]),
            fpn_out=int(cfg["fpn_out"]),
            head_ch=int(cfg["head_ch"]),
            num_classes=int(cfg.get("num_classes", NUM_CLASSES)),
        )
        model = build_student_from_config(config)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model, asdict(config)

    base = args.base_width
    if base % 4 != 0:
        raise ValueError("--base-width must be divisible by 4")
    config = StudentConfig(
        embed_dim=[base, base * 2, base * 4, base * 8],
        depth=list(args.depth),
        fpn_out=args.fpn_out,
        head_ch=args.head_ch,
    )
    model = build_student_from_config(config)
    if args.init == "pretrained":
        ckpt = torch.load(args.teacher_weights, map_location="cpu", weights_only=False)
        copied, skipped = warm_start_student(model, ckpt.get("state_dict", ckpt))
        print(f"[student] warm-start copied={copied} skipped={skipped}")
    return model, asdict(config)


class AdeDataset(Dataset):
    def __init__(self, ade_root: Path, split: str, size: int, stems: list[str]) -> None:
        base = ade_root / "ADEChallengeData2016"
        self.img_dir = base / "images" / split
        self.ann_dir = base / "annotations" / split
        self.size = size
        self.grid = size // 4
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()

        mask = Image.open(self.ann_dir / f"{stem}.png")
        mask = mask.resize((self.grid, self.grid), Image.Resampling.NEAREST)
        raw = np.asarray(mask, dtype=np.int64)
        y = raw - 1
        y[raw == 0] = IGNORE_INDEX
        y[(y < 0) | (y >= NUM_CLASSES)] = IGNORE_INDEX
        return x, torch.from_numpy(y), stem


def list_stems(ade_root: Path, split: str, limit: int | None, stride: int, seed: int) -> list[str]:
    img_dir = ade_root / "ADEChallengeData2016" / "images" / split
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if split == "training":
        rng = random.Random(seed)
        rng.shuffle(stems)
    stems = stems[:: max(1, stride)]
    return stems[:limit] if limit else stems


def collate(batch):
    xs, ys, stems = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(stems)


@torch.inference_mode()
def evaluate(model: LogitsModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for x, gt, _stems in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy().astype(np.int64)
        target = gt.numpy().astype(np.int64)
        valid = target != IGNORE_INDEX
        p = pred[valid]
        t = target[valid]
        keep = (p >= 0) & (p < NUM_CLASSES) & (t >= 0) & (t < NUM_CLASSES)
        idx = t[keep] * NUM_CLASSES + p[keep]
        conf += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(conf).astype(np.float64)
    gt_count = conf.sum(1).astype(np.float64)
    pred_count = conf.sum(0).astype(np.float64)
    denom = gt_count + pred_count - tp
    valid_cls = denom > 0
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=valid_cls)
    pix = float(tp.sum() / max(conf.sum(), 1))
    return {
        "mIoU": float(iou[valid_cls].mean()) if valid_cls.any() else 0.0,
        "pixel_acc": pix,
        "classes_present": int(valid_cls.sum()),
    }


def save_checkpoint(path: Path, seg: export_onnx.RepNeXtSeg, config: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": seg.state_dict(), "config": config, **meta}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ade-root", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--student", type=Path, default=None, help="optional config-bundled student checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--teacher-size", type=int, default=384)
    parser.add_argument("--base-width", type=int, default=48)
    parser.add_argument("--depth", type=int, nargs=4, default=[4, 4, 8, 2])
    parser.add_argument("--fpn-out", type=int, default=96)
    parser.add_argument("--head-ch", type=int, default=64)
    parser.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--freeze-bn-after", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[env] torch={torch.__version__} device={device} cuda={torch.cuda.is_available()}")

    seg, config = build_or_load_student(args)
    student = LogitsModel(seg).to(device)
    teacher = build_teacher(args.teacher_weights, device)
    quantizers = attach_fake_quant(seg)

    train_stems = list_stems(args.ade_root, "training", args.train_limit, args.train_stride, args.seed)
    val_stems = list_stems(args.ade_root, "validation", args.val_limit, 1, args.seed)
    train_ds = AdeDataset(args.ade_root, "training", args.size, train_stems)
    val_ds = AdeDataset(args.ade_root, "validation", args.size, val_stems)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    print(f"[student] config={json.dumps(config)} input={args.size} logits={args.size//4}x{args.size//4}")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} batch={args.batch}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = args.distill_epochs + args.qat_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_epochs))
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    best_miou = -math.inf
    best_path = args.out.with_name(args.out.stem + "_best.pth")

    base = evaluate(student, val_loader, device)
    print(f"[eval] epoch=0 mIoU={base['mIoU']:.4f} pixel={base['pixel_acc']:.4f}")

    for epoch in range(total_epochs):
        qat_on = epoch >= args.distill_epochs
        set_fake_quant(quantizers, qat_on)
        student.train()
        teacher.eval()
        if epoch >= args.freeze_bn_after:
            for m in seg.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        t0 = time.perf_counter()
        total_loss = total_ce = total_kd = 0.0
        for step, (x, gt, _stems) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_in = F.interpolate(x, size=(args.teacher_size, args.teacher_size), mode="bilinear", align_corners=False)
                t_logits = teacher(teacher_in)
                t_logits = F.interpolate(t_logits, size=(args.size // 4, args.size // 4), mode="bilinear", align_corners=False)

            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = student(x)
                ce = F.cross_entropy(logits, gt, ignore_index=IGNORE_INDEX)
                temp = args.temperature
                kd = F.kl_div(
                    F.log_softmax(logits / temp, dim=1),
                    F.softmax(t_logits / temp, dim=1),
                    reduction="batchmean",
                ) * (temp * temp)
                loss = ce + args.lambda_kd * kd
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.detach())
            total_ce += float(ce.detach())
            total_kd += float(kd.detach())
            if step % args.log_every == 0:
                ms = (time.perf_counter() - t0) * 1000.0 / step
                print(
                    f"[train] epoch={epoch+1}/{total_epochs} phase={'qat' if qat_on else 'distill'} "
                    f"step={step}/{len(train_loader)} loss={total_loss/step:.4f} "
                    f"ce={total_ce/step:.4f} kd={total_kd/step:.4f} {ms:.0f} ms/step",
                    flush=True,
                )

        sched.step()
        meta = {
            "epoch": epoch + 1,
            "phase": "qat" if qat_on else "distill",
            "train": {
                "loss": total_loss / max(1, len(train_loader)),
                "ce": total_ce / max(1, len(train_loader)),
                "kd": total_kd / max(1, len(train_loader)),
            },
            "args": vars(args),
        }

        if (epoch + 1) % args.eval_every == 0 or epoch == total_epochs - 1:
            ev = evaluate(student, val_loader, device)
            meta["val"] = ev
            elapsed = (time.perf_counter() - t0) / 60.0
            print(
                f"[eval] epoch={epoch+1}/{total_epochs} phase={meta['phase']} "
                f"mIoU={ev['mIoU']:.4f} pixel={ev['pixel_acc']:.4f} elapsed={elapsed:.1f} min",
                flush=True,
            )
            if qat_on and ev["mIoU"] > best_miou:
                best_miou = ev["mIoU"]
                save_checkpoint(best_path, seg, config, meta)
                print(f"[ckpt] best QAT -> {best_path} mIoU={best_miou:.4f}", flush=True)

        save_checkpoint(args.out, seg, config, meta)
        print(f"[ckpt] latest -> {args.out}", flush=True)

    print(f"[done] latest={args.out} best={best_path} best_qat_mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
