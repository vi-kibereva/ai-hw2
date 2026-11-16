import contextlib
import io
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from eda_and_prepare import CLASS_NAMES, N_CLASSES, read_yolo_txt, set_global_seed

IMG_EXT = {".jpg"}
EPOCH_HOOKS = []


@dataclass(frozen=True)
class BaseConfig:
    data_root: str = "prepared"
    out_dir: str = "runs"
    imgsz: int = 1280
    trainable_backbone_layers: int = 3
    box_detections_per_img: int = 500
    box_score_thresh: float = 0.001
    box_nms_thresh: float = 0.5
    init_weights: str = ""
    epochs: int = 12
    batch_size: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_iters: int = 300
    grad_clip: float = 10.0
    amp: bool = True
    train_stride: int = 1
    val_stride: int = 1
    hflip_prob: float = 0.5
    color_jitter: float = 0.2
    crop_prob: float = 0.0
    crop_min_frac: float = 0.5
    eval_every: int = 0
    eval_val_stride: int = 3
    num_workers: int = min(4, os.cpu_count() or 2)
    seed: int = 42


@dataclass(frozen=True)
class Config(BaseConfig):
    arch: str = "v1"
    anchor_sizes: tuple = ((16,), (32,), (64,), (128,), (256,))
    rpn_proposals: int = 1000
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25


def save_config(cfg, run_dir):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def _tuplify(x):
    return tuple(_tuplify(i) for i in x) if isinstance(x, list) else x


def load_config(run_dir, cls=None):
    d = json.loads((Path(run_dir) / "config.json").read_text())
    return (cls or Config)(**{k: _tuplify(v) for k, v in d.items()})


def common_setup_lines(cfg):
    init = [f"- Initialisation: continued from {cfg.init_weights}"] if cfg.init_weights else []
    return init + [
        f"- Input: long side resized to {cfg.imgsz} px (aspect ratio kept); torchvision normalises with ImageNet mean/std",
        f"- Backbone: frozen BatchNorm, last {cfg.trainable_backbone_layers} ResNet stages + FPN + heads trained",
        f"- Inference: score threshold {cfg.box_score_thresh}, NMS IoU {cfg.box_nms_thresh}, "
        f"max {cfg.box_detections_per_img} detections/image",
        f"- Optimiser: SGD momentum {cfg.momentum}, lr {cfg.lr}, weight decay {cfg.weight_decay}, "
        f"{cfg.lr_schedule} schedule, {cfg.warmup_iters} warm-up iterations, grad-clip {cfg.grad_clip}",
        f"- Batch size {cfg.batch_size}, {cfg.epochs} epochs, mixed precision: {cfg.amp}",
        f"- Augmentation: horizontal flip p={cfg.hflip_prob}, colour jitter {cfg.color_jitter} "
        f"(brightness/contrast/saturation, hue/4), random zoom-in crop p={cfg.crop_prob} "
        f"(side fraction {cfg.crop_min_frac}-1); ignored regions are masked in the train images",
        (f"- Model selection: validation mAP every {cfg.eval_every} epoch(s) on every {cfg.eval_val_stride}-th val frame; "
         "last.pt holds the best epoch" if cfg.eval_every else "- Model selection: last epoch"),
        f"- Data: every {cfg.train_stride}-th frame of the prepared train split and every {cfg.val_stride}-th of val "
        "(the prepared dataset is already sub-sampled in time)",
        f"- Reproducibility: seed {cfg.seed} (python/numpy/torch, DataLoader generator and worker seeds); "
        "CUDA ops are not bit-deterministic, so repeated runs differ slightly",
    ]


def describe_setup(cfg):
    weights = "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1" if cfg.arch == "v2" else "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
    return "\n".join([
        "# Training setup: Faster R-CNN",
        f"- Model: torchvision Faster R-CNN, ResNet-50-FPN ({cfg.arch}), initialised from COCO weights `{weights}`;"
        f" box predictor replaced by a new {N_CLASSES + 1}-way head (8 classes + background)",
        f"- Anchors: sizes per FPN level {cfg.anchor_sizes}, aspect ratios (0.5, 1, 2)",
        f"- RPN: {cfg.rpn_proposals} proposals after NMS (train and test); RoI head samples "
        f"{cfg.box_batch_size_per_image} RoIs/image, positive fraction {cfg.box_positive_fraction}",
        *common_setup_lines(cfg),
    ]) + "\n"


class VisDroneDet(torch.utils.data.Dataset):
    def __init__(self, root, split, train=False, stride=1, hflip_prob=0.5, color_jitter=0.2,
                 crop_prob=0.0, crop_min_frac=0.5):
        self.root, self.split, self.train, self.hflip_prob = Path(root), split, train, hflip_prob
        self.crop_prob, self.crop_min_frac = crop_prob, crop_min_frac
        files = sorted(p for p in (self.root / "images" / split).glob("*/*.*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no images in {self.root / 'images' / split}")
        self.files = files[::stride]
        self.jitter = (T.ColorJitter(color_jitter, color_jitter, color_jitter, color_jitter / 4)
                       if train and color_jitter > 0 else None)

    def __len__(self):
        return len(self.files)

    def clip_of(self, i):
        return self.files[i].parent.name

    def _txt(self, i, kind):
        p = self.files[i]
        return self.root / kind / self.split / p.parent.name / f"{p.stem}.txt"

    def load_annotations(self, i, W, H):
        lab = read_yolo_txt(self._txt(i, "labels"), W, H)
        boxes = np.stack([lab[:, 1], lab[:, 2], lab[:, 1] + lab[:, 3], lab[:, 2] + lab[:, 4]], 1)
        keep = (lab[:, 3] >= 1) & (lab[:, 4] >= 1)
        ign = read_yolo_txt(self._txt(i, "ignore"), W, H, with_class=False)
        ign = np.stack([ign[:, 0], ign[:, 1], ign[:, 0] + ign[:, 2], ign[:, 1] + ign[:, 3]], 1)
        return (boxes[keep].astype(np.float32), (lab[keep, 0] + 1).astype(np.int64), ign.astype(np.float32))

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        W, H = img.size
        boxes, labels, _ = self.load_annotations(i, W, H)
        if self.train and self.crop_prob > 0 and torch.rand(1).item() < self.crop_prob:
            f = self.crop_min_frac + (1 - self.crop_min_frac) * torch.rand(1).item()
            cw, ch = max(int(f * W), 2), max(int(f * H), 2)
            x0, y0 = int(torch.rand(1).item() * (W - cw)), int(torch.rand(1).item() * (H - ch))
            window = (x0, y0, x0 + cw, y0 + ch)
            img = img.crop(window)
            boxes, labels = crop_with_boxes(boxes, labels, window)
            W, H = img.size
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.jitter is not None:
            img = self.jitter(img)
        img = TF.to_tensor(img)
        if self.train and torch.rand(1).item() < self.hflip_prob:
            img = img.flip(-1)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return img, {"boxes": boxes, "labels": torch.as_tensor(labels, dtype=torch.int64),
                     "image_id": torch.tensor(i)}


def crop_with_boxes(boxes, labels, window, min_visible=0.4, min_px=1.0):
    boxes, labels = np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels)
    x0, y0, x1, y1 = window
    clipped = np.stack([np.clip(boxes[:, 0], x0, x1), np.clip(boxes[:, 1], y0, y1),
                        np.clip(boxes[:, 2], x0, x1), np.clip(boxes[:, 3], y0, y1)], 1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    cw, chh = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    keep = (cw * chh >= min_visible * np.maximum(area, 1e-6)) & (cw >= min_px) & (chh >= min_px)
    shifted = clipped[keep] - np.array([x0, y0, x0, y0], dtype=np.float32)
    return shifted.astype(np.float32), labels[keep]


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg, pretrained=True):
    n = cfg.rpn_proposals
    kwargs = dict(
        min_size=cfg.imgsz, max_size=cfg.imgsz,
        rpn_pre_nms_top_n_train=n, rpn_post_nms_top_n_train=n,
        rpn_pre_nms_top_n_test=n, rpn_post_nms_top_n_test=n,
        box_batch_size_per_image=cfg.box_batch_size_per_image,
        box_positive_fraction=cfg.box_positive_fraction,
        box_detections_per_img=cfg.box_detections_per_img,
        box_score_thresh=cfg.box_score_thresh, box_nms_thresh=cfg.box_nms_thresh,
        weights_backbone=None)
    if pretrained:
        kwargs["trainable_backbone_layers"] = cfg.trainable_backbone_layers
    if cfg.arch == "v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None, **kwargs)
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None, **kwargs)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=cfg.anchor_sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(cfg.anchor_sizes))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, N_CLASSES + 1)
    return model


def set_test_proposals(model, n):
    model.rpn._pre_nms_top_n["testing"] = n
    model.rpn._post_nms_top_n["testing"] = n


def set_test_size(model, long_side):
    model.transform.min_size = (long_side,)
    model.transform.max_size = long_side


def load_trained(cfg, ckpt_path, device, build_fn=None):
    model = (build_fn or build_model)(cfg, pretrained=False)
    missing, unexpected = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    bad = [k for k in missing if not k.endswith("num_batches_tracked")] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint does not match the model: {bad[:5]}")
    return model.to(device).eval()


def lr_factor(it, total, warmup, kind):
    if it < warmup:
        return 0.001 + 0.999 * it / max(1, warmup)
    prog = (it - warmup) / max(1, total - warmup)
    if kind == "step":
        return 1.0 if prog < 0.67 else (0.1 if prog < 0.89 else 0.01)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch, log, log_every=50):
    model.train()
    use_amp = cfg.amp and device.type == "cuda"
    epoch_sum, window, n_epoch = defaultdict(float), defaultdict(float), 0
    n_win = 0
    for it, (imgs, tgts) in enumerate(loader):
        imgs = [i.to(device, non_blocking=True) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            losses = model(imgs, tgts)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            print(f"  non-finite loss at epoch {epoch} it {it}, skipping the batch")
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        vals = {k: v.item() for k, v in losses.items()}
        vals["loss"] = loss.item()
        for k, v in vals.items():
            epoch_sum[k] += v
            window[k] += v
        n_epoch += 1
        n_win += 1
        if n_win == log_every:
            row = {"iter": epoch * len(loader) + it + 1, "epoch": epoch,
                   "lr": optimizer.param_groups[0]["lr"], **{k: v / n_win for k, v in window.items()}}
            log.append(row)
            print(f"  ep {epoch} it {it + 1}/{len(loader)} lr {row['lr']:.5f} loss {row['loss']:.3f}")
            window, n_win = defaultdict(float), 0
    return {k: v / max(n_epoch, 1) for k, v in epoch_sum.items()}


def save_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def fit(cfg, name, device, resume=True, build_fn=None, describe_fn=None):
    run_dir = Path(cfg.out_dir) / name
    save_config(cfg, run_dir)
    (run_dir / "SETUP.md").write_text((describe_fn or describe_setup)(cfg))
    set_global_seed(cfg.seed)

    ds = VisDroneDet(cfg.data_root, "train", train=True, stride=cfg.train_stride,
                     hflip_prob=cfg.hflip_prob, color_jitter=cfg.color_jitter,
                     crop_prob=cfg.crop_prob, crop_min_frac=cfg.crop_min_frac)
    model = (build_fn or build_model)(cfg)
    if cfg.init_weights:
        missing, unexpected = model.load_state_dict(torch.load(cfg.init_weights, map_location="cpu"), strict=False)
        bad = [k for k in missing if not k.endswith("num_batches_tracked")] + list(unexpected)
        if bad:
            raise RuntimeError(f"init_weights do not match the model: {bad[:5]}")
        print(f"[{name}] starting from {cfg.init_weights}")
    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    n_iter = len(ds) // cfg.batch_size
    total = cfg.epochs * n_iter
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: lr_factor(it, total, cfg.warmup_iters, cfg.lr_schedule))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    start_epoch, log, minutes, best_map, eval_log = 0, [], 0.0, -1.0, []
    ckpt_path = run_dir / "ckpt.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, log, minutes = ck["epoch"], ck["log"], ck["minutes"]
        best_map, eval_log = ck.get("best_map", -1.0), ck.get("eval_log", [])
        print(f"[{name}] resuming after epoch {start_epoch}/{cfg.epochs}")

    gen = torch.Generator()
    gen.manual_seed(cfg.seed + start_epoch)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate,
        drop_last=True, generator=gen, worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)

    val_ds = coco_gt = None
    if cfg.eval_every:
        val_ds, coco_gt = build_val(replace(cfg, val_stride=cfg.eval_val_stride))

    print(f"[{name}] {len(ds)} train frames, {len(loader)} it/epoch, {total} iterations in total")
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        stats = train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch, log)
        print(f"[{name}] epoch {epoch + 1}/{cfg.epochs}: " + ", ".join(f"{k} {v:.3f}" for k, v in stats.items()))
        if cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
            m = evaluate_only(model, cfg, device, val_ds, coco_gt)
            eval_log.append({"epoch": epoch + 1, "mAP": m["mAP"], "AP50": m["AP50"], "AP_small": m["AP_small"]})
            if m["mAP"] > best_map:
                best_map = m["mAP"]
                save_checkpoint(run_dir / "best.pt", model.state_dict())
            print(f"[{name}] epoch {epoch + 1} val mAP {m['mAP']:.4f}  AP50 {m['AP50']:.4f}  "
                  f"AP_small {m['AP_small']:.4f}  (best {best_map:.4f})")
            pd.DataFrame(eval_log).to_csv(run_dir / "eval_log.csv", index=False)
        save_checkpoint(ckpt_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                                    "epoch": epoch + 1, "log": log, "best_map": best_map, "eval_log": eval_log,
                                    "minutes": minutes + (time.time() - t0) / 60})
        pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
        for hook in EPOCH_HOOKS:
            hook(run_dir, epoch + 1, cfg)
    minutes += (time.time() - t0) / 60
    if cfg.eval_every and (run_dir / "best.pt").exists():
        save_checkpoint(run_dir / "final_epoch.pt", model.state_dict())
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
        print(f"[{name}] using the best epoch (val mAP {best_map:.4f}) as last.pt")
    save_checkpoint(run_dir / "last.pt", model.state_dict())
    return model, minutes


def dataset_to_coco_dict(dataset, class_agnostic=False):
    cat_ids = [1] if class_agnostic else list(range(1, N_CLASSES + 1))
    cats = [{"id": c, "name": "object" if class_agnostic else CLASS_NAMES[c - 1]} for c in cat_ids]
    images, anns, aid = [], [], 1
    for i in range(len(dataset)):
        W, H = Image.open(dataset.files[i]).size
        boxes, labels, ign = dataset.load_annotations(i, W, H)
        images.append({"id": i, "file_name": str(dataset.files[i]), "width": W, "height": H})
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            w, h = float(x2 - x1), float(y2 - y1)
            anns.append({"id": aid, "image_id": i, "category_id": 1 if class_agnostic else int(lab),
                         "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 0})
            aid += 1
        for x1, y1, x2, y2 in ign:
            w, h = float(x2 - x1), float(y2 - y1)
            for c in cat_ids:
                anns.append({"id": aid, "image_id": i, "category_id": c,
                             "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 1})
                aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def make_coco(d):
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def build_val(cfg):
    ds = VisDroneDet(cfg.data_root, "val", train=False, stride=cfg.val_stride)
    return ds, make_coco(dataset_to_coco_dict(ds))


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    out = []
    for imgs, tgts in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            preds = model([i.to(device) for i in imgs])
        for t, p in zip(tgts, preds):
            keep = (p["labels"] > 0).cpu().numpy()          # RetinaNet has a background logit column
            out.append({"image_id": int(t["image_id"]), "boxes": p["boxes"].float().cpu().numpy()[keep],
                        "scores": p["scores"].float().cpu().numpy()[keep], "labels": p["labels"].cpu().numpy()[keep]})
    return out


def to_coco_results(preds, class_agnostic=False):
    res = []
    for p in preds:
        for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):
            res.append({"image_id": p["image_id"], "category_id": 1 if class_agnostic else int(l),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s)})
    return res


def coco_evaluate(coco_gt, preds, class_agnostic=False, img_ids=None):
    from pycocotools.cocoeval import COCOeval
    keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500", "AR_small"]
    res = to_coco_results(preds, class_agnostic)
    if not res:
        return {**{k: 0.0 for k in keys}, "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(coco_gt, coco_gt.loadRes(res), "bbox")
        E.params.maxDets = [1, 10, 500]
        if img_ids is not None:
            E.params.imgIds = sorted(img_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats
    prec_all = E.eval["precision"][:, :, :, 0, -1]
    prec_all = prec_all[prec_all > -1]
    out = {"mAP": float(prec_all.mean()) if prec_all.size else -1.0, "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
           "AR500": s[8], "AR_small": s[9], "per_class": {}}
    if not class_agnostic:
        prec = E.eval["precision"][:, :, :, 0, -1]               # iou x recall x class (all areas, 500 dets)
        for k, cid in enumerate(E.params.catIds):
            p = prec[:, :, k]
            p = p[p > -1]
            out["per_class"][CLASS_NAMES[cid - 1]] = float(p.mean()) if p.size else float("nan")
    return out


def evaluate_only(model, cfg, device, val_ds, coco_gt):
    loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    return coco_evaluate(coco_gt, predict(model, loader, device, cfg.amp))


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_speed(model, dataset, device, amp=True, n_images=100, warmup=10):
    model.eval()
    idx = np.linspace(0, len(dataset) - 1, min(n_images, len(dataset))).astype(int)
    imgs = [dataset[int(i)][0].to(device) for i in idx]
    warmup = min(warmup, len(imgs) // 2)
    times = []
    for k, img in enumerate(imgs):
        cuda_sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            model([img])
        cuda_sync(device)
        if k >= warmup:
            times.append(time.perf_counter() - t0)
    ms = 1000 * float(np.mean(times))
    return {"ms_per_img": ms, "ms_median": 1000 * float(np.median(times)), "fps": 1000 / ms,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}


def full_evaluation(model, cfg, device, val_ds, coco_gt, n_speed_images=100):
    loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    preds = predict(model, loader, device, cfg.amp)
    metrics = coco_evaluate(coco_gt, preds)
    per_class = metrics.pop("per_class")
    speed = measure_speed(model, val_ds, device, cfg.amp, n_images=n_speed_images)
    return {**metrics, **{f"AP_{k}": v for k, v in per_class.items()}, **speed}, preds
