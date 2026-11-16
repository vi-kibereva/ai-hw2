import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import anchor_analysis as AA
from frcnn_lib import (VisDroneDet, build_val, coco_evaluate, collate, dataset_to_coco_dict, full_evaluation,
                       get_device, make_coco, predict, set_test_size)
from run_experiments import _load_run

METRICS = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


def run_history(run_dirs):
    rows = []
    for d in map(Path, run_dirs):
        if not (d / "metrics.json").exists():
            print(f"skipping {d.name}: no metrics.json")
            continue
        m, cfg = json.loads((d / "metrics.json").read_text()), json.loads((d / "config.json").read_text())
        rows.append({"run": d.name, "imgsz": cfg.get("imgsz"), "epochs": cfg.get("epochs"), "lr": cfg.get("lr"),
                     **{k: m.get(k) for k in METRICS}})
    df = pd.DataFrame(rows).set_index("run")
    df["d_mAP"] = df["mAP"].diff()
    return df


def per_class_ap(run_dirs):
    rows = {}
    for d in map(Path, run_dirs):
        if (d / "metrics.json").exists():
            m = json.loads((d / "metrics.json").read_text())
            rows[d.name] = {k[3:]: v for k, v in m.items() if k.startswith("AP_") and k not in ("AP_small", "AP_medium", "AP_large")}
    return pd.DataFrame(rows).T


def train_vs_val(run_dir, kind="faster", train_stride=8, device=None):
    device = device or get_device()
    model, cfg = _load_run(run_dir, kind, device)
    val_ds, val_gt = build_val(cfg)
    tr_ds = VisDroneDet(cfg.data_root, "train", train=False, stride=train_stride)
    tr_gt = make_coco(dataset_to_coco_dict(tr_ds))
    out = {}
    for name, ds, gt in (("train subset", tr_ds, tr_gt), ("validation", val_ds, val_gt)):
        m, _ = full_evaluation(model, cfg, device, ds, gt, n_speed_images=10)
        out[name] = {k: m[k] for k in METRICS}
    return pd.DataFrame(out)


def faster_anchor_coverage(cfg, n_images=200, thrs=(0.3, 0.5, 0.7)):
    ds = VisDroneDet(cfg.data_root, "train", train=False, stride=cfg.train_stride)
    samples = AA.sample_scaled_gt(ds, cfg.imgsz, n_images=n_images, seed=cfg.seed)
    res = AA.evaluate_anchor_config(samples, cfg.anchor_sizes, (0.5, 1.0, 2.0), iou_pairs=((0.7, 0.3),),
                                    strides=(4, 8, 16, 32, 64))
    return AA.coverage_summary(res["gt_best"], res["gt_size"], thrs), AA.assignment_summary(res)


def per_clip_by_size(run_dir, sizes=(1536, 1920), kind=None, device=None):
    device = device or get_device()
    model, cfg = _load_run(run_dir, kind, device)
    val_ds, coco_gt = build_val(cfg)
    clips = np.array([val_ds.clip_of(i) for i in range(len(val_ds))])
    info = {}
    for clip in sorted(set(clips)):
        ids = [int(i) for i in np.where(clips == clip)[0]]
        ann = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=ids, iscrowd=False))
        info[clip] = {"ids": ids, "size": "x".join(map(str, Image.open(val_ds.files[ids[0]]).size)),
                      "frames": len(ids), "n_gt": len(ann),
                      "median_gt_px": float(np.median([np.sqrt(a["area"]) for a in ann])) if ann else np.nan}
    loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    rows = {c: {k: v for k, v in d.items() if k != "ids"} for c, d in info.items()}
    for size in sizes:
        set_test_size(model, size)
        preds = predict(model, loader, device, cfg.amp)
        for clip, d in info.items():
            keep = set(d["ids"])
            m = coco_evaluate(coco_gt, [p for p in preds if p["image_id"] in keep], img_ids=d["ids"])
            rows[clip][f"mAP_{size}"], rows[clip][f"AP50_{size}"] = m["mAP"], m["AP50"]
        print(f"size {size} done")
    df = pd.DataFrame(rows).T
    df["delta_mAP"] = df[f"mAP_{sizes[-1]}"].astype(float) - df[f"mAP_{sizes[0]}"].astype(float)
    print(f"clips improved by {sizes[-1]} vs {sizes[0]}: {(df['delta_mAP'] > 0).sum()} of {len(df)}; "
          f"mean delta {df['delta_mAP'].mean():+.4f}, min {df['delta_mAP'].min():+.4f}, max {df['delta_mAP'].max():+.4f}")
    return df


def sweep_nms(run_dir, values=(0.5, 0.6, 0.7), kind="faster", device=None):
    device = device or get_device()
    model, cfg = _load_run(run_dir, kind, device)
    val_ds, val_gt = build_val(cfg)
    rows = []
    for v in values:
        if kind == "retina":
            model.nms_thresh = v
        else:
            model.roi_heads.nms_thresh = v
        m, _ = full_evaluation(model, cfg, device, val_ds, val_gt, n_speed_images=20)
        rows.append({"nms_thresh": v, **{k: m[k] for k in ("mAP", "AP50", "AP75", "AP_small", "AR500", "ms_per_img")}})
        print(f"nms {v}: mAP {m['mAP']:.4f}  AP_small {m['AP_small']:.4f}")
    return pd.DataFrame(rows)
