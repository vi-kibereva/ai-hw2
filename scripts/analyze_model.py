import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models.detection.transform import resize_boxes

import scripts.detection_errors as de
from scripts.eda import finish_figure, set_global_seed
from scripts.fr_cnn import (build_val, cuda_sync, coco_evaluate, collate, dataset_to_coco_dict, get_device,
                       load_config, load_trained, make_coco, set_test_proposals)

MAX_PROPS = 2000


@torch.no_grad()
def collect_records(model, dataset, cfg, device, max_props=MAX_PROPS, batch_size=2):
    model.eval()
    set_test_proposals(model, max_props)
    use_amp = cfg.amp and device.type == "cuda"
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    records = []
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        orig = [tuple(i.shape[-2:]) for i in imgs]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            images, _ = model.transform(imgs)
            feats = model.backbone(images.tensors)
            props, _ = model.rpn(images, feats)
            dets, _ = model.roi_heads(feats, [p[:cfg.rpn_proposals] for p in props], images.image_sizes)
        dets = model.transform.postprocess(dets, images.image_sizes, orig)
        for k, t in enumerate(tgts):
            i = int(t["image_id"])
            H, W = orig[k]
            gt_boxes, gt_labels, ign = dataset.load_annotations(i, W, H)
            p = resize_boxes(props[k].float(), images.image_sizes[k], orig[k]).cpu().numpy()
            records.append({
                "image_id": i, "path": dataset.files[i], "clip": dataset.clip_of(i),
                "gt_boxes": gt_boxes, "gt_labels": gt_labels, "ignore": ign, "props": p,
                "det_boxes": dets[k]["boxes"].float().cpu().numpy(),
                "det_scores": dets[k]["scores"].float().cpu().numpy(),
                "det_labels": dets[k]["labels"].cpu().numpy()})
    set_test_proposals(model, cfg.rpn_proposals)
    return records


def records_to_preds(records):
    return [{"image_id": r["image_id"], "boxes": r["det_boxes"], "scores": r["det_scores"],
             "labels": r["det_labels"]} for r in records]


@torch.no_grad()
def measure_stage_times(model, dataset, cfg, device, n_images=50, warmup=5):
    model.eval()
    use_amp = cfg.amp and device.type == "cuda"
    idx = np.linspace(0, len(dataset) - 1, min(n_images, len(dataset))).astype(int)
    imgs = [dataset[int(i)][0].to(device) for i in idx]
    warmup = min(warmup, len(imgs) // 2)
    rows = []
    for k, img in enumerate(imgs):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            cuda_sync(device)
            t0 = time.perf_counter()
            images, _ = model.transform([img])
            feats = model.backbone(images.tensors)
            cuda_sync(device)
            t1 = time.perf_counter()
            props, _ = model.rpn(images, feats)
            cuda_sync(device)
            t2 = time.perf_counter()
            dets, _ = model.roi_heads(feats, props, images.image_sizes)
            model.transform.postprocess(dets, images.image_sizes, [tuple(img.shape[-2:])])
            cuda_sync(device)
            t3 = time.perf_counter()
        if k >= warmup:
            rows.append((t1 - t0, t2 - t1, t3 - t2))
    m = 1000 * np.mean(rows, axis=0)
    return {"backbone_fpn_ms": m[0], "rpn_ms": m[1], "roi_heads_ms": m[2], "total_ms": m.sum()}


def save_table(df, path, show=True, title=None):
    df.to_csv(path)
    if show:
        print(f"\n--- {title or Path(path).stem} ---")
        print(df.round(3).to_string())


def outcome_report(records, out, n_props, score_thr=0.3, n_failures=6):
    out = Path(out)
    gt_df = de.gt_outcomes(records, n_props, score_thr)
    det_df = de.det_outcomes(records, score_thr)
    gt_df.to_csv(out / "gt_outcomes.csv", index=False)
    det_df.to_csv(out / "det_outcomes.csv", index=False)
    print(f"\n--- GT outcomes at score >= {score_thr}, IoU 0.5 ---")
    print(gt_df["outcome"].value_counts(normalize=True).reindex(de.GT_OUTCOMES, fill_value=0).round(3).to_string())
    print(f"\n--- confident detections (score >= {score_thr}) ---")
    print(det_df["outcome"].value_counts(normalize=True).reindex(de.DET_OUTCOMES, fill_value=0).round(3).to_string())
    gain = de.refinement_gain(gt_df)
    print("\n--- box regression (detected GTs): best candidate IoU -> final box IoU ---")
    print(json.dumps({k: round(v, 3) for k, v in gain.items()}))
    (out / "refinement_gain.json").write_text(json.dumps(gain, indent=2))

    by_class, by_size, by_clip = (de.summarize_outcomes(gt_df, b) for b in ("class", "size", "clip"))
    save_table(by_size, out / "outcomes_by_size.csv", title="GT outcome by object size (px)")
    save_table(by_class, out / "outcomes_by_class.csv", title="GT outcome by class")
    by_clip.to_csv(out / "outcomes_by_clip.csv")
    finish_figure(de.plot_outcome_breakdown(by_size, "GT outcome by object size [px]"), "outcomes_by_size", out, False)
    finish_figure(de.plot_outcome_breakdown(by_class, "GT outcome by class"), "outcomes_by_class", out, False)
    finish_figure(de.plot_outcome_breakdown(by_clip, "GT outcome, 20 hardest clips"), "outcomes_by_clip", out, False)
    finish_figure(de.plot_confusion(de.confusion_matrix(gt_df)), "class_confusion", out, False)

    ids = de.select_failure_frames(gt_df, det_df, n=n_failures)
    finish_figure(de.plot_failure_cases(records, gt_df, det_df, ids), "failure_cases", out, False)
    print(f"\nfailure-case frames: {[records[i]['clip'] + '/' + records[i]['path'].stem for i in ids]}")
    return gt_df, det_df


def analyze(run_dir, score_thr=0.3, n_failures=6, device=None):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    device = device or get_device()
    cfg = load_config(run_dir)
    set_global_seed(cfg.seed)

    val_ds, coco_gt = build_val(cfg)
    coco_agn = make_coco(dataset_to_coco_dict(val_ds, class_agnostic=True))
    model = load_trained(cfg, run_dir / "last.pt", device)
    print(f"collecting stage outputs for {len(val_ds)} validation frames ...")
    records = collect_records(model, val_ds, cfg, device)

    ns = sorted({100, 300, cfg.rpn_proposals, MAX_PROPS})
    recall = de.proposal_recall(records, ns=ns)
    save_table(recall, out / "rpn_recall.csv", title="RPN: share of GT boxes with a proposal IoU>=thr (top-N proposals)")

    preds = records_to_preds(records)
    ap_cls, ap_agn = coco_evaluate(coco_gt, preds), coco_evaluate(coco_agn, preds, class_agnostic=True)
    ap_table = pd.DataFrame({"class-aware": {k: v for k, v in ap_cls.items() if k != "per_class"},
                             "class-agnostic": {k: v for k, v in ap_agn.items() if k != "per_class"}})
    save_table(ap_table, out / "ap_class_aware_vs_agnostic.csv",
               title="AP with and without caring about the class (gap = cost of the classifier)")
    gt_df, det_df = outcome_report(records, out, cfg.rpn_proposals, score_thr, n_failures)

    stage = pd.DataFrame([measure_stage_times(model, val_ds, cfg, device)], index=[f"{cfg.rpn_proposals} proposals"])
    save_table(stage, out / "stage_latency_ms.csv", title="latency per image [ms]")
    print(f"results written to {out}")
    return records, gt_df, det_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing config.json and last.pt")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--n-failures", type=int, default=6)
    a = ap.parse_args()
    analyze(a.run, a.score_thr, a.n_failures)
