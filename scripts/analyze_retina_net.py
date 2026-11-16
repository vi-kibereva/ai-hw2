import argparse
from pathlib import Path

import numpy as np
import torch

import scripts.anchor_analysis as AA
import scripts.detection_errors as de
from scripts.analyze_model import outcome_report, save_table
from scripts.eda import set_global_seed
from scripts.frc_nn import build_val, collate, get_device, load_config, load_trained
from scripts.retinanet import RetinaConfig, build_retinanet, retina_anchor_params


@torch.no_grad()
def collect_records_retina(model, dataset, cfg, device, batch_size=4):
    sizes, ratios = retina_anchor_params(cfg)
    model.eval()
    use_amp = cfg.amp and device.type == "cuda"
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    records = []
    for imgs, tgts in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model([i.to(device) for i in imgs])
        for img, t, p in zip(imgs, tgts, preds):
            i = int(t["image_id"])
            H, W = img.shape[-2:]
            gt_boxes, gt_labels, ign = dataset.load_annotations(i, W, H)
            s = cfg.imgsz / max(W, H)
            gt_best, _, _ = AA.match_anchors(gt_boxes * s, (int(H * s), int(W * s)), sizes, ratios)
            keep = (p["labels"] > 0).cpu().numpy()                       # drop the background logit column
            records.append({
                "image_id": i, "path": dataset.files[i], "clip": dataset.clip_of(i),
                "gt_boxes": gt_boxes, "gt_labels": gt_labels, "ignore": ign, "props": np.zeros((0, 4)),
                "gt_best_cand_iou": gt_best, "gt_size_in": AA.box_sizes(gt_boxes * s),
                "det_boxes": p["boxes"].float().cpu().numpy()[keep],
                "det_scores": p["scores"].float().cpu().numpy()[keep],
                "det_labels": p["labels"].cpu().numpy()[keep]})
    return records


def analyze_retinanet(run_dir, score_thr=0.3, n_failures=6, device=None):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    device = device or get_device()
    cfg = load_config(run_dir, RetinaConfig)
    set_global_seed(cfg.seed)

    val_ds, _ = build_val(cfg)
    model = load_trained(cfg, run_dir / "last.pt", device, build_fn=build_retinanet)
    print(f"collecting detections for {len(val_ds)} validation frames ...")
    records = collect_records_retina(model, val_ds, cfg, device)

    gt_best = np.concatenate([r["gt_best_cand_iou"] for r in records])
    gt_size = np.concatenate([r["gt_size_in"] for r in records])
    save_table(AA.coverage_summary(gt_best, gt_size), out / "anchor_coverage_val.csv",
               title="validation GT: share whose best anchor reaches IoU >= thr (sizes in input px)")

    gt_df, det_df = outcome_report(records, out, n_props=None, score_thr=score_thr, n_failures=n_failures)
    print(f"results written to {out}")
    return records, gt_df, det_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing config.json and last.pt")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--n-failures", type=int, default=6)
    a = ap.parse_args()
    analyze_retinanet(a.run, a.score_thr, a.n_failures)
