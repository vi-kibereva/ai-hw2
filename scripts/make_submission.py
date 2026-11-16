import argparse
import gc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from scripts.fr_cnn import (Config, build_val, coco_evaluate, collate, get_device, load_config, load_trained, predict,
                       set_test_proposals, set_test_size)
from scripts.retinanet import RetinaConfig, build_retinanet

MAX_DET = 500
N_CLASSES = 8
OUTPUT_NAMES = {"faster": "submission_faster_rcnn.csv", "retina": "submission_retinanet.csv"}


def format_prediction_string(boxes, scores, labels, W, H, max_det=MAX_DET):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    x1, y1 = np.clip(boxes[:, 0], 0, W), np.clip(boxes[:, 1], 0, H)
    x2, y2 = np.clip(boxes[:, 2], 0, W), np.clip(boxes[:, 3], 0, H)
    w, h = x2 - x1, y2 - y1
    ok = (w >= 0.1) & (h >= 0.1) & (labels >= 1) & (labels <= N_CLASSES)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-scores[idx], kind="stable")][:max_det]
    return " ".join(f"{labels[i] - 1} {scores[i]:.5f} {x1[i]:.1f} {y1[i]:.1f} {w[i]:.1f} {h[i]:.1f}" for i in idx)


def parse_prediction_string(s):
    if not isinstance(s, str) or not s.strip():
        return np.zeros(0, int), np.zeros(0), np.zeros((0, 4))
    tokens = s.split()
    if len(tokens) % 6:
        raise ValueError(f"{len(tokens)} tokens, not a multiple of 6")
    a = np.array(tokens, dtype=float).reshape(-1, 6)
    return a[:, 0].astype(int), a[:, 1], a[:, 2:6]


def validate_submission(sub, test_df, sample_df=None, max_det=MAX_DET, tol=1.0):
    problems = []
    if list(sub.columns) != ["image_id", "PredictionString"]:
        problems.append(f"columns are {list(sub.columns)}")
    if sub["image_id"].duplicated().any():
        problems.append("duplicate image_id values")
    ids, expected = set(sub["image_id"]), set(test_df["image_id"])
    if ids != expected:
        problems.append(f"{len(expected - ids)} test images missing, {len(ids - expected)} unknown ids")
    if sample_df is not None and list(sample_df["image_id"]) != list(sub["image_id"]):
        problems.append("row order differs from sample_submission.csv")

    size = test_df.set_index("image_id")[["width", "height"]]
    n_det, bad = [], 0
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        try:
            cls, score, b = parse_prediction_string(s)
        except ValueError as e:
            problems.append(f"{image_id}: {e}")
            continue
        n_det.append(len(cls))
        if image_id not in size.index or not len(cls):
            continue
        W, H = size.loc[image_id]
        wrong = (len(cls) > max_det or (cls < 0).any() or (cls >= N_CLASSES).any() or ((score < 0) | (score > 1)).any()
                 or (b[:, 2] <= 0).any() or (b[:, 3] <= 0).any() or (b[:, 0] < -tol).any() or (b[:, 1] < -tol).any()
                 or (b[:, 0] + b[:, 2] > W + tol).any() or (b[:, 1] + b[:, 3] > H + tol).any())
        if wrong:
            bad += 1
            if bad <= 5:
                problems.append(f"{image_id}: invalid class/score/box values or more than {max_det} detections")
    if bad > 5:
        problems.append(f"... {bad} images with invalid values in total")
    if problems:
        raise ValueError("invalid submission:\n  " + "\n  ".join(problems))
    n_det = np.array(n_det)
    return {"images": len(sub), "empty": int((n_det == 0).sum()), "detections": int(n_det.sum()),
            "mean_per_image": float(n_det.mean()), "max_per_image": int(n_det.max())}


def write_submission(strings, image_ids, sample_df, out_path):
    sub = pd.DataFrame({"image_id": list(image_ids), "PredictionString": list(strings)})
    if sample_df is not None:
        sub = sub.set_index("image_id").loc[sample_df["image_id"]].reset_index()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return sub


class TestImages(torch.utils.data.Dataset):

    def __init__(self, data_root, test_df):
        self.paths = [Path(data_root) / f for f in test_df["file_name"]]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} test images not found, e.g. {missing[0]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF.to_tensor(Image.open(self.paths[i]).convert("RGB")), {"image_id": torch.tensor(i)}


def load_model(run_dir, kind, device, test_proposals=None, test_imgsz=None):
    run_dir = Path(run_dir)
    if kind == "retina":
        cfg = load_config(run_dir, RetinaConfig)
        model = load_trained(cfg, run_dir / "last.pt", device, build_fn=build_retinanet)
    else:
        cfg = load_config(run_dir, Config)
        model = load_trained(cfg, run_dir / "last.pt", device)
        if test_proposals:
            set_test_proposals(model, test_proposals)
    if test_imgsz:
        set_test_size(model, test_imgsz)
    return model, cfg


def detect_test_set(model, cfg, data_root, test_df, device, max_images=None):
    if max_images:
        test_df = test_df.iloc[:max_images]
    ds = TestImages(data_root, test_df)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    preds = predict(model, loader, device, cfg.amp)
    return sorted(preds, key=lambda p: p["image_id"]), test_df


def image_size_check(data_root, test_df, n=50):
    for _, r in test_df.sample(min(n, len(test_df)), random_state=0).iterrows():
        W, H = Image.open(Path(data_root) / r["file_name"]).size
        if (W, H) != (r["width"], r["height"]):
            raise ValueError(f"{r['file_name']}: file is {W}x{H}, csv says {r['width']}x{r['height']}")


def roundtrip_check(model, cfg, device, val_stride=5):
    cfg = replace(cfg, val_stride=val_stride)
    val_ds, coco_gt = build_val(cfg)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    preds = predict(model, loader, device, cfg.amp)
    parsed = []
    for p in preds:
        W, H = Image.open(val_ds.files[p["image_id"]]).size
        cls, score, b = parse_prediction_string(format_prediction_string(p["boxes"], p["scores"], p["labels"], W, H))
        parsed.append({"image_id": p["image_id"], "scores": score, "labels": cls + 1,
                       "boxes": np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])})
    keys = ("mAP", "AP50", "AP_small")
    before, after = coco_evaluate(coco_gt, preds), coco_evaluate(coco_gt, parsed)
    table = pd.DataFrame({"raw detections": {k: before[k] for k in keys}, "after formatting": {k: after[k] for k in keys}})
    print(table.round(4).to_string())
    return table


def generate_submission(run_dir, kind, data_root, out_path, device=None, test_proposals=None,
                        max_images=None, check=False, test_imgsz=None):
    device = device or get_device()
    data_root = Path(data_root)
    test_df = pd.read_csv(data_root / "test_images.csv")
    sample_path = data_root / "sample_submission.csv"
    sample_df = pd.read_csv(sample_path, keep_default_na=False) if sample_path.exists() else None
    image_size_check(data_root, test_df)

    model, cfg = load_model(run_dir, kind, device, test_proposals, test_imgsz)
    if check:
        print(f"[{kind}] round-trip check on validation frames")
        roundtrip_check(model, cfg, device)
    print(f"[{kind}] detecting on {len(test_df) if not max_images else max_images} test images ...")
    preds, used_df = detect_test_set(model, cfg, data_root, test_df, device, max_images)
    sizes = used_df[["width", "height"]].values
    strings = [format_prediction_string(p["boxes"], p["scores"], p["labels"], *sizes[p["image_id"]]) for p in preds]
    ids = used_df["image_id"].values

    if max_images:
        rest = test_df.iloc[max_images:]
        ids, strings = list(ids) + list(rest["image_id"]), strings + [""] * len(rest)
    sub = write_submission(strings, ids, sample_df, out_path)
    stats = validate_submission(pd.read_csv(out_path, keep_default_na=False), test_df, sample_df)
    print(f"[{kind}] wrote {out_path}: {stats}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sub


def generate_all(data_root, runs, out_dir, device=None, test_proposals=None, max_images=None, check=False,
                 test_imgsz=None):
    out = {}
    for kind, run_dir in runs.items():
        path = Path(out_dir) / OUTPUT_NAMES[kind]
        generate_submission(run_dir, kind, data_root, path, device, test_proposals if kind == "faster" else None,
                            max_images, check, test_imgsz)
        out[kind] = path
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with test/, test_images.csv, sample_submission.csv")
    ap.add_argument("--faster", help="Faster R-CNN run directory (config.json + last.pt)")
    ap.add_argument("--retina", help="RetinaNet run directory")
    ap.add_argument("--out-dir", default="submissions")
    ap.add_argument("--test-proposals", type=int, help="override the number of RPN proposals (Faster R-CNN)")
    ap.add_argument("--test-imgsz", type=int, help="inference input size (long side, px), e.g. 1536 or 1920")
    ap.add_argument("--max-images", type=int, help="only run the first N test images (rest left empty) - smoke test")
    ap.add_argument("--check", action="store_true", help="round-trip check of the formatting on validation frames")
    a = ap.parse_args()
    runs = {k: v for k, v in (("faster", a.faster), ("retina", a.retina)) if v}
    if not runs:
        ap.error("give --faster and/or --retina")
    print(generate_all(a.data_root, runs, a.out_dir, None, a.test_proposals, a.max_images, a.check, a.test_imgsz))


if __name__ == "__main__":
    main()
