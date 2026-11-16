import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scripts.make_submission as MS
from scripts.frcnn_lib import build_val, coco_evaluate, collate, get_device


@torch.no_grad()
def detect(model, cfg, dataset, device, flip=False):
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=cfg.num_workers, collate_fn=collate)
    out = []
    model.eval()
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            preds = model([i.flip(-1) for i in imgs] if flip else imgs)
        for img, t, p in zip(imgs, tgts, preds):
            W = img.shape[-1]
            b = p["boxes"].float().cpu().numpy()
            if flip:
                b = np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1)
            keep = (p["labels"] > 0).cpu().numpy()
            out.append({"image_id": int(t["image_id"]), "boxes": b[keep],
                        "scores": p["scores"].float().cpu().numpy()[keep], "labels": p["labels"].cpu().numpy()[keep]})
    return sorted(out, key=lambda d: d["image_id"])


def member_name(spec, flip=False):
    return (f"{Path(spec['run']).name}{'_flip' if flip else ''}"
            f"{'_' + str(spec['test_imgsz']) if spec.get('test_imgsz') else ''}")


def spec_member_names(spec):
    return {flip: member_name(spec, flip) for flip in spec.get("flips", (False,))}


def select_specs(specs, selected):
    out = []
    for sp in specs:
        flips = tuple(f for f, n in spec_member_names(sp).items() if n in selected)
        if flips:
            out.append({**sp, "flips": flips})
    return out


def collect_detections(specs, dataset, device=None, cache_dir=None, split="val"):
    device = device or get_device()
    named, seen = [], set()
    for sp in specs:
        run, kind = Path(sp["run"]), sp.get("kind", "faster")
        if not (run / "last.pt").exists():
            raise FileNotFoundError(f"{run / 'last.pt'} not found - check the path of this spec")
        todo = []
        for flip in sp.get("flips", (False,)):
            name = member_name(sp, flip)
            if name in seen:
                raise ValueError(f"two ensemble members are both called {name}; they must differ in run, flip or test_imgsz")
            seen.add(name)
            cache = (Path(cache_dir) / f"{split}_{name}_{sp.get('test_imgsz') or 'native'}_{sp.get('test_proposals') or 'default'}.pkl"
                     if cache_dir else None)
            if cache is not None and cache.exists():
                named.append((name, pickle.loads(cache.read_bytes()), sp.get("weight", 1.0)))
            else:
                todo.append((flip, name, cache))
        if not todo:
            continue
        model, cfg = MS.load_model(run, kind, device, sp.get("test_proposals"), sp.get("test_imgsz"))
        for flip, name, cache in todo:
            print(f"detecting with {name} ...")
            dets = detect(model, cfg, dataset, device, flip)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(pickle.dumps(dets))
            named.append((name, dets, sp.get("weight", 1.0)))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return named


def _iou_one(box, boxes):
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def wbf_image(dets, weights, iou_thr=0.55, skip_thr=0.001):
    weights = np.asarray(weights, dtype=float)
    boxes = np.concatenate([np.asarray(d["boxes"], float).reshape(-1, 4) for d in dets])
    scores = np.concatenate([np.asarray(d["scores"], float) * w for d, w in zip(dets, weights)])
    labels = np.concatenate([np.asarray(d["labels"]).astype(int) for d in dets])
    ok = scores >= skip_thr * weights.min()
    boxes, scores, labels = boxes[ok], scores[ok], labels[ok]
    out_b, out_s, out_l = [], [], []
    for lab in np.unique(labels):
        sel = np.where(labels == lab)[0]
        sel = sel[np.argsort(-scores[sel], kind="stable")]
        fused = np.zeros((len(sel), 4))
        sum_sb, sum_s, count = np.zeros((len(sel), 4)), np.zeros(len(sel)), np.zeros(len(sel), int)
        n = 0
        for i in sel:
            j = -1
            if n:
                ious = _iou_one(boxes[i], fused[:n])
                k = int(np.argmax(ious))
                j = k if ious[k] > iou_thr else -1
            if j < 0:
                j, n = n, n + 1
            sum_sb[j] += scores[i] * boxes[i]
            sum_s[j] += scores[i]
            count[j] += 1
            fused[j] = sum_sb[j] / sum_s[j]
        mean_score = sum_s[:n] / count[:n]
        out_b.append(fused[:n])
        out_s.append(mean_score * np.minimum(len(weights), count[:n]) / weights.sum())
        out_l.append(np.full(n, lab))
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_l)


def fuse_detections(named, iou_thr=0.55, skip_thr=0.001):
    weights = [w for _, _, w in named]
    n_images = len(named[0][1])
    assert all(len(d) == n_images for _, d, _ in named), "models must be evaluated on the same images"
    fused = []
    for k in range(n_images):
        per_model = [d[k] for _, d, _ in named]
        ids = {d["image_id"] for d in per_model}
        assert len(ids) == 1, f"image ids differ at position {k}: {ids}"
        b, s, l = wbf_image(per_model, weights, iou_thr, skip_thr)
        fused.append({"image_id": per_model[0]["image_id"], "boxes": b, "scores": s, "labels": l})
    return fused


def compare_fusion(named, coco_gt, iou_thrs=(0.55,), skip_thr=0.001):
    keys = ("mAP", "AP50", "AP75", "AP_small")
    rows = {}
    for name, dets, _ in named:
        m = coco_evaluate(coco_gt, dets)
        rows[name] = {k: m[k] for k in keys}
    for thr in iou_thrs:
        m = coco_evaluate(coco_gt, fuse_detections(named, thr, skip_thr))
        rows[f"WBF iou_thr={thr}"] = {k: m[k] for k in keys}
    return pd.DataFrame(rows).T


def greedy_select(named, coco_gt, base, iou_thr=0.7, min_gain=0.002, max_add=3, full=False, metric="mAP"):
    by_name = {n: (n, d, w) for n, d, w in named}
    if len(by_name) != len(named):
        raise ValueError(f"duplicate member names: {[n for n, _, _ in named]}")
    unknown = [n for n in base if n not in by_name]
    if unknown:
        raise KeyError(f"unknown members {unknown}; available: {list(by_name)}")

    def score(names):
        m = coco_evaluate(coco_gt, fuse_detections([by_name[n] for n in names], iou_thr))
        return {k: m[k] for k in ("mAP", "AP50", "AP75", "AP_small")}

    cur, rows = list(base), []
    best = score(cur)
    rows.append({"members": "+".join(cur), "action": "start", **best})
    print("start", {k: round(v, 4) for k, v in best.items()})
    candidates = [n for n in by_name if n not in cur]
    added = 0
    while candidates and added < max_add:
        pool = candidates if full else candidates[:1]
        results = {n: score(cur + [n]) for n in pool}
        for n, r in results.items():
            rows.append({"members": "+".join(cur + [n]), "action": f"try {n}", **r})
            print(f"try {n}:", {k: round(v, 4) for k, v in r.items()})
        top = max(results, key=lambda n: results[n][metric])
        if results[top][metric] >= best[metric] + min_gain:
            cur.append(top)
            best, added = results[top], added + 1
            print(f"  -> added {top}")
        candidates = [n for n in candidates if n != top] if (full or results[top][metric] >= best[metric]) else candidates[1:]
        if not full and added == len(cur) - len(base) and top not in cur:
            candidates = candidates  # one-pass mode: the tried candidate was dropped above
    print("selected:", cur, {k: round(v, 4) for k, v in best.items()})
    return cur, pd.DataFrame(rows)


def ensemble_submission(specs, data_root, out_path, device=None, iou_thr=0.55, cache_dir=None, max_images=None):
    device = device or get_device()
    data_root = Path(data_root)
    test_df = pd.read_csv(data_root / "test_images.csv")
    sample_path = data_root / "sample_submission.csv"
    sample_df = pd.read_csv(sample_path, keep_default_na=False) if sample_path.exists() else None
    MS.image_size_check(data_root, test_df)
    used = test_df.iloc[:max_images] if max_images else test_df
    ds = MS.TestImages(data_root, used)
    named = collect_detections(specs, ds, device, cache_dir, split="test" if not max_images else f"test{max_images}")
    fused = fuse_detections(named, iou_thr)
    sizes = used[["width", "height"]].values
    strings = [MS.format_prediction_string(f["boxes"], f["scores"], f["labels"], *sizes[f["image_id"]]) for f in fused]
    ids = list(used["image_id"])
    if max_images:
        rest = test_df.iloc[max_images:]
        ids, strings = ids + list(rest["image_id"]), strings + [""] * len(rest)
    sub = MS.write_submission(strings, ids, sample_df, out_path)
    stats = MS.validate_submission(pd.read_csv(out_path, keep_default_na=False), test_df, sample_df)
    print(f"wrote {out_path}: {stats}")
    return sub
