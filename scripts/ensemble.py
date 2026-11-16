from pathlib import Path

import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png"}


def repeat_factor_yaml(prepared, class_names=None, t=0.2, max_r=4.0, size_factor=1.0, seed=42,
                       split="train", out_name="data_rfs.yaml"):
    root = Path(prepared)
    base = root / "images" / split
    imgs = sorted(p for p in base.rglob("*") if p.suffix.lower() in IMG_EXT)
    present = []
    for p in imgs:
        lab = root / "labels" / split / p.relative_to(base).with_suffix(".txt")
        if lab.exists() and lab.stat().st_size:
            present.append(set(np.loadtxt(lab, ndmin=2)[:, 0].astype(int).tolist()))
        else:
            present.append(set())
    n_cls = max((max(s) for s in present if s), default=0) + 1
    f = np.array([sum(c in s for s in present) for c in range(n_cls)], dtype=float) / len(imgs)
    r_c = np.clip(np.sqrt(t / np.maximum(f, 1e-9)), 1.0, max_r)   
    w = np.array([max((r_c[c] for c in s), default=1.0) for s in present])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=int(len(imgs) * size_factor), replace=True, p=w / w.sum())
    list_path = root / f"train_rfs_t{t}.txt"
    list_path.write_text("\n".join(str(imgs[i]) for i in idx))   

    print(f"{'class':<10}{'frames %':>10}{'repeat r_c':>12}{'drawn x':>9}")
    drawn = np.bincount(idx, minlength=len(imgs))
    for c in range(n_cls):
        have = np.array([c in s for s in present])
        name = class_names[c] if class_names else str(c)
        print(f"{name:<10}{100 * f[c]:>10.1f}{r_c[c]:>12.2f}{(drawn[have].mean() if have.any() else 0):>9.2f}")
    print(f"unique frames in list: {len(set(idx))}/{len(imgs)}, list length {len(idx)}")

    cfg = yaml.safe_load(open(root / "data.yaml"))
    cfg["train"] = str(list_path)
    out = root / out_name
    yaml.safe_dump(cfg, open(out, "w"))
    return str(out)


def predict_raw(weights, paths, imgsz, model_cls, conf=0.001, max_det=1000, bs=16):
    m, out = model_cls(weights), {}
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        for p, r in zip(chunk, m.predict(chunk, imgsz=imgsz, conf=conf, iou=1.0, max_det=max_det,
                                         half=False, device=0, verbose=False)):
            b = r.boxes
            out[p] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
    return out


def _iou_one_to_many(b, boxes):
    x1 = np.maximum(b[0], boxes[:, 0]); y1 = np.maximum(b[1], boxes[:, 1])
    x2 = np.minimum(b[2], boxes[:, 2]); y2 = np.minimum(b[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def _soft_nms_one_class(boxes, scores, sigma, thr):
    kept_b, kept_s = [], []
    while len(scores):
        m = scores.argmax()
        kept_b.append(boxes[m]); kept_s.append(scores[m])
        top = boxes[m]
        boxes, scores = np.delete(boxes, m, 0), np.delete(scores, m)
        if not len(scores):
            break
        scores = scores * np.exp(-(_iou_one_to_many(top, boxes) ** 2) / sigma)  
        keep = scores > thr
        boxes, scores = boxes[keep], scores[keep]
    return np.array(kept_b).reshape(-1, 4), np.array(kept_s)


def soft_nms_preds(preds, sigma=0.5, score_thr=0.001, topk=300):
    out = {}
    for p, (xyxy, sc, cl) in preds.items():
        B, S, C = [], [], []
        for c in np.unique(cl):
            sel = cl == c
            b, s = _soft_nms_one_class(xyxy[sel].astype(float), sc[sel].astype(float), sigma, score_thr)
            B.append(b); S.append(s); C.append(np.full(len(s), c))
        if B:
            B, S, C = np.concatenate(B), np.concatenate(S), np.concatenate(C)
            o = np.argsort(-S)[:topk]
            out[p] = (B[o], S[o], C[o].astype(int))
        else:
            out[p] = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    return out



def _starts(L, tile, overlap):
    if L <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    s = list(range(0, L - tile, stride)) + [L - tile]
    return sorted(set(s))


def tiled_predict(weights, paths, model_cls, tile=960, tile_imgsz=960, overlap=0.25,
                  full_imgsz=None, conf=0.001, nms_iou=0.6, max_det=300):

    import cv2
    import torch
    import torchvision

    m, out = model_cls(weights), {}
    kw = dict(conf=conf, half=False, device=0, verbose=False)
    for p in paths:
        img = cv2.imread(p)
        H, W = img.shape[:2]
        crops = [(x, y, img[y:y + tile, x:x + tile]) for y in _starts(H, tile, overlap) for x in _starts(W, tile, overlap)]
        B, S, C = [], [], []
        for (x, y, _), r in zip(crops, m.predict([c for *_, c in crops], imgsz=tile_imgsz, **kw)):
            b = r.boxes
            xy = b.xyxy.cpu().numpy().copy()
            xy[:, [0, 2]] += x; xy[:, [1, 3]] += y
            B.append(xy); S.append(b.conf.cpu().numpy()); C.append(b.cls.cpu().numpy().astype(int))
        if full_imgsz:                                   # whole frame helps large objects cut by tile borders
            b = m.predict(img, imgsz=full_imgsz, **kw)[0].boxes
            B.append(b.xyxy.cpu().numpy()); S.append(b.conf.cpu().numpy()); C.append(b.cls.cpu().numpy().astype(int))
        B, S, C = np.concatenate(B), np.concatenate(S), np.concatenate(C)
        if len(S):
            keep = torchvision.ops.batched_nms(torch.from_numpy(B).float(), torch.from_numpy(S).float(),
                                               torch.from_numpy(C), nms_iou)[:max_det].numpy()
            B, S, C = B[keep], S[keep], C[keep]
        out[p] = (B, S, C)
    return out


def coco_eval(gt, dets, ids, verbose=False):
    from pycocotools.cocoeval import COCOeval
    nan = dict.fromkeys(["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"], float("nan"))
    if not dets:
        return nan
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = ids
    ev.params.maxDets = [1, 10, 300]
    ev.evaluate(); ev.accumulate()
    if verbose:
        ev.summarize()
    p = ev.eval["precision"]                     

    def ap(area, iou=None):
        x = p[:, :, :, area, 2] if iou is None else p[iou, :, :, area, 2]
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    return {"mAP": ap(0), "AP50": ap(0, 0), "AP75": ap(0, 5),
            "AP_small": ap(1), "AP_medium": ap(2), "AP_large": ap(3)}


def score(gt, id2path, preds, ids=None, verbose=False):
    ids = ids or sorted(id2path)
    dets = [{"image_id": i, "category_id": int(c) + 1, "score": float(s),
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
            for i in ids for (x1, y1, x2, y2), s, c in zip(*preds[id2path[i]])]
    return coco_eval(gt, dets, ids, verbose)


def wbf_preds(pred_list, sizes, weights=None, iou_thr=0.55, skip_box_thr=0.01,
              conf_type="avg", topk=300, paths=None):
    from ensemble_boxes import weighted_boxes_fusion
    out = {}
    for p in (paths or list(pred_list[0])):
        W, H = sizes[p]
        scale = np.array([W, H, W, H], dtype=float)
        bl, sl, ll = [], [], []
        for preds in pred_list:
            xyxy, sc, cl = preds[p]
            bl.append(np.clip(np.asarray(xyxy, float) / scale, 0, 1).tolist())
            sl.append(np.asarray(sc, float).tolist())
            ll.append(np.asarray(cl, int).tolist())
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou_thr,
                                        skip_box_thr=skip_box_thr, conf_type=conf_type)
        o = np.argsort(-s)[:topk]
        out[p] = (b[o] * scale, s[o], l[o].astype(int))
    return out


def write_submission(preds, test_df, paths, out):
    import pandas as pd
    rows = []
    for iid, p in zip(test_df["image_id"], paths):
        xyxy, sc, cl = preds[p]
        s = " ".join(f"{int(c)} {v:.4f} {int(x1)} {int(y1)} {int(x2 - x1)} {int(y2 - y1)}"
                     for (x1, y1, x2, y2), v, c in zip(xyxy, sc, cl))
        rows.append({"image_id": iid, "PredictionString": s})
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "rows")