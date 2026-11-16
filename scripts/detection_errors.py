import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from scripts.eda import CLASS_NAMES, N_CLASSES

SIZE_BINS = [0, 8, 16, 32, 96, np.inf]
SIZE_LABELS = ["<8", "8-16", "16-32", "32-96", ">96"]
GT_OUTCOMES = ["detected", "class_confusion", "mislocalised", "no_proposal", "not_scored"]
DET_OUTCOMES = ["tp", "duplicate", "class_confusion", "mislocalised", "background"]
OUTCOME_COLORS = {"detected": "tab:green", "class_confusion": "tab:purple", "mislocalised": "tab:orange",
                  "no_proposal": "tab:red", "not_scored": "tab:gray"}


def pairwise_iou_xyxy(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def box_size(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))


def box_centers(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)


def centers_in_regions(boxes, regions):
    c = box_centers(boxes)
    regions = np.asarray(regions, dtype=float).reshape(-1, 4)
    inside = np.zeros(len(c), dtype=bool)
    for x1, y1, x2, y2 in regions:
        inside |= (c[:, 0] >= x1) & (c[:, 0] <= x2) & (c[:, 1] >= y1) & (c[:, 1] <= y2)
    return inside


def _max_over(iou, n=None):
    iou = iou[:, :n] if n is not None else iou
    return iou.max(1) if iou.shape[1] else np.zeros(iou.shape[0])


def proposal_recall(records, ns=(100, 300, 1000, 2000), iou_thrs=(0.5, 0.7), small_px=32):
    best = {n: [] for n in ns}
    sizes = []
    for r in records:
        if len(r["gt_boxes"]) == 0:
            continue
        iou = pairwise_iou_xyxy(r["gt_boxes"], r["props"])
        for n in ns:
            best[n].append(_max_over(iou, n))
        sizes.append(box_size(r["gt_boxes"]))
    sizes = np.concatenate(sizes)
    rows = []
    for n in ns:
        b = np.concatenate(best[n])
        for thr in iou_thrs:
            rows.append({"n_proposals": n, "iou_thr": thr, "recall_all": (b >= thr).mean(),
                         f"recall_<{small_px}px": (b[sizes < small_px] >= thr).mean() if (sizes < small_px).any() else np.nan,
                         f"recall_>={small_px}px": (b[sizes >= small_px] >= thr).mean() if (sizes >= small_px).any() else np.nan,
                         "mean_best_iou": b.mean()})
    return pd.DataFrame(rows)


def gt_outcomes(records, n_props, score_thr=0.3, iou_thr=0.5, loc_thr=0.1):
    rows = []
    for i, r in enumerate(records):
        gb, gl = np.asarray(r["gt_boxes"]).reshape(-1, 4), np.asarray(r["gt_labels"])
        if len(gb) == 0:
            continue
        sel = np.asarray(r["det_scores"]) >= score_thr
        db = np.asarray(r["det_boxes"]).reshape(-1, 4)[sel]
        ds, dl = np.asarray(r["det_scores"])[sel], np.asarray(r["det_labels"])[sel]
        if "gt_best_cand_iou" in r:
            best_prop = np.asarray(r["gt_best_cand_iou"], dtype=float)
        else:
            best_prop = _max_over(pairwise_iou_xyxy(gb, r["props"]), n_props)
        iou = pairwise_iou_xyxy(gb, db)
        sizes = box_size(gb)
        for g in range(len(gb)):
            ious = iou[g] if iou.shape[1] else np.zeros(0)
            same = dl == gl[g]
            hit = ious >= iou_thr
            confused = -1
            if (hit & same).any():
                outcome = "detected"
            elif (hit & ~same).any():
                cand = np.where(hit & ~same)[0]
                confused = int(dl[cand[np.argmax(ds[cand])]]) - 1
                outcome = "class_confusion"
            elif (same & (ious >= loc_thr)).any():
                outcome = "mislocalised"
            elif best_prop[g] < iou_thr:
                outcome = "no_proposal"
            else:
                outcome = "not_scored"
            same_iou = ious[same].max() if same.any() else 0.0
            rows.append({"image": i, "clip": r["clip"], "cls": int(gl[g]) - 1, "class_name": CLASS_NAMES[int(gl[g]) - 1],
                         "size_px": sizes[g], "outcome": outcome, "confused_as": confused,
                         "best_prop_iou": best_prop[g], "best_det_iou": same_iou,
                         "x1": gb[g, 0], "y1": gb[g, 1], "x2": gb[g, 2], "y2": gb[g, 3]})
    return pd.DataFrame(rows)


def det_outcomes(records, score_thr=0.3, iou_thr=0.5, loc_thr=0.1):
    rows = []
    for i, r in enumerate(records):
        sel = np.asarray(r["det_scores"]) >= score_thr
        if not sel.any():
            continue
        db = np.asarray(r["det_boxes"]).reshape(-1, 4)[sel]
        ds, dl = np.asarray(r["det_scores"])[sel], np.asarray(r["det_labels"])[sel]
        order = np.argsort(-ds)
        db, ds, dl = db[order], ds[order], dl[order]
        gb, gl = np.asarray(r["gt_boxes"]).reshape(-1, 4), np.asarray(r["gt_labels"])
        skip = centers_in_regions(db, r["ignore"]) if len(r["ignore"]) else np.zeros(len(db), bool)
        iou = pairwise_iou_xyxy(db, gb)
        matched = np.zeros(len(gb), dtype=bool)
        for d in range(len(db)):
            if skip[d]:
                continue
            ious = iou[d] if iou.shape[1] else np.zeros(0)
            same = gl == dl[d]
            cand = np.where(same & (ious >= iou_thr))[0]
            if len(cand):
                free = cand[~matched[cand]]
                if len(free):
                    matched[free[np.argmax(ious[free])]] = True
                    outcome = "tp"
                else:
                    outcome = "duplicate"
            elif ((ious >= iou_thr) & ~same).any():
                outcome = "class_confusion"
            elif (same & (ious >= loc_thr)).any():
                outcome = "mislocalised"
            else:
                outcome = "background"
            rows.append({"image": i, "clip": r["clip"], "cls": int(dl[d]) - 1, "class_name": CLASS_NAMES[int(dl[d]) - 1],
                         "score": ds[d], "outcome": outcome,
                         "x1": db[d, 0], "y1": db[d, 1], "x2": db[d, 2], "y2": db[d, 3]})
    return pd.DataFrame(rows)


def summarize_outcomes(gt_df, by):
    g = gt_df.copy()
    g["size_bin"] = pd.cut(g["size_px"], SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    key = {"class": "class_name", "size": "size_bin", "clip": "clip"}[by]
    t = pd.crosstab(g[key], g["outcome"], normalize="index").reindex(columns=GT_OUTCOMES, fill_value=0)
    t.insert(0, "n_gt", g.groupby(key).size())
    if by == "size":
        t = t.reindex([s for s in SIZE_LABELS if s in t.index])
    elif by == "class":
        t = t.reindex([c for c in CLASS_NAMES if c in t.index])
    return t


def confusion_matrix(gt_df):
    m = np.zeros((N_CLASSES, N_CLASSES + 1))
    for cls, outcome, conf in zip(gt_df["cls"], gt_df["outcome"], gt_df["confused_as"]):
        if outcome == "detected":
            m[cls, cls] += 1
        elif outcome == "class_confusion":
            m[cls, conf] += 1
        else:
            m[cls, N_CLASSES] += 1
    return m


def refinement_gain(gt_df):
    d = gt_df[gt_df["outcome"] == "detected"]
    return {"n_detected": len(d), "mean_best_proposal_iou": float(d["best_prop_iou"].mean()),
            "mean_final_box_iou": float(d["best_det_iou"].mean())}


def select_failure_frames(gt_df, det_df, n=6):
    fn = gt_df[gt_df["outcome"] != "detected"].groupby("image").size()
    fp = det_df[det_df["outcome"] != "tp"].groupby("image").size()
    total = fn.add(fp, fill_value=0).sort_values(ascending=False)
    clip_of = pd.concat([gt_df[["image", "clip"]], det_df[["image", "clip"]]]).drop_duplicates("image").set_index("image")["clip"]
    chosen, seen = [], set()
    for img in total.index:
        if clip_of[img] not in seen:
            chosen.append(int(img))
            seen.add(clip_of[img])
        if len(chosen) == n:
            break
    return chosen


def plot_outcome_breakdown(table, title, max_rows=20):
    t = table if len(table) <= max_rows else table.sort_values("detected").head(max_rows)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(t) + 1.8))
    left = np.zeros(len(t))
    for o in GT_OUTCOMES:
        ax.barh(range(len(t)), t[o].values, left=left, color=OUTCOME_COLORS[o], label=o)
        left += t[o].values
    ax.set_yticks(range(len(t)), [f"{i} (n={int(n)})" for i, n in zip(t.index, t["n_gt"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of ground-truth boxes")
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return fig


def plot_confusion(m):
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = CLASS_NAMES + ["missed"]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("What happens to each GT box (row-normalised)")
    fig.colorbar(im, ax=ax)
    return fig


def hotspot_window(points, W, H, cw, ch):
    cw, ch = min(cw, W), min(ch, H)
    if len(points) == 0:
        return 0, 0, cw, ch
    best, best_n = None, -1
    for px, py in points:
        x0 = float(np.clip(px - cw / 2, 0, W - cw))
        y0 = float(np.clip(py - ch / 2, 0, H - ch))
        n = ((points[:, 0] >= x0) & (points[:, 0] <= x0 + cw) & (points[:, 1] >= y0) & (points[:, 1] <= y0 + ch)).sum()
        if n > best_n:
            best, best_n = (x0, y0, x0 + cw, y0 + ch), n
    return best


def _draw_errors(ax, img, g, d, window=None):
    ax.imshow(img)
    for r in g.itertuples():
        missed = r.outcome != "detected"
        ax.add_patch(patches.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fc="none",
                                       ec="red" if missed else "lime", lw=1.6 if missed else 0.7))
    for r in d.itertuples():
        tp = r.outcome == "tp"
        ax.add_patch(patches.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fc="none",
                                       ec="cyan" if tp else "orange", lw=0.9 if tp else 1.6, ls="--"))
    if window is not None:
        ax.set_xlim(window[0], window[2])
        ax.set_ylim(window[3], window[1])
    ax.axis("off")


def plot_failure_cases(records, gt_df, det_df, image_ids, crop=(480, 270)):
    fig, axes = plt.subplots(len(image_ids), 2, figsize=(17, 4.7 * len(image_ids)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    for (a_full, a_zoom), i in zip(axes, image_ids):
        r = records[i]
        img = Image.open(r["path"]).convert("RGB")
        W, H = img.size
        g, d = gt_df[gt_df["image"] == i], det_df[det_df["image"] == i]
        fn, fp = g[g["outcome"] != "detected"], d[d["outcome"] != "tp"]
        pts = np.concatenate([box_centers(fn[["x1", "y1", "x2", "y2"]].values),
                              box_centers(fp[["x1", "y1", "x2", "y2"]].values)])
        win = hotspot_window(pts, W, H, *crop)
        _draw_errors(a_full, img, g, d)
        a_full.add_patch(patches.Rectangle((win[0], win[1]), win[2] - win[0], win[3] - win[1],
                                           fc="none", ec="yellow", lw=2))
        a_full.set_title(f"{r['clip']}/{r['path'].stem}   missed GT {len(fn)}: {fn['outcome'].value_counts().to_dict()}\n"
                         f"false positives {len(fp)}: {fp['outcome'].value_counts().to_dict()}", fontsize=9)
        _draw_errors(a_zoom, img, g, d, window=win)
    return fig
