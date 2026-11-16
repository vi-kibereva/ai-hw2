import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

STRIDES = (8, 16, 32, 64, 128)
DEFAULT_SCALES = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))
SIZE_BINS = [0, 8, 16, 32, np.inf]
SIZE_LABELS = ["<8", "8-16", "16-32", ">=32"]


def anchor_sizes_from_base(base, scales=DEFAULT_SCALES):
    return tuple(tuple(int(b * s) for s in scales) for b in base)


def base_anchors(sizes, ratios):
    sizes, ratios = np.asarray(sizes, float), np.asarray(ratios, float)
    h_r = np.sqrt(ratios)
    w_r = 1.0 / h_r
    ws = (w_r[:, None] * sizes[None]).reshape(-1)
    hs = (h_r[:, None] * sizes[None]).reshape(-1)
    return np.round(np.stack([-ws, -hs, ws, hs], 1) / 2)


def feature_map_sizes(img_hw, strides=STRIDES, divisible=32):
    H = int(np.ceil(img_hw[0] / divisible) * divisible)
    W = int(np.ceil(img_hw[1] / divisible) * divisible)
    return [(int(np.ceil(H / s)), int(np.ceil(W / s))) for s in strides]


def level_anchors(feat_hw, stride, base):
    fh, fw = feat_hw
    cx, cy = np.meshgrid(np.arange(fw) * stride, np.arange(fh) * stride)
    shifts = np.stack([cx.ravel(), cy.ravel(), cx.ravel(), cy.ravel()], 1)
    return (shifts[:, None, :] + base[None, :, :]).reshape(-1, 4)


def _iou(gt, anc):
    lt = np.maximum(gt[:, None, :2], anc[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], anc[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_a = (anc[:, 2] - anc[:, 0]) * (anc[:, 3] - anc[:, 1])
    return inter / np.maximum(area_g[:, None] + area_a[None] - inter, 1e-9)


def match_anchors(gt_boxes, img_hw, sizes_per_level, ratios, iou_pairs=(), strides=STRIDES, chunk=20000):
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    G = len(gt)
    gt_best, gt_level = np.zeros(G, np.float32), np.zeros(G, int)
    counts = np.zeros((len(iou_pairs), 3))
    for lvl, (stride, fhw, sizes) in enumerate(zip(strides, feature_map_sizes(img_hw, strides), sizes_per_level)):
        anc = level_anchors(fhw, stride, base_anchors(sizes, ratios)).astype(np.float32)
        for s in range(0, len(anc), chunk):
            a = anc[s:s + chunk]
            if G:
                iou = _iou(gt, a)
                m = iou.max(1)
                better = m > gt_best
                gt_best[better], gt_level[better] = m[better], lvl
                a_best = iou.max(0)
            else:
                a_best = np.zeros(len(a), np.float32)
            for k, (fg, bg) in enumerate(iou_pairs):
                counts[k, 0] += (a_best >= fg).sum()
                counts[k, 1] += ((a_best >= bg) & (a_best < fg)).sum()
                counts[k, 2] += (a_best < bg).sum()
    return gt_best, gt_level, counts


def sample_scaled_gt(dataset, imgsz, n_images=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(dataset), min(n_images, len(dataset)), replace=False))
    samples = []
    for i in idx:
        W, H = Image.open(dataset.files[i]).size
        boxes, _, _ = dataset.load_annotations(int(i), W, H)
        s = imgsz / max(W, H)
        samples.append({"boxes": boxes.astype(np.float32) * s, "hw": (int(H * s), int(W * s))})
    return samples


def box_sizes(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))


def all_gt_sizes(samples):
    return np.concatenate([box_sizes(s["boxes"]) for s in samples])


def all_gt_aspects(samples):
    b = np.concatenate([np.asarray(s["boxes"]).reshape(-1, 4) for s in samples])
    return (b[:, 3] - b[:, 1]) / np.maximum(b[:, 2] - b[:, 0], 1e-6)


def derive_anchor_base(samples, n_levels=5, p_low=5, min_base=6, max_base=32):
    b = int(np.clip(round(np.percentile(all_gt_sizes(samples), p_low)), min_base, max_base))
    return tuple(b * 2 ** k for k in range(n_levels))


def derive_aspect_ratios(samples, q=(10, 50, 90)):
    r = np.unique(np.round(np.percentile(all_gt_aspects(samples), q), 1))
    return tuple(float(x) for x in r if x > 0)


def evaluate_anchor_config(samples, sizes_per_level, ratios, iou_pairs=((0.5, 0.4), (0.4, 0.3), (0.3, 0.2)),
                           strides=STRIDES):
    best, level, size, counts = [], [], [], np.zeros((len(iou_pairs), 3))
    for s in samples:
        gb, gl, c = match_anchors(s["boxes"], s["hw"], sizes_per_level, ratios, iou_pairs, strides)
        best.append(gb)
        level.append(gl)
        size.append(box_sizes(s["boxes"]))
        counts += c
    return {"gt_best": np.concatenate(best), "gt_level": np.concatenate(level), "gt_size": np.concatenate(size),
            "counts": counts, "n_images": len(samples), "iou_pairs": tuple(iou_pairs),
            "n_anchors_per_location": len(sizes_per_level[0]) * len(ratios)}


def compare_anchor_configs(samples, configs, iou_pairs=((0.5, 0.4), (0.4, 0.3), (0.3, 0.2))):
    return {n: evaluate_anchor_config(samples, sz, rt, iou_pairs) for n, (sz, rt) in configs.items()}


def coverage_summary(gt_best, gt_size, thrs=(0.3, 0.4, 0.5, 0.6)):
    gt_best, gt_size = np.asarray(gt_best), np.asarray(gt_size)
    bins = pd.cut(pd.Series(gt_size), SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str).values
    rows = []
    for label, mask in [("all", np.ones(len(gt_best), bool))] + [(l, bins == l) for l in SIZE_LABELS]:
        if not mask.any():
            continue
        row = {"size_px": label, "n_gt": int(mask.sum()), "mean_best_iou": float(gt_best[mask].mean())}
        row.update({f"share>={t}": float((gt_best[mask] >= t).mean()) for t in thrs})
        rows.append(row)
    return pd.DataFrame(rows).set_index("size_px")


def assignment_summary(res):
    rows = []
    n_gt_img = len(res["gt_best"]) / res["n_images"]
    for k, (fg, bg) in enumerate(res["iou_pairs"]):
        pos, ign, neg = res["counts"][k] / res["n_images"]
        low = float((res["gt_best"] < fg).mean())
        pos_total = pos + low * n_gt_img
        rows.append({"fg_iou": fg, "bg_iou": bg, "gt_per_img": n_gt_img, "positive_anchors_per_img": pos_total,
                     "ignored_anchors_per_img": ign, "negative_anchors_per_img": neg,
                     "negatives_per_positive": neg / max(pos_total, 1.0), "gt_without_positive_anchor": low})
    return pd.DataFrame(rows)


def level_usage(res, n_levels=len(STRIDES)):
    bins = pd.cut(pd.Series(res["gt_size"]), SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str).values
    table = pd.crosstab(bins, res["gt_level"], normalize="index").reindex(columns=range(n_levels), fill_value=0)
    table.columns = [f"P{3 + i}" for i in range(n_levels)]
    return table.reindex([l for l in SIZE_LABELS if l in table.index])


def configs_overview(results, fg=0.5):
    rows = []
    for name, r in results.items():
        cov = coverage_summary(r["gt_best"], r["gt_size"], thrs=(fg,))
        row = {"config": name, "anchors_per_location": r["n_anchors_per_location"],
               "mean_best_iou": cov.loc["all", "mean_best_iou"]}
        row.update({f"covered {l}": cov.loc[l, f"share>={fg}"] for l in ["all"] + SIZE_LABELS if l in cov.index})
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def plot_sizes_vs_anchors(samples, configs):
    sizes = all_gt_sizes(samples)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(sizes, bins=np.logspace(np.log10(max(sizes.min(), 1)), np.log10(sizes.max()), 60), color="lightgray")
    ax.set_xscale("log")
    ax.set_xlabel("sqrt(w*h) of GT boxes in network input pixels")
    ax.set_ylabel("GT boxes")
    top = ax.get_ylim()[1]
    for k, (name, (per_level, _)) in enumerate(configs.items()):
        color = plt.get_cmap("tab10")(k)
        xs = sorted({s for lvl in per_level for s in lvl})
        ax.vlines(xs, 0, top * (0.95 - 0.07 * k), color=color, lw=1.2, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Object sizes vs anchor sizes (vertical lines)")
    return fig


def plot_iou_survival(results, small_px=16):
    ts = np.linspace(0.1, 0.9, 33)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for k, (name, r) in enumerate(results.items()):
        color = plt.get_cmap("tab10")(k)
        axes[0].plot(ts, [(r["gt_best"] >= t).mean() for t in ts], color=color, label=name)
        small = r["gt_size"] < small_px
        if small.any():
            axes[1].plot(ts, [(r["gt_best"][small] >= t).mean() for t in ts], color=color, label=name)
    for ax, title in zip(axes, ["all objects", f"objects < {small_px} px"]):
        for t in (0.3, 0.4, 0.5):
            ax.axvline(t, color="k", ls=":", lw=0.8)
        ax.set_xlabel("IoU threshold t")
        ax.set_title(f"GT with a best-anchor IoU >= t ({title})")
    axes[0].set_ylabel("share of GT boxes")
    axes[0].legend(fontsize=8)
    return fig
