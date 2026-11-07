import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DATA_ROOT = Path("../object_detection_data")
OUT_ROOT = Path("prepared")
REPORT_DIR = Path("eda_report")
SEED = 42

USE_OFFICIAL_SPLIT = False
VAL_FRACTION = 0.12
STEP = None
MASK_IGNORE_SPLITS = ("train",)
IGNORE_FILL = 114
CANDIDATE_IMGSZ = (640, 960, 1280, 1536)

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
N_CLASSES = len(CLASS_NAMES)
CLASS_COLORS = [plt.get_cmap("tab10")(i) for i in range(N_CLASSES)]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def finish_figure(fig, name, out_dir, show=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def xywh_to_yolo(x, y, w, h, W, H):
    x1, y1 = np.clip(x, 0, W), np.clip(y, 0, H)
    x2, y2 = np.clip(x + w, 0, W), np.clip(y + h, 0, H)
    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H


def yolo_to_xywh(cx, cy, w, h, W, H):
    return (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H


def read_yolo_txt(path, W, H, with_class=True):
    ncol = 5 if with_class else 4
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, ncol))
    arr = np.loadtxt(path, ndmin=2)
    out = arr.copy()
    out[:, ncol - 4:] = np.stack(yolo_to_xywh(*arr[:, ncol - 4:].T, W, H), axis=1)
    return out


def box_iou_xywh(a, b):
    ix = np.clip(np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def letterbox_scale(width, height, imgsz):
    return imgsz / np.maximum(width, height)


def draw_frame(ax, img, boxes, ignores, title=""):
    ax.imshow(img)
    for x, y, w, h in ignores:
        ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.5, ec="white", fc="gray",
                                       alpha=0.35, ls="--"))
    for cls, x, y, w, h in boxes:
        ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.2, fc="none",
                                       ec=CLASS_COLORS[int(cls) % N_CLASSES]))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def load_coco_split(data_root, split, id_offset=0):
    with open(Path(data_root) / split / "annotations_coco.json") as f:
        coco = json.load(f)

    images = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id", "frame_id": "frame"})
    images["image_id"] += id_offset
    paths = images["file_name"].map(Path)
    video = images["video"].astype(str) if "video" in images else pd.Series("", index=images.index)
    images["clip"] = [p.parent.name or v for p, v in zip(paths, video)]
    images["stem"] = paths.map(lambda p: p.stem)
    images["ext"] = paths.map(lambda p: p.suffix or ".jpg")
    if "frame" not in images:
        images["frame"] = images["stem"].astype(int)
    images["official_split"] = split
    images["src_path"] = [Path(data_root) / split / "images" / c / (s + e)
                          for c, s, e in zip(images["clip"], images["stem"], images["ext"])]

    anns = pd.DataFrame(coco["annotations"])
    anns[["x", "y", "w", "h"]] = np.array(anns["bbox"].tolist(), dtype=float)
    anns["cls"] = anns["category_id"] - 1
    anns["image_id"] += id_offset
    if "iscrowd" not in anns:
        anns["iscrowd"] = 0
    keep = ["image_id", "cls", "x", "y", "w", "h", "iscrowd", "truncation", "occlusion", "track_id"]
    anns = anns[[c for c in keep if c in anns]]
    meta = images[["image_id", "clip", "frame", "width", "height", "official_split"]]
    return images, anns.merge(meta, on="image_id")


def load_dataset(data_root, splits=("train", "val")):
    parts = [load_coco_split(data_root, s, id_offset=i * 10**8) for i, s in enumerate(splits)]
    images = pd.concat([p[0] for p in parts], ignore_index=True)
    anns = pd.concat([p[1] for p in parts], ignore_index=True)

    clash = images.groupby("clip")["official_split"].nunique()
    clash = clash[clash > 1].index
    if len(clash):
        for df in (images, anns):
            df["clip"] = np.where(df["clip"].isin(clash), df["official_split"] + "_" + df["clip"], df["clip"])

    valid = (anns["w"] > 0) & (anns["h"] > 0)
    objects = anns[(anns["iscrowd"] == 0) & anns["cls"].between(0, N_CLASSES - 1) & valid].copy()
    objects["size"] = np.sqrt(objects["w"] * objects["h"])
    ignores = anns[anns["iscrowd"] == 1].copy()
    return images, objects, ignores


def frame_table(images, objects, ignores):
    f = images[["image_id", "clip", "frame", "width", "height", "official_split",
                "stem", "ext", "src_path"]].copy().set_index("image_id")
    f["n_objects"] = objects.groupby("image_id").size().reindex(f.index, fill_value=0)
    f["n_ignore"] = ignores.groupby("image_id").size().reindex(f.index, fill_value=0)
    return f.reset_index()


def class_counts_by_clip(objects, clip_index):
    return pd.crosstab(objects["clip"], objects["cls"]).reindex(
        index=clip_index, columns=range(N_CLASSES), fill_value=0)


def clip_table(frames, objects):
    t = frames.groupby("clip").agg(
        split=("official_split", "first"), n_frames=("image_id", "size"),
        width=("width", "first"), height=("height", "first"),
        objs_per_frame=("n_objects", "mean"), ignore_per_frame=("n_ignore", "mean"))
    g = objects.groupby("clip")
    t["median_size_px"] = g["size"].median()
    t["small_frac"] = g["size"].apply(lambda s: (s < 32).mean())      # COCO "small" (< 32x32 px)
    if "occlusion" in objects:
        t["occluded_frac"] = g["occlusion"].apply(lambda s: (s > 0).mean())
    return t


def compare_resolutions(images, test_csv):
    table = images.groupby(["width", "height"]).size().rename("train_val_frames").to_frame()
    if Path(test_csv).exists():
        test = pd.read_csv(test_csv).groupby(["width", "height"]).size().rename("test_frames")
        table = table.join(test, how="outer").fillna(0).astype(int)
    return table


def size_summary_by_class(objects):
    def stats(s):
        return pd.Series({"count": len(s), "p5": s.quantile(.05), "median": s.median(),
                          "p95": s.quantile(.95), "<16px %": 100 * (s < 16).mean(),
                          "<32px %": 100 * (s < 32).mean(), ">96px %": 100 * (s > 96).mean()})
    table = objects.groupby("cls")["size"].apply(stats).unstack()
    table.index = [CLASS_NAMES[i] for i in table.index]
    table.loc["ALL"] = stats(objects["size"])
    return table.round(1)


def plot_per_clip_range(ax, grouped, ylabel, log=False):
    q = grouped.quantile([0.1, 0.5, 0.9]).unstack().sort_values(0.5)
    x = np.arange(len(q))
    ax.vlines(x, q[0.1], q[0.9], color="tab:blue", alpha=0.5)
    ax.plot(x, q[0.5], "o", color="tab:red", ms=3)
    ax.set_xlabel("clips (sorted by median)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    if log:
        ax.set_yscale("log")


def plot_class_distribution(counts_by_clip):
    totals = counts_by_clip.sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    a1.bar(CLASS_NAMES, totals.values, color=CLASS_COLORS)
    a1.set_yscale("log")
    a1.set_title("Objects per class (log scale)")
    a1.tick_params(axis="x", rotation=45)
    for i, v in enumerate(totals.values):
        a1.text(i, max(v, 1), f"{v / totals.sum():.1%}", ha="center", va="bottom", fontsize=8)
    im = a2.imshow(np.log10(1 + counts_by_clip.values), aspect="auto", cmap="viridis")
    a2.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=45)
    a2.set_yticks([])
    a2.set_ylabel(f"{len(counts_by_clip)} clips (sorted by name)")
    a2.set_title("Class counts per clip")
    fig.colorbar(im, ax=a2, label="log10(1 + #objects)")
    return fig


def plot_size_distribution(objects):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.logspace(np.log10(max(objects["size"].min(), 1)), np.log10(objects["size"].max()), 60)
    axes[0].hist(objects["size"], bins=bins, color="tab:blue")
    axes[0].set_xscale("log")
    for px in (8, 16, 32, 96):
        axes[0].axvline(px, color="k", ls=":", lw=1)
    axes[0].set_xlabel("sqrt(w*h) [px]")
    axes[0].set_title("Object size (dotted: 8, 16, 32, 96 px)")

    axes[1].boxplot([objects.loc[objects["cls"] == c, "size"] for c in range(N_CLASSES)], showfliers=False)
    axes[1].set_xticks(range(1, N_CLASSES + 1), CLASS_NAMES, rotation=45)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("sqrt(w*h) [px]")
    axes[1].set_title("Size per class")

    plot_per_clip_range(axes[2], objects.groupby("clip")["size"], "sqrt(w*h) [px]", log=True)
    axes[2].set_title("Size per clip (median, 10-90 %)")
    return fig


def plot_objects_per_frame(frames):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    n = frames["n_objects"]
    a1.hist(n, bins=60, color="tab:blue")
    a1.set_xlabel("objects per frame")
    a1.set_ylabel("frames")
    a1.set_title(f"Objects per frame (mean {n.mean():.1f}, median {n.median():.0f}, max {n.max()})")
    plot_per_clip_range(a2, frames.groupby("clip")["n_objects"], "objects per frame")
    a2.set_title("Objects per frame per clip (median, 10-90 %)")
    return fig


def size_coverage_by_imgsz(objects, candidates=CANDIDATE_IMGSZ, tiny_px=8, small_px=16):
    rows = []
    for s in candidates:
        px = objects["size"].values * letterbox_scale(objects["width"].values, objects["height"].values, s)
        rows.append({"imgsz": s, "median_px": np.median(px),
                     "frac_tiny": (px < tiny_px).mean(), "frac_small": (px < small_px).mean()})
    return pd.DataFrame(rows)


def calc_imgsz(coverage, max_tiny_frac=0.10):
    ok = coverage[coverage["frac_tiny"] <= max_tiny_frac]
    row = ok.iloc[0] if len(ok) else coverage.iloc[-1]
    return int(row["imgsz"])


def _wh_iou(wh, centroids):
    inter = (np.minimum(wh[:, None, 0], centroids[None, :, 0]) *
             np.minimum(wh[:, None, 1], centroids[None, :, 1]))
    union = wh[:, 0:1] * wh[:, 1:2] + (centroids[:, 0] * centroids[:, 1])[None] - inter
    return inter / union


def kmeans_anchors(wh, k=9, iters=100, seed=SEED):
    rng = np.random.default_rng(seed)
    if len(wh) > 200_000:
        wh = wh[rng.choice(len(wh), 200_000, replace=False)]
    c = wh[rng.choice(len(wh), k, replace=False)].astype(float)
    for _ in range(iters):
        assign = _wh_iou(wh, c).argmax(1)
        new = np.array([np.median(wh[assign == j], 0) if (assign == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    c = c[np.argsort(c.prod(1))]
    return c, _wh_iou(wh, c).max(1).mean()


def track_iou_by_gap(objects, gaps=(1, 2, 3, 5, 8, 10, 15, 30)):
    if "track_id" not in objects:
        return None
    base = objects[["clip", "track_id", "frame", "x", "y", "w", "h"]]
    rows = []
    for g in gaps:
        shifted = base.assign(frame=base["frame"] + g)
        m = base.merge(shifted, on=["clip", "track_id", "frame"], suffixes=("_a", "_b"))
        iou = box_iou_xywh(m[["x_a", "y_a", "w_a", "h_a"]].values, m[["x_b", "y_b", "w_b", "h_b"]].values)
        rows.append({"gap": g, "persistence": len(m) / len(base),
                     "mean_iou": iou.mean() if len(iou) else np.nan,
                     "median_iou": np.median(iou) if len(iou) else np.nan})
    return pd.DataFrame(rows)


def calc_step(gap_table, iou_target=0.6):
    if gap_table is None:
        return 1
    ok = gap_table[gap_table["mean_iou"] >= iou_target]
    return int(ok["gap"].max()) if len(ok) else 1


def report_decisions(objects, candidates=CANDIDATE_IMGSZ):
    coverage = size_coverage_by_imgsz(objects, candidates)
    imgsz = calc_imgsz(coverage)
    scale = letterbox_scale(objects["width"].values, objects["height"].values, imgsz)
    anchors, anchor_iou = kmeans_anchors(np.stack([objects["w"].values * scale, objects["h"].values * scale], 1))
    gap_table = track_iou_by_gap(objects)
    step = calc_step(gap_table)
    counts = objects["cls"].value_counts()

    print("Object size after letterbox resize (tiny < 8 px, small < 16 px)")
    print(coverage.round(3).to_string(index=False))
    print(f"Optimal imgsz: {imgsz}")
    print(f"K-means anchors at imgsz={imgsz} (w, h in input px; mean best IoU {anchor_iou:.2f})")
    print(anchors.round(1))
    print("Frame similarity vs gap (same track_id)")
    print(gap_table.round(3).to_string(index=False) if gap_table is not None else "no track_id available")
    print(f"Best subsampling step: {step}")
    print(f"Class imbalance: most/least frequent class ratio: {counts.max() / counts.min():.0f}x "
          f"({CLASS_NAMES[counts.idxmax()]} vs {CLASS_NAMES[counts.idxmin()]})")
    small_frac = float(coverage.loc[coverage["imgsz"] == imgsz, "frac_small"].iloc[0])
    return {"imgsz": imgsz, "step": step, "anchors": anchors, "small_frac": small_frac}


def frame_to_yolo_rows(g, W, H, min_px=1.0):
    cx, cy, w, h = xywh_to_yolo(g["x"].values, g["y"].values, g["w"].values, g["h"].values, W, H)
    keep = (w * W >= min_px) & (h * H >= min_px)
    return np.stack([g["cls"].values, cx, cy, w, h], axis=1)[keep]


def verify_conversion(objects, frames, data_root):
    expected = objects.groupby("image_id").size().reindex(frames["image_id"], fill_value=0).values
    shipped = []
    for r in frames.itertuples():
        p = Path(data_root) / r.official_split / "labels" / Path(r.src_path).parent.name / f"{r.stem}.txt"
        shipped.append(sum(1 for line in open(p) if line.strip()) if p.exists() else 0)
    bad = frames.assign(coco_count=expected, shipped_count=shipped)
    bad = bad[bad["coco_count"] != bad["shipped_count"]]
    print(f"Conversion check: {len(bad)} / {len(frames)} frames differ from the shipped YOLO labels")
    return bad


def split_by_clip(clips, counts_by_clip, val_fraction, seed, n_trials=200):
    names = np.array(clips.index)
    n_val = max(1, round(len(names) * val_fraction))
    cc = counts_by_clip.loc[names].values.astype(float)
    n_obj, n_frames = cc.sum(1), clips["n_frames"].values.astype(float)
    p_all, dens_all = cc.sum(0) / cc.sum(), n_obj.sum() / n_frames.sum()
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(n_trials):
        idx = rng.permutation(len(names))[:n_val]
        p = cc[idx].sum(0) / max(cc[idx].sum(), 1)
        dens = n_obj[idx].sum() / n_frames[idx].sum()
        score = np.abs(p - p_all).sum() + 0.5 * abs(dens / dens_all - 1)
        if score < best_score:
            best, best_score = idx, score
    val = set(names[best])
    return {c: ("val" if c in val else "train") for c in names}


def official_clip_split(clips):
    return clips["split"].to_dict()


def assert_no_clip_leakage(frames):
    n = frames.groupby("clip")["split"].nunique()
    assert (n == 1).all(), f"clips in several splits: {list(n[n > 1].index)}"
    print("OK: every clip belongs to exactly one split")


def describe_split(frames, objects):
    o = objects.merge(frames[["image_id", "split"]], on="image_id")
    summary = frames.groupby("split").agg(clips=("clip", "nunique"), frames=("image_id", "size"))
    summary["objects"] = o.groupby("split").size()
    summary["objs_per_frame"] = (summary["objects"] / summary["frames"]).round(1)
    mix = pd.crosstab(o["split"], o["cls"], normalize="index").reindex(columns=range(N_CLASSES), fill_value=0)
    mix.columns = CLASS_NAMES
    print("Split summary")
    print(summary.to_string())
    print("Class mix per split [%]")
    print((100 * mix).round(2).to_string())


def subsample_frames(frames, step):
    frames = frames.sort_values(["clip", "frame"])
    return frames[frames.groupby("clip").cumcount() % step == 0]


def mask_ignore_regions(img, regions, fill=IGNORE_FILL):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for x, y, w, h in regions:
        draw.rectangle([x, y, x + w, y + h], fill=(fill,) * 3)
    return img


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(Path(src).resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_rows(path, rows, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt=fmt)


def write_data_yaml(out_root):
    lines = [f"path: {Path(out_root).resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    path = Path(out_root) / "data.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def build_yolo_dataset(frames, objects, ignores, out_root, mask_splits=MASK_IGNORE_SPLITS):
    out_root = Path(out_root).resolve()
    for sub in ("images", "labels", "ignore"):
        shutil.rmtree(out_root / sub, ignore_errors=True)
    obj_by_img = {k: g for k, g in objects.groupby("image_id")}
    ign_by_img = {k: g for k, g in ignores.groupby("image_id")}
    empty = objects.iloc[0:0]

    for r in frames.itertuples():
        W, H, rel = r.width, r.height, Path(r.split) / r.clip
        img_dst = out_root / "images" / rel / f"{r.stem}{r.ext}"
        lab_dst = out_root / "labels" / rel / f"{r.stem}.txt"

        write_rows(lab_dst, frame_to_yolo_rows(obj_by_img.get(r.image_id, empty), W, H), "%d %.6f %.6f %.6f %.6f")

        ign = ign_by_img.get(r.image_id)
        if ign is not None:
            cx, cy, w, h = xywh_to_yolo(ign["x"].values, ign["y"].values, ign["w"].values, ign["h"].values, W, H)
            write_rows(out_root / "ignore" / rel / f"{r.stem}.txt", np.stack([cx, cy, w, h], 1), "%.6f")

        if ign is not None and r.split in mask_splits:
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            masked = mask_ignore_regions(Image.open(r.src_path).convert("RGB"), ign[["x", "y", "w", "h"]].values)
            masked.save(img_dst, quality=95, subsampling=0)
        else:
            link_or_copy(r.src_path, img_dst)

    yaml_path = write_data_yaml(out_root)
    print(f"\nwrote {len(frames)} frames to {out_root}")
    return yaml_path


def visualize_samples(dataset_root, split="train", n=6, seed=SEED, cols=2):
    root = Path(dataset_root)
    files = sorted(p for p in (root / "images" / split).glob("*/*.*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = random.Random(seed).sample(files, min(n, len(files)))
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.8 * rows), squeeze=False)
    for ax, p in zip(axes.flat, files):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        boxes = read_yolo_txt(root / "labels" / split / p.parent.name / f"{p.stem}.txt", W, H)
        ign = read_yolo_txt(root / "ignore" / split / p.parent.name / f"{p.stem}.txt", W, H, with_class=False)
        draw_frame(ax, img, boxes, ign, f"{split}: {p.parent.name}/{p.stem}  ({W}x{H}), {len(boxes)} objects")
    for ax in list(axes.flat)[len(files):]:
        ax.axis("off")
    handles = [patches.Patch(ec=CLASS_COLORS[i], fc="none", label=n_) for i, n_ in enumerate(CLASS_NAMES)]
    handles.append(patches.Patch(ec="white", fc="gray", alpha=0.5, label="ignore region"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
    return fig


def build_train_config(data_yaml, imgsz, small_frac, seed=SEED):
    return {
        "data": str(data_yaml), "imgsz": imgsz, "rect": False,
        "seed": seed, "deterministic": True,
        "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.0,
        "scale": 0.3 if small_frac > 0.5 else 0.5, "translate": 0.1,
        "degrees": 0.0, "shear": 0.0, "perspective": 0.0,
        "fliplr": 0.5, "flipud": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    }


def load_tables(data_root):
    images, objects, ignores = load_dataset(data_root)
    frames = frame_table(images, objects, ignores)
    clips = clip_table(frames, objects)
    counts = class_counts_by_clip(objects, clips.index)
    return images, objects, ignores, frames, clips, counts


def split_and_build(frames, objects, ignores, clips, counts, step, out_root):
    split_map = official_clip_split(clips) if USE_OFFICIAL_SPLIT else split_by_clip(clips, counts, VAL_FRACTION, SEED)
    frames = frames.assign(split=frames["clip"].map(split_map))
    assert_no_clip_leakage(frames)
    describe_split(frames, objects)
    kept = subsample_frames(frames, step)
    print(f"Subsampling step {step}: {len(frames)} -> {len(kept)} frames")
    return build_yolo_dataset(kept, objects, ignores, out_root)


def rebuild_dataset(data_root, out_root, step):
    set_global_seed(SEED)
    _, objects, ignores, frames, clips, counts = load_tables(Path(data_root))
    return split_and_build(frames, objects, ignores, clips, counts, step, Path(out_root))
