import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import scripts.anchor_analysis as AA
from scripts.eda import finish_figure
from scripts.fr_cnn import VisDroneDet, get_device
from scripts.retinanet import RetinaConfig, build_retinanet, describe_retina_setup, retina_anchor_params
from scripts.run_experiments import apply_overrides, parse_set, run_experiment

TABLE_COLS = ["stage", "experiment", "anchor_base", "aspect_ratios", "fg_iou", "bg_iou", "alpha", "gamma",
              "mAP", "AP50", "AP_small", "AP_medium", "AP_large", "ms_per_img", "train_min"]
COMPARE_COLS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500", "ms_per_img", "fps", "train_min"]

IOU_CANDIDATES = {"fg0.4_bg0.3": {"fg_iou_thresh": 0.4, "bg_iou_thresh": 0.3},
                  "fg0.3_bg0.2": {"fg_iou_thresh": 0.3, "bg_iou_thresh": 0.2}}
FOCAL_CANDIDATES = {"alpha0.5": {"focal_alpha": 0.5},
                    "gamma1": {"focal_gamma": 1.0},
                    "gamma3": {"focal_gamma": 3.0}}


def retina_params(cfg):
    return {"anchor_base": str(tuple(cfg.anchor_base)), "aspect_ratios": str(tuple(cfg.aspect_ratios)),
            "fg_iou": cfg.fg_iou_thresh, "bg_iou": cfg.bg_iou_thresh,
            "alpha": cfg.focal_alpha, "gamma": cfg.focal_gamma}


def data_driven_anchors(cfg, n_images=300):
    ds = VisDroneDet(cfg.data_root, "train", train=False, stride=cfg.train_stride)
    samples = AA.sample_scaled_gt(ds, cfg.imgsz, n_images=n_images, seed=cfg.seed)
    return samples, {"anchor_base": AA.derive_anchor_base(samples), "aspect_ratios": AA.derive_aspect_ratios(samples)}


def anchor_study(cfg, report_dir, n_images=300):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    samples, derived = data_driven_anchors(cfg, n_images)
    sizes = AA.all_gt_sizes(samples)
    print(f"object size at imgsz={cfg.imgsz}: p5 {pd.Series(sizes).quantile(.05):.1f}  median {pd.Series(sizes).median():.1f}"
          f"  p95 {pd.Series(sizes).quantile(.95):.1f} px;  aspect h/w p10/p50/p90 = "
          f"{[round(float(x), 2) for x in pd.Series(AA.all_gt_aspects(samples)).quantile([.1, .5, .9])]}")
    print("derived:", derived)

    default = (retina_anchor_params(RetinaConfig()))
    configs = {"default": default,
               "data sizes": (AA.anchor_sizes_from_base(derived["anchor_base"]), default[1]),
               "data sizes + ratios": (AA.anchor_sizes_from_base(derived["anchor_base"]), derived["aspect_ratios"])}
    results = AA.compare_anchor_configs(samples, configs)

    overview = AA.configs_overview(results, fg=0.5)
    overview.to_csv(report_dir / "anchor_overview.csv")
    print("\n  share of GT whose best anchor has IoU >= 0.5  ")
    print(overview.round(3).to_string())
    for name, res in results.items():
        cov = AA.coverage_summary(res["gt_best"], res["gt_size"])
        asg = AA.assignment_summary(res)
        cov.to_csv(report_dir / f"anchor_coverage_{name.replace(' ', '_').replace('+', 'and')}.csv")
        asg.to_csv(report_dir / f"anchor_assignment_{name.replace(' ', '_').replace('+', 'and')}.csv", index=False)
        print(f"\n  {name}: coverage by object size (input px)  ")
        print(cov.round(3).to_string())
        print(f"  {name}: anchors per image under different IoU thresholds  ")
        print(asg.round(2).to_string(index=False))
        print(f"  {name}: pyramid level of the best anchor  ")
        print(AA.level_usage(res).round(2).to_string())
    finish_figure(AA.plot_sizes_vs_anchors(samples, configs), "anchor_sizes_vs_objects", report_dir, show=True)
    finish_figure(AA.plot_iou_survival(results), "anchor_iou_survival", report_dir, show=True)
    (report_dir / "derived_anchors.json").write_text(json.dumps(derived))
    return derived


def build_stages(derived, default_cfg=None):
    default_cfg = default_cfg or RetinaConfig()
    anchors = {}
    if tuple(derived["anchor_base"]) != tuple(default_cfg.anchor_base):
        anchors["data_sizes"] = {"anchor_base": derived["anchor_base"]}
        if tuple(derived["aspect_ratios"]) != tuple(default_cfg.aspect_ratios):
            anchors["data_sizes_ratios"] = {"anchor_base": derived["anchor_base"],
                                            "aspect_ratios": derived["aspect_ratios"]}
    return [("anchors", anchors), ("iou", IOU_CANDIDATES), ("focal", FOCAL_CANDIDATES)]


def run_tuning(base_cfg, stages, device, force=False):
    def run(name, overrides):
        cfg = apply_overrides(base_cfg, overrides)
        return run_experiment(name, cfg, device, force, build_fn=build_retinanet,
                              describe_fn=describe_retina_setup, params=retina_params(cfg))

    current, rows = {}, []
    incumbent = {**run("initial", {}), "stage": "initial"}
    rows.append(incumbent)
    for stage, candidates in stages:
        pool = [(dict(current), incumbent)]
        for cname, ov in candidates.items():
            total = {**current, **ov}
            row = {**run(f"{stage}_{cname}", total), "stage": stage}
            rows.append(row)
            pool.append((total, row))
        current, incumbent = max(pool, key=lambda t: t[1]["mAP"])
        print(f"\n>>> stage '{stage}': best = {incumbent['experiment']} (mAP {incumbent['mAP']:.3f})\n")

    out = Path(base_cfg.out_dir)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tuning.csv", index=False)
    (out / "tuned_overrides.json").write_text(json.dumps(current, indent=2))
    print(df[[c for c in TABLE_COLS if c in df]].round(3).to_string(index=False))
    print("\ntuned overrides:", current)
    return df, current


def plot_tuning(df):
    d = df.set_index("experiment")[["mAP", "AP50", "AP_small"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    d.plot.bar(ax=ax, rot=35)
    ax.set_ylabel("AP")
    ax.set_title("RetinaNet tuning (short runs, same budget)")
    return fig


def run_final(base_cfg, tuned_overrides, device, force=False):
    rows = {}
    for name, ov in (("initial", {}), ("tuned", tuned_overrides)):
        cfg = apply_overrides(base_cfg, ov)
        rows[name] = run_experiment(name, cfg, device, force, build_fn=build_retinanet,
                                    describe_fn=describe_retina_setup, params=retina_params(cfg))
    return rows


def compare_models(runs):
    rows = {}
    for label, run_dir in runs.items():
        rows[label] = json.loads((Path(run_dir) / "metrics.json").read_text())
    df = pd.DataFrame(rows).T
    per_class = [c for c in df.columns if c.startswith("AP_") and c not in COMPARE_COLS]
    per_class = [c for c in per_class if c not in ("AP_small", "AP_medium", "AP_large")]
    return df[[c for c in COMPARE_COLS if c in df]].astype(float), df[per_class].astype(float)


def plot_comparison(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    df[["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]].T.plot.bar(ax=a1, rot=0)
    a1.set_ylabel("AP")
    a1.set_title("Validation accuracy")
    df["ms_per_img"].plot.bar(ax=a2, rot=20, color="tab:gray")
    a2.set_ylabel("ms / image (batch 1)")
    a2.set_title("Inference latency")
    return fig


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root")
    common.add_argument("--out-dir")
    common.add_argument("--imgsz", type=int)
    common.add_argument("--epochs", type=int)
    common.add_argument("--batch-size", type=int)
    common.add_argument("--train-stride", type=int)
    common.add_argument("--val-stride", type=int)
    common.add_argument("--num-workers", type=int)
    common.add_argument("--set", nargs="*", help="any RetinaConfig field, e.g. focal_gamma=1.5")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("anchors", parents=[common])
    a.add_argument("--report-dir", default="runs/retina_anchors")
    t = sub.add_parser("tune", parents=[common])
    t.add_argument("--force", action="store_true")
    f = sub.add_parser("final", parents=[common])
    f.add_argument("--tuned", default="runs/retina_tune/tuned_overrides.json")
    f.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True, help='"label=run_dir" pairs')
    c.add_argument("--out", default="runs/comparison")
    return ap


def config_from_args(args, default_out):
    cli = {k: getattr(args, k) for k in ("data_root", "out_dir", "imgsz", "epochs", "batch_size", "train_stride",
                                         "val_stride", "num_workers") if getattr(args, k, None) is not None}
    cfg = apply_overrides(RetinaConfig(), {**cli, **parse_set(args.set)})
    return cfg if args.out_dir else replace(cfg, out_dir=default_out)


def main():
    args = build_parser().parse_args()
    if args.cmd == "compare":
        runs = dict(kv.split("=", 1) for kv in args.runs)
        df, per_class = compare_models(runs)
        print(df.round(3).to_string())
        print(per_class.round(3).to_string())
        finish_figure(plot_comparison(df), "model_comparison", args.out, show=False)
        df.to_csv(Path(args.out) / "model_comparison.csv")
        return
    device = get_device()
    if args.cmd == "anchors":
        anchor_study(config_from_args(args, "runs"), args.report_dir)
    elif args.cmd == "tune":
        cfg = config_from_args(args, "runs/retina_tune")
        derived = anchor_study(cfg, Path(cfg.out_dir) / "anchor_study")
        df, _ = run_tuning(cfg, build_stages(derived), device, args.force)
        finish_figure(plot_tuning(df), "tuning", cfg.out_dir, show=False)
    else:
        cfg = config_from_args(args, "runs/retina_final")
        tuned = json.loads(Path(args.tuned).read_text())
        run_final(cfg, tuned, device, args.force)


if __name__ == "__main__":
    main()
