import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.eda import finish_figure
from scripts.fr_cnn import (Config, _tuplify, build_val, fit, full_evaluation, get_device, load_config,
                       load_trained, set_test_proposals, set_test_size)

EXPERIMENTS = {
    "baseline": {},
    "rpn_300": {"rpn_proposals": 300},
    "rpn_2000": {"rpn_proposals": 2000},
    "pos_0.10": {"box_positive_fraction": 0.10},
    "pos_0.50": {"box_positive_fraction": 0.50},
}
RUN_DONE_HOOKS = []
TABLE_COLS = ["experiment", "rpn_proposals", "box_positive_fraction", "mAP", "AP50", "AP_small", "AP_medium",
              "AP_large", "ms_per_img", "fps", "train_min"]


def apply_overrides(cfg, overrides):
    return replace(cfg, **{k: _tuplify(v) for k, v in overrides.items()})


def parse_set(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def config_from_args(args):
    cli = {k: getattr(args, k) for k in ("data_root", "out_dir", "arch", "imgsz", "epochs", "batch_size",
                                         "train_stride", "val_stride", "num_workers")
           if getattr(args, k, None) is not None}
    return apply_overrides(Config(), {**cli, **parse_set(args.set)})


def run_experiment(name, cfg, device, force=False, build_fn=None, describe_fn=None, params=None):
    run_dir = Path(cfg.out_dir) / name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"[{name}] found {metrics_path}, skipping (use --force to retrain)")
        return json.loads(metrics_path.read_text())

    model, train_min = fit(cfg, name, device, resume=not force, build_fn=build_fn, describe_fn=describe_fn)
    val_ds, coco_gt = build_val(cfg)
    metrics, _ = full_evaluation(model, cfg, device, val_ds, coco_gt)
    if params is None:
        params = {"rpn_proposals": cfg.rpn_proposals, "box_positive_fraction": cfg.box_positive_fraction}
    row = {"experiment": name, **params, "epochs": cfg.epochs, "train_stride": cfg.train_stride,
           "train_min": train_min, **metrics}
    metrics_path.write_text(json.dumps(row, indent=2))
    for hook in RUN_DONE_HOOKS:
        hook(name, cfg)
    print(f"[{name}] mAP {row['mAP']:.3f}  AP50 {row['AP50']:.3f}  AP_small {row['AP_small']:.3f}  "
          f"{row['ms_per_img']:.0f} ms/img")
    return row


def run_ablation(base_cfg, device, only=None, force=False):
    rows = []
    for name, overrides in EXPERIMENTS.items():
        if only and name not in only:
            continue
        cfg = apply_overrides(base_cfg, overrides)
        rows.append(run_experiment(name, cfg, device, force))
    df = pd.DataFrame(rows)
    out = Path(base_cfg.out_dir)
    df.to_csv(out / "ablation.csv", index=False)
    print("\n" + df[[c for c in TABLE_COLS if c in df]].round(3).to_string(index=False))
    return df


def sweep_test_proposals(run_dir, values=(100, 300, 1000, 2000), device=None):
    run_dir = Path(run_dir)
    device = device or get_device()
    cfg = load_config(run_dir)
    model = load_trained(cfg, run_dir / "last.pt", device)
    val_ds, coco_gt = build_val(cfg)
    rows = []
    for n in values:
        set_test_proposals(model, n)
        metrics, _ = full_evaluation(model, cfg, device, val_ds, coco_gt)
        rows.append({"test_proposals": n, **{k: metrics[k] for k in
                                             ("mAP", "AP50", "AP_small", "AR500", "ms_per_img", "fps")}})
        print(f"test proposals {n}: mAP {metrics['mAP']:.3f}  AP_small {metrics['AP_small']:.3f}  "
              f"{metrics['ms_per_img']:.0f} ms/img")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "test_proposal_sweep.csv", index=False)
    return df


def _load_run(run_dir, kind, device):
    run_dir = Path(run_dir)
    if kind == "retina":
        from scripts.retinanet import RetinaConfig, build_retinanet
        cfg = load_config(run_dir, RetinaConfig)
        return load_trained(cfg, run_dir / "last.pt", device, build_fn=build_retinanet), cfg
    cfg = load_config(run_dir, Config)
    return load_trained(cfg, run_dir / "last.pt", device), cfg


def reevaluate_run(run_dir, kind="faster", device=None):
    run_dir, device = Path(run_dir), device or get_device()
    model, cfg = _load_run(run_dir, kind, device)
    val_ds, coco_gt = build_val(cfg)
    metrics, _ = full_evaluation(model, cfg, device, val_ds, coco_gt)
    path = run_dir / "metrics.json"
    row = json.loads(path.read_text()) if path.exists() else {"experiment": run_dir.name}
    row.update(metrics)
    path.write_text(json.dumps(row, indent=2))
    print(f"[{run_dir.name}] mAP {row['mAP']:.3f}  AP50 {row['AP50']:.3f}  AP_small {row['AP_small']:.3f}")
    return row


def sweep_test_sizes(run_dir, sizes=(1280, 1536, 1920), kind="faster", device=None):
    run_dir, device = Path(run_dir), device or get_device()
    model, cfg = _load_run(run_dir, kind, device)
    val_ds, coco_gt = build_val(cfg)
    rows = []
    for size in sizes:
        set_test_size(model, size)
        metrics, _ = full_evaluation(model, cfg, device, val_ds, coco_gt)
        rows.append({"test_imgsz": size, **{k: metrics[k] for k in ("mAP", "AP50", "AP_small", "AR500", "ms_per_img", "fps")}})
        print(f"test size {size}: mAP {metrics['mAP']:.3f}  AP50 {metrics['AP50']:.3f}  "
              f"AP_small {metrics['AP_small']:.3f}  {metrics['ms_per_img']:.0f} ms/img")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "test_size_sweep.csv", index=False)
    return df


def plot_ablation(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    df.set_index("experiment")[["mAP", "AP50", "AP_small"]].plot.bar(ax=a1, rot=30)
    a1.set_ylabel("AP")
    a1.set_title("Validation accuracy per experiment")
    df.set_index("experiment")["ms_per_img"].plot.bar(ax=a2, rot=30, color="tab:gray")
    a2.set_ylabel("ms / image (batch 1)")
    a2.set_title("Inference latency")
    return fig


def plot_loss_curves(run_dirs):
    logs = {Path(r).name: pd.read_csv(Path(r) / "train_log.csv") for r in run_dirs}
    terms = [c for c in next(iter(logs.values())).columns if c not in ("iter", "epoch", "lr", "loss")]
    n_cols = 2 if len(terms) > 1 else 1
    n_rows = -(-len(terms) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), sharex=True, squeeze=False)
    for ax, term in zip(axes.flat, terms):
        for name, log in logs.items():
            ax.plot(log["iter"], log[term], label=name)
        ax.set_title(term)
        ax.set_xlabel("iteration")
    axes.flat[0].legend(fontsize=8)
    return fig


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root")
    common.add_argument("--out-dir")
    common.add_argument("--arch", choices=["v1", "v2"])
    common.add_argument("--imgsz", type=int)
    common.add_argument("--epochs", type=int)
    common.add_argument("--batch-size", type=int)
    common.add_argument("--train-stride", type=int)
    common.add_argument("--val-stride", type=int)
    common.add_argument("--num-workers", type=int)
    common.add_argument("--set", nargs="*", help="any Config field, e.g. box_positive_fraction=0.5")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ablation", parents=[common])
    a.add_argument("--only", nargs="*", help=f"subset of {list(EXPERIMENTS)}")
    a.add_argument("--force", action="store_true")
    s = sub.add_parser("sweep")
    s.add_argument("--run", required=True)
    s.add_argument("--values", nargs="*", type=int, default=[100, 300, 1000, 2000])
    f = sub.add_parser("final", parents=[common])
    f.add_argument("--name", default="final")
    f.add_argument("--force", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    device = get_device()
    if args.cmd == "ablation":
        cfg = config_from_args(args)
        if args.out_dir is None:
            cfg = replace(cfg, out_dir="runs/ablation")
        df = run_ablation(cfg, device, args.only, args.force)
        out = Path(cfg.out_dir)
        finish_figure(plot_ablation(df), "ablation", out, show=False)
        finish_figure(plot_loss_curves([out / n for n in df["experiment"]]), "loss_curves", out, show=False)
    elif args.cmd == "sweep":
        df = sweep_test_proposals(args.run, args.values, device)
        print(df.round(3).to_string(index=False))
    else:
        cfg = config_from_args(args)
        if args.out_dir is None:
            cfg = replace(cfg, out_dir="runs/final")
        run_experiment(args.name, cfg, device, args.force)


if __name__ == "__main__":
    main()
