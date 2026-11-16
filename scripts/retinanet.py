from dataclasses import dataclass

import torch
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.ops import sigmoid_focal_loss

from scripts.anchor_analysis import DEFAULT_SCALES, anchor_sizes_from_base
from scripts.fr_cnn import BaseConfig, N_CLASSES, common_setup_lines


@dataclass(frozen=True)
class RetinaConfig(BaseConfig):
    lr: float = 0.0025
    anchor_base: tuple = (32, 64, 128, 256, 512)
    anchor_scales: tuple = DEFAULT_SCALES
    aspect_ratios: tuple = (0.5, 1.0, 2.0)
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.4
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    topk_candidates: int = 1000


def retina_anchor_params(cfg):
    return anchor_sizes_from_base(cfg.anchor_base, cfg.anchor_scales), tuple(cfg.aspect_ratios)


class FocalRetinaClassificationHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes, alpha=0.25, gamma=2.0):
        super().__init__(in_channels, num_anchors, num_classes)
        self.alpha, self.gamma = alpha, gamma

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        for tgt, logits, matched in zip(targets, head_outputs["cls_logits"], matched_idxs):
            fg = matched >= 0
            target = torch.zeros_like(logits)
            target[fg, tgt["labels"][matched[fg]]] = 1.0
            valid = matched != det_utils.Matcher.BETWEEN_THRESHOLDS      # anchors between bg and fg are ignored
            loss = sigmoid_focal_loss(logits[valid], target[valid], alpha=self.alpha, gamma=self.gamma,
                                      reduction="sum")
            losses.append(loss / max(1, int(fg.sum())))
        return sum(losses) / len(targets)


def build_retinanet(cfg, pretrained=True):
    model = retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None, min_size=cfg.imgsz, max_size=cfg.imgsz,
        score_thresh=cfg.box_score_thresh, nms_thresh=cfg.box_nms_thresh,
        detections_per_img=cfg.box_detections_per_img, topk_candidates=cfg.topk_candidates,
        fg_iou_thresh=cfg.fg_iou_thresh, bg_iou_thresh=cfg.bg_iou_thresh,
        **({"trainable_backbone_layers": cfg.trainable_backbone_layers} if pretrained else {}))

    sizes, ratios = retina_anchor_params(cfg)
    model.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ratios,) * len(sizes))
    n_anchors = len(sizes[0]) * len(ratios)
    ch = model.backbone.out_channels

    old_cls, old_reg = model.head.classification_head, model.head.regression_head
    new_cls = FocalRetinaClassificationHead(ch, n_anchors, N_CLASSES + 1, cfg.focal_alpha, cfg.focal_gamma)
    new_cls.conv.load_state_dict(old_cls.conv.state_dict())
    model.head.classification_head = new_cls
    if old_reg.bbox_reg.out_channels != 4 * n_anchors:
        new_reg = RetinaNetRegressionHead(ch, n_anchors)
        new_reg.conv.load_state_dict(old_reg.conv.state_dict())
        model.head.regression_head = new_reg
    return model


def describe_retina_setup(cfg):
    sizes, ratios = retina_anchor_params(cfg)
    return "\n".join([
        "# Training setup: RetinaNet",
        "- Model: torchvision RetinaNet, ResNet-50-FPN (P3-P7, strides 8-128), initialised from COCO weights "
        f"`RetinaNet_ResNet50_FPN_Weights.COCO_V1`; heads' conv towers kept, final layers re-initialised "
        f"({N_CLASSES + 1} logits per anchor: 8 classes + an unused background column)",
        f"- Anchors: sizes per level {sizes}, aspect ratios (h/w) {ratios}, "
        f"{len(sizes[0]) * len(ratios)} anchors per location",
        f"- Assignment: positive IoU >= {cfg.fg_iou_thresh}, negative IoU < {cfg.bg_iou_thresh}, "
        "ignored in between; each GT also keeps its best anchor",
        f"- Loss: sigmoid focal loss alpha {cfg.focal_alpha}, gamma {cfg.focal_gamma} + L1 box regression",
        f"- Inference: top {cfg.topk_candidates} candidates per level before NMS",
        *common_setup_lines(cfg),
    ]) + "\n"
