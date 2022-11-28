# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .anchor_head import AnchorHead

from mmcv.runner import force_fp32
import torch
from mmdet.core import images_to_levels, multi_apply, bbox_overlaps
import torchsort
import pdb

@HEADS.register_module()
class CorrRetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 corr_w=0.20,
                 use_loss_single = 1,
                 corr_type = 'spearman',
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(CorrRetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.corr_w = corr_w
        self.use_loss_single = use_loss_single
        self.corr_type = corr_type


    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # Estimate CorrLoss coefficient
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        # There must be at least two positives to compute CorrLoss
        if pos_inds.shape[0] > 1:
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            anchors = anchors.reshape(-1, 4)
            pos_anchors = anchors[pos_inds]
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            ious = bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets,
                                 is_aligned=True).clamp(min=1e-6).detach()
            labels_pos = labels[pos_inds]
            cls_score_pos = cls_score[pos_inds, labels_pos]
            cls_score_pos = cls_score_pos.sigmoid()
            if self.corr_type == 'spearman':
                loss_corr = self.spearmanr(cls_score_pos, ious)
            elif self.corr_type == 'concordance':
                loss_corr = self.concordancer(cls_score_pos, ious)
            elif self.corr_type == 'pearson':
                loss_corr = self.pearsonr(cls_score_pos, ious)
            if torch.isnan(loss_corr).any():
                loss_corr = cls_score_pos.sum() * 0
            fpn_level_active = 1
        else:
            loss_corr = cls_score.sum()*0
            fpn_level_active = 0

        return loss_cls, loss_bbox, loss_corr, fpn_level_active

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        if self.use_loss_single:
            losses_cls, losses_bbox, losses_corr, fpn_levels_active = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples)

            # Normalize CorrLoss coefficients based on active fpn levels
            # and using CorrLoss weight
            total_fpn_levels_active = sum(fpn_levels_active)
            if total_fpn_levels_active > 0:
                losses_corr = [self.corr_w * (x / sum(fpn_levels_active)) for x in losses_corr]

        else:
            all_anchors = []
            all_labels = []
            all_label_weights = []
            all_cls_scores = []
            all_bbox_targets = []
            all_bbox_weights = []
            all_bbox_preds = []
            for anc, labels, label_weights, cls_score, bbox_targets, bbox_weights, bbox_pred in zip(all_anchor_list,
                                                                                                    labels_list,
                                                                                                    label_weights_list,
                                                                                                    cls_scores,
                                                                                                    bbox_targets_list,
                                                                                                    bbox_weights_list,
                                                                                                    bbox_preds):
                all_anchors.append(anc.reshape(-1, 4))
                all_labels.append(labels.reshape(-1))
                all_label_weights.append(label_weights.reshape(-1))
                all_cls_scores.append(cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))
                all_bbox_targets.append(bbox_targets.reshape(-1, 4))
                all_bbox_weights.append(bbox_weights.reshape(-1, 4))
                all_bbox_preds.append(bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4))
            cls_labels = torch.cat(all_labels)
            cls_score = torch.cat(all_cls_scores)
            label_weights = torch.cat(all_label_weights)
            anchors = torch.cat(all_anchors)
            bbox_preds = torch.cat(all_bbox_preds)
            bbox_targets = torch.cat(all_bbox_targets)
            bbox_weights = torch.cat(all_bbox_weights)
            losses_cls = self.loss_cls(
                cls_score, cls_labels, label_weights, avg_factor=num_total_samples)

            losses_bbox = self.loss_bbox(
                bbox_preds,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

            bg_class_ind = self.num_classes
            pos_inds = ((cls_labels >= 0)
                        & (cls_labels < bg_class_ind)).nonzero().squeeze(1)
            pos_bbox_pred = bbox_preds[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            ious = bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets, is_aligned=True).clamp(min=1e-6).detach()
            labels_pos = cls_labels[pos_inds]
            cls_score_pos = cls_score[pos_inds, labels_pos]
            cls_score_pos = cls_score_pos.sigmoid()

            if self.corr_type == 'spearman':
                loss_corr = self.spearmanr(cls_score_pos, ious, )
            elif self.corr_type == 'concordance':
                loss_corr = self.concordancer(cls_score_pos, ious)
            elif self.corr_type == 'pearson':
                loss_corr = self.pearsonr(cls_score_pos, ious)
            if torch.isnan(loss_corr).any():
                loss_corr = cls_score_pos.sum() * 0
            losses_corr = loss_corr * self.corr_w

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_corr=losses_corr)

    def concordancer(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        x_var = torch.var(x)
        y_var = torch.var(y)
        x_std = torch.std(x)
        y_std = torch.std(y)
        vx = x - x_mean
        vy = y - y_mean
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        ccc = (2 * pcc * x_std * y_std) / (x_var + y_var + (x_mean - y_mean) ** 2)
        return 1 - ccc

    def pearsonr(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        vx = x - x_mean
        vy = y - y_mean
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - pcc

    def spearmanr(self, pred_, target_, regularization_strength=1.0):
        pred = torchsort.soft_rank(pred_.unsqueeze(dim=0), regularization_strength=regularization_strength)
        target = torchsort.soft_rank(target_.unsqueeze(dim=0), regularization_strength=regularization_strength)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return 1 - (pred * target).sum()