import torch
import torch.nn as nn
import torch.nn.functional as F
from anchor_boxes import generate_anchor_boxes_xywh
import config
from mapping1 import Mapping


class RPN(nn.Module):
    def __init__(self, fc_dim, train=True):
        super(RPN, self).__init__()
        self.dim = fc_dim
        self.anchor_scales = config.ANCHOR_SCALES
        self.anchor_ratios = config.ANCHOR_RATIOS
        self.bg_fg_out_cnt = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_cls_score = nn.Conv2d(self.dim, self.bg_fg_out_cnt, 1, 1, 0)
        self.RPN_bbox_pred = nn.Conv2d(self.dim, self.nc_bbox_out, 1, 1, 0)
        self.anchors = generate_anchor_boxes_xywh(config.ANCHOR_SCALES, config.ANCHOR_RATIOS)
        self.batch_size = config.TRAIN_BATCH_SIZE if train else config.INF_BATCH_SIZE
        self.image_shape = config.IMAGE_SIZE
        self.mapping = Mapping(16, 16, 512, 512)

    def forward(self, img_feature):
        batch_size, channels, feat_wid, feat_hei = img_feature.shape
        bg_fg_pred = F.relu(self.RPN_cls_score(img_feature))
        deltas_pred = F.relu(self.RPN_bbox_pred(img_feature))
        print(bg_fg_pred.shape)
        print(deltas_pred.shape)
        print(f"batch_size: {batch_size}")
        deltas_pred = deltas_pred.permute(0, 2, 3, 1).contiguous()
        deltas_pred = deltas_pred.view(
            batch_size,
            feat_wid,
            feat_hei,
            -1,
            4
        )

        bg_fg_pred = bg_fg_pred.permute(0, 2, 3, 1).contiguous()
        bg_fg_pred = bg_fg_pred.view(batch_size, -1)
        proposals = self.__proposal_predictions(deltas_pred)
        print(f"proposal shape: {proposals.shape}")
        print(f"bg_fg_pred shape: {bg_fg_pred.shape}")
        print(f"deltas_pred shape: {deltas_pred.shape}")
        feature_map_projection = self.mapping.proposal_to_featuremap(proposals)
        print(f"feature_map_projection: {feature_map_projection.shape}")
        return proposals, feature_map_projection

    def __proposal_predictions(self, deltas_pred):
        boxes = self.anchors
        deltas = deltas_pred
        print(f"boxes shape: {boxes.shape}")
        print(f"deltas shape: {deltas.shape}")
        pred_boxes = deltas + boxes
        pred_boxes[:, :, :, :, 2] = pred_boxes[:, :, :, :, 0] + pred_boxes[:, :, :, :, 2]
        pred_boxes[:, :, :, :, 3] = pred_boxes[:, :, :, :, 1] + pred_boxes[:, :, :, :, 3]
        clipped_bboxes = self.clip_boxes_batch(pred_boxes)
        return clipped_bboxes

    def clip_boxes_batch(self, boxes):
        boxes[boxes < 0] = 0

        batch_x = self.image_shape[0] - 1
        batch_y = self.image_shape[1] - 1

        boxes[:, :, :, :, 0][boxes[:, :, :, :, 0] > batch_x] = batch_x
        boxes[:, :, :, :, 1][boxes[:, :, :, :, 1] > batch_y] = batch_y
        boxes[:, :, :, :, 2][boxes[:, :, :, :, 2] > batch_x] = batch_x
        boxes[:, :, :, :, 3][boxes[:, :, :, :, 3] > batch_y] = batch_y

        return boxes


if __name__ == "__main__":
    rpn = RPN(512)
    ip_t = torch.rand((10, 512, 16, 16))
    proposal, feature_map = rpn(ip_t)
    print(torch.min(feature_map))
    print(torch.max(feature_map))
