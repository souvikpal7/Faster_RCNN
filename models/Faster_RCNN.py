import torch
from torch import nn
from RPN import RPN
from Backbones import get_backbone
from roi_pooling import ROIPooling
from prediction_layer import BBoxClassifier, BboxDeltaFC


class Faster_RCNN(nn.Module):
    def __init__(self, backbone_name):
        super(Faster_RCNN, self).__init__()
        self.backbone_name = backbone_name
        backbone_conv_cls = get_backbone(self.backbone_name)
        self.backbone = backbone_conv_cls()
        feature_dim = backbone_conv_cls.op_chnls
        self.rpn = RPN(feature_dim)
        self.roi_pool = ROIPooling(7)
        self.classifier = BBoxClassifier()
        self.bboxdel = BboxDeltaFC()

    def forward(self, ip_batch):
        feature_set = self.backbone(ip_batch)
        bg_fg_pred, proposals, feature_map_projection = self.rpn(feature_set)

        b_size, wid, hei, K, _ = proposals.shape
        proposals = proposals.view(b_size, -1, 4)
        # can't filter as differnt images have different valid boxes
        # proposals = proposals[bg_fg_pred[:, :, 1] > 0.5]
        rois = feature_map_projection.view(b_size, -1, 4)
        pooled_rois = self.roi_pool(feature_set, rois)
        prediected_classes = self.classifier(pooled_rois)
        bbox_del = self.bboxdel(pooled_rois)
        # print(f"prediected_classes: {prediected_classes.shape}")
        # print(f"bbox_del: {bbox_del.shape}")
        predicted_boxes = proposals + bbox_del
        predicted_boxes = torch.clamp(predicted_boxes, 0, 512)
        return bg_fg_pred, prediected_classes, predicted_boxes


if __name__ == "__main__":
    f_rcnn = Faster_RCNN("VGG16")
    ip_tensor = torch.rand((16, 3, 512, 512))
    print(f"Input Tensor Shape: {ip_tensor.shape}")
    bg_fg_pred, prediected_classes, predicted_boxes = f_rcnn(ip_tensor)
    print(f"Background Foreground prediction shape: {bg_fg_pred.shape}")
    print(f"Predicted Class shape: {prediected_classes.shape}")
    print(f"Predicted BBox shape{predicted_boxes.shape}")
