import torch
from torch import nn
from RPN import RPN
from Backbones import get_backbone
from roi_pooling import ROIPooling


class Faster_RCNN(nn.Module):
    def __init__(self, backbone_name):
        super(Faster_RCNN, self).__init__()
        self.backbone_name = backbone_name
        backbone_conv_cls = get_backbone(self.backbone_name)
        self.backbone = backbone_conv_cls()
        feature_dim = backbone_conv_cls.op_chnls
        self.rpn = RPN(feature_dim)
        self.roi_pool = ROIPooling(7)

    def forward(self, ip_batch):
        feature_set = self.backbone(ip_batch)
        proposals, feature_map_projection = self.rpn(feature_set)
        b_size, wid, hei, K, _ = proposals.shape
        rois = feature_map_projection.view(b_size, -1, 4)
        pooled_rois = self.roi_pool(feature_set, rois)
        return pooled_rois


if __name__ == "__main__":
    f_rcnn = Faster_RCNN("VGG16")
    pooled_rois = f_rcnn(torch.rand((21, 3, 512, 512)))
    print(pooled_rois.shape)
    # print(feature_maps.shape)
