import torch
from torch import nn
from RPN import RPN
from Backbones import get_backbone


class Faster_RCNN(nn.Module):
    def __init__(self, backbone_name, anchor_boxes, anchor_ratios):
        super(Faster_RCNN, self).__init__()
        self.backbone_name = backbone_name
        backbone_conv_cls = get_backbone(self.backbone_name)
        self.backbone = backbone_conv_cls()
        feature_dim = backbone_conv_cls.op_chnls
        self.rpn = RPN(feature_dim)


    def forward(self, ip_batch):
        feature_set = self.backbone(ip_batch)
        self.rpn(feature_set)


if __name__ == "__main__":
    f_rcnn = Faster_RCNN("VGG16", [], [])
    t = torch.rand((16, 3, 224, 244))
    f_rcnn(t)
