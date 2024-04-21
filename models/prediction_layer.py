import torch
from torch import nn
import config


class BBoxClassifier(nn.Module):
    def __init__(self):
        super(BBoxClassifier, self).__init__()
        self.n_class = config.NUM_CLASS
        model = config.backbone
        if model == "VGG16":
            ip_dim = 512 * 7 * 7
        else:
            raise RuntimeError(f"{model} is not supported")
        self.del_pred = torch.nn.Linear(ip_dim, self.n_class)

    def forward(self, feature_rois):
        b_size, roi_cnt, _, _, _ = feature_rois.shape
        batch_result = torch.zeros(b_size, roi_cnt, self.n_class)
        ip_tensor = feature_rois.view(b_size, roi_cnt, -1)
        for b_idx in range(b_size):
            for roi_idx in range(roi_cnt):
                op_tensor = torch.sigmoid(self.del_pred(ip_tensor[b_idx, roi_idx]))
                batch_result[b_idx, roi_idx] = op_tensor

        return batch_result


class BboxDeltaFC(nn.Module):
    def __init__(self):
        super(BboxDeltaFC, self).__init__()
        model = config.backbone
        if model == "VGG16":
            ip_dim = 512 * 7 * 7
            op_dim = 4
        else:
            raise RuntimeError(f"{model} is not supported")
        self.del_pred = torch.nn.Linear(ip_dim, op_dim)

    def forward(self, feature_rois):
        b_size, roi_cnt, _, _, _ = feature_rois.shape
        batch_result = torch.zeros(b_size, roi_cnt, 4)
        ip_tensor = feature_rois.view(b_size, roi_cnt, -1)
        for b_idx in range(b_size):
            for roi_idx in range(roi_cnt):
                op_tensor = self.del_pred(ip_tensor[b_idx, roi_idx])
                batch_result[b_idx, roi_idx] = op_tensor

        return batch_result


if __name__ == "__main__":
    # bbox_cls = BboxDeltaFC()
    # r = torch.rand(11, 2304, 512, 7, 7)
    # op_ten = bbox_cls(r)
    # print(op_ten.shape)

    bbox_cls = BBoxClassifier()
    r = torch.rand(11, 2304, 512, 7, 7)
    op_ten = bbox_cls(r)
    print(op_ten.shape)
