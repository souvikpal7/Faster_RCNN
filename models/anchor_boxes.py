import torch
import numpy as np

import config


def generate_anchor_boxes(scales=[2, 1, 0.5], ratios=[2, 1, 0.5]):
    if config.backbone == "VGG16":
        d_width = 211
        d_hei = 211
        feature_size = 16
        feature_stride = 32
        anchor_cnt = len(scales) * len(ratios)
        anchor_bboxes = []

        anchors = []
        for scale in scales:
            for ratio in ratios:
                eff_wid = scale * d_width
                eff_hei = scale * d_hei
                height_factor = 1 / ratio
                eff_hei /= height_factor
                bbox = [- eff_wid / 2, - eff_hei / 2, eff_wid / 2, eff_hei / 2]
                anchors.append(bbox)

        anchors = np.array(anchors)
        for x_idx in range(feature_size):
            x_vals = np.ones((anchor_cnt, 1)) * x_idx
            for y_idx in range(feature_size):
                y_vals = np.ones((anchor_cnt, 1)) * y_idx
                shifts = np.concatenate((x_vals, y_vals, x_vals, y_vals), axis=1)
                shifts = shifts * feature_stride
                pos_anchors = anchors + shifts
                anchor_bboxes.append(pos_anchors)

        op_shape = (feature_size, feature_size, anchor_cnt , 4)
        anchor_bboxes = np.array(anchor_bboxes).reshape(op_shape)
        anchor_bboxes = torch.from_numpy(anchor_bboxes)
    else:
        raise RuntimeError(f"{config.backbone} not supported")

    return anchor_bboxes


def generate_anchor_boxes_xywh(scales=[2, 1, 0.5], ratios=[2, 1, 0.5]):
    if config.backbone == "VGG16":
        d_width = 211
        d_hei = 211
        feature_size = 16
        feature_stride = 32
        anchor_cnt = len(scales) * len(ratios)
        anchor_bboxes = [[0 for x_idx in range(feature_size)] for y_idx in range(feature_size)]

        for x_idx in range(feature_size):
            for y_idx in range(feature_size):
                anchors = []
                for scale in scales:
                    for ratio in ratios:
                        eff_wid = scale * d_width
                        eff_hei = scale * d_hei
                        height_factor = 1 / ratio
                        eff_hei /= height_factor
                        bbox = [
                            x_idx * feature_stride,
                            y_idx * feature_stride,
                            eff_wid,
                            eff_hei
                        ]
                        anchors.append(bbox)

                anchor_bboxes[x_idx][y_idx] = anchors

        anchor_bboxes = np.array(anchor_bboxes)
        anchor_bboxes = torch.from_numpy(anchor_bboxes)
    else:
        raise RuntimeError(f"{config.backbone} not supported")

    return anchor_bboxes


if __name__ == "__main__":
    anc_bboxs = generate_anchor_boxes_xywh()
    print(anc_bboxs.shape)
    print(anc_bboxs[0][0])
    print(anc_bboxs[3][4])
