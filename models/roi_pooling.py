import torch


class ROIPooling(torch.nn.Module):
    def __init__(self, output_size):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.pool_logic = torch.nn.AdaptiveMaxPool2d((self.output_size, self.output_size))

    def forward(self, feature_map, rois):
        b_size, no_chnls, f_wid, f_hei = feature_map.shape
        # feature_map = feature_map.view(-1, f_wid, f_hei)
        _, roi_cnt, _ = rois.shape
        print(f"roi_cnt: {roi_cnt}")
        pooled_rois = torch.zeros(b_size, roi_cnt, no_chnls, self.output_size, self.output_size)

        for b_idx in range(b_size):
            for roi_idx in range(roi_cnt):
                x1, y1, x2, y2 = rois[b_idx, roi_idx, :]
                x1 = x1.to(torch.int)
                y1 = y1.to(torch.int)
                x2 = x2.to(torch.int)
                y2 = y2.to(torch.int)
                pooled_feature = self.pool_logic(feature_map[b_idx, :, x1: x2 + 1, y1: y2 + 1])
                # print(f"pooled_featre {pooled_feature.shape}")
                pooled_rois[b_idx, roi_idx] = pooled_feature

        return pooled_rois


if __name__ == "__main__":
    # Example usage:
    # Define the ROI Pooling layer
    roi_pooling = ROIPooling(output_size=7)

    # Example feature map
    feature_map = torch.randn(2, 64, 100, 100)  # Example feature map

    # Example ROI bounding boxes
    rois = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        # [60, 70, 80, 90],
        # [70, 80, 90, 99]
    ])  # Example proposal bounding boxes
    rois = rois.unsqueeze(1)
    print(f"main_roi_shape: {rois.shape}")
    # Example forward pass
    pooled_rois = roi_pooling(feature_map, rois)
    print("Pooled ROIs shape:", pooled_rois.shape)
