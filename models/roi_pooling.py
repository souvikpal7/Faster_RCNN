import torch
import torch.nn.functional as F


class ROIPooling2(torch.nn.Module):
    def __init__(self, output_size):
        super(ROIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, rois):
        """
        Args:
        - feature_map (torch.Tensor): Feature map with shape (batch_size, channels, height, width).
        - rois (torch.Tensor): Region of Interest (ROI) bounding boxes with shape (num_rois, 4).
                                Each ROI is represented as (top, left, bottom, right) coordinates.

        Returns:
        - pooled_rois (torch.Tensor): Pooled ROIs with shape (num_rois, channels, output_height, output_width).
        """

        num_rois = rois.size(0)
        pooled_rois = torch.zeros(num_rois, feature_map.size(1), self.output_size, self.output_size, device=feature_map.device)

        for i, roi in enumerate(rois):
            top, left, bottom, right = roi
            # Compute ROI height and width
            roi_height = max(bottom - top, 1)
            roi_width = max(right - left, 1)

            # Compute the height and width of each ROI bin
            bin_height = roi_height / self.output_size
            bin_width = roi_width / self.output_size

            for y in range(self.output_size):
                for x in range(self.output_size):
                    # Compute the coordinates of the bin
                    start_y = int(top + y * bin_height)
                    end_y = int(top + (y + 1) * bin_height)
                    start_x = int(left + x * bin_width)
                    end_x = int(left + (x + 1) * bin_width)

                    # Clip the coordinates to ensure they are within the feature map
                    start_y = min(max(start_y, 0), feature_map.size(2))
                    end_y = min(max(end_y, 0), feature_map.size(2))
                    start_x = min(max(start_x, 0), feature_map.size(3))
                    end_x = min(max(end_x, 0), feature_map.size(3))

                    # Pool the features within the bin using max pooling
                    pooled_roi = F.adaptive_max_pool2d(feature_map[:, :, start_y:end_y, start_x:end_x], (1, 1))

                    pooled_rois[i, :, y, x] = pooled_roi.view(-1)

        return pooled_rois


class ROIPooling(torch.nn.Module):
    def __init__(self, output_size):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.pool_logic = torch.nn.AdaptiveMaxPool2d((self.output_size, self.output_size))

    def forward(self, feature_map, rois):
        b_size, no_chnls, f_wid, f_hei = feature_map.shape
        _, roi_cnt, _ = rois.shape
        print(f"roi_cnt: {roi_cnt}")
        pooled_rois = torch.zeros(b_size, roi_cnt, no_chnls, self.output_size, self.output_size)
        rois = rois.view(-1, 4).to(torch.int32)
        for idx, roi in enumerate(rois):
            x1, y1, x2, y2 = roi
            pooled_feature = self.pool_logic(feature_map[:, :, x1: x2 + 1, y1: y2 + 1])
            print(f"pooled_featre {pooled_feature.shape}")
            pooled_rois[:, idx, :, :, :] = pooled_feature

        pooled_rois.permute(1, 0, 2, 3, 4).contiguous()
        pooled_rois_flatten = pooled_rois.view(b_size, roi_cnt, no_chnls, -1)
        return pooled_rois_flatten


if __name__ == "__main__":
    # Example usage:
    # Define the ROI Pooling layer
    roi_pooling = ROIPooling(output_size=7)

    # Example feature map
    feature_map = torch.randn(2, 64, 100, 100)  # Example feature map

    # Example ROI bounding boxes
    rois = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])  # Example proposal bounding boxes
    rois = rois.unsqueeze(1)
    print(f"main_roi_shape: {rois.shape}")
    # Example forward pass
    pooled_rois = roi_pooling(feature_map, rois)
    print("Pooled ROIs shape:", pooled_rois.shape)
