import numpy as np
from models import config


class Mapping:
    def __init__(self, feature_wid, feature_hei, img_wid, img_hei):
        self.feature_wid = feature_wid
        self.feature_hei = feature_hei
        self.img_wid = img_wid
        self.img_hei = img_hei
        if config.backbone.lower() == "vgg16":
            pass
        else:
            raise RuntimeError(f"{config.backbone} is not supported")

    def imagebbox_to_feature_vgg16(self, p1_x, p1_y, p2_x, p2_y):
        p1_4_x = np.floor((p1_x - 2) / 2)
        p2_4_x = np.floor((p2_x + 2) / 2)
        p1_4_y = np.floor((p1_y - 2) / 2)
        p2_4_y = np.floor((p2_y + 2) / 2)
        print(f"P4 ({p1_4_x}, {p1_4_y}) , ({p2_4_x}, {p2_4_y})")

        p1_9_x = np.floor((p1_4_x - 2) / 2)
        p2_9_x = np.floor((p2_4_x + 2) / 2)
        p1_9_y = np.floor((p1_4_y - 2) / 2)
        p2_9_y = np.floor((p2_4_y + 2) / 2)
        print(f"P9 ({p1_9_x}, {p1_9_y}) , ({p2_9_x}, {p2_9_y})")

        p1_16_x = np.floor((p1_9_x - 3) / 2)
        p2_16_x = np.floor((p2_9_x + 3) / 2)
        p1_16_y = np.floor((p1_9_y - 3) / 2)
        p2_16_y = np.floor((p2_9_y + 3) / 2)
        print(f"P16 ({p1_16_x}, {p1_16_y}) , ({p2_16_x}, {p2_16_y})")

        p1_23_x = np.floor((p1_16_x - 3) / 2)
        p2_23_x = np.floor((p2_16_x + 3) / 2)
        p1_23_y = np.floor((p1_16_y - 3) / 2)
        p2_23_y = np.floor((p2_16_y + 3) / 2)
        print(f"P23 ({p1_23_x}, {p1_23_y}) , ({p2_23_x}, {p2_23_y})")

        p1_30_x = np.floor((p1_23_x - 3) / 2)
        p2_30_x = np.floor((p2_23_x + 3) / 2)
        p1_30_y = np.floor((p1_23_y - 3) / 2)
        p2_30_y = np.floor((p2_23_y + 3) / 2)
        print(f"P30 ({p1_30_x}, {p1_30_y}) , ({p2_30_x}, {p2_30_y})")

        p1_30_x = np.clip(p1_30_x, 0, self.feature_wid - 1)
        p2_30_x = np.clip(p2_30_x, 0, self.feature_hei - 1)

        p1_30_y = np.clip(p1_30_y, 0, self.feature_wid - 1)
        p2_30_y = np.clip(p2_30_y, 0, self.feature_hei - 1)

        return p1_30_x, p1_30_y, p2_30_x, p2_30_y

    def featurebbox_to_image_vff16(self, p1_x, p1_y, p2_x, p2_y):
        # maxpool 2 x 2
        p1_30_x = 2 * p1_x
        p1_30_y = 2 * p1_y

        p2_30_x = 2 * p2_x + 1
        p2_30_y = 2 * p2_y + 1

        # 3 conv of 3 x 3
        p1_24_x = p1_30_x - 3
        p1_24_y = p1_30_y - 3

        p2_24_x = p2_30_x + 3
        p2_24_y = p2_30_y + 3

        # maaxpool 2 x 2
        p1_23_x = 2 * p1_24_x
        p1_23_y = 2 * p1_24_y

        p2_23_x = 2 * p2_24_x + 1
        p2_23_y = 2 * p2_24_y + 1

        # 3 conv of 3 x 3
        p1_17_x = p1_23_x - 3
        p1_17_y = p1_23_y - 3

        p2_17_x = p2_23_x + 3
        p2_17_y = p2_23_y + 3

        # maxpool 2 x 2
        p1_16_x = 2 * p1_17_x
        p1_16_y = 2 * p1_17_y

        p2_16_x = 2 * p2_17_x + 1
        p2_16_y = 2 * p2_17_y + 1

        # 3 conv of 3 x 3
        p1_10_x = p1_16_x - 3
        p1_10_y = p1_16_y - 3

        p2_10_x = p2_16_x + 3
        p2_10_y = p2_16_y + 3

        # maxpool 2 x 2
        p1_9_x = 2 * p1_10_x
        p1_9_y = 2 * p1_10_y

        p2_9_x = 2 * p2_10_x + 1
        p2_9_y = 2 * p2_10_y + 1

        # 2 conv of 3 x 3
        p1_5_x = p1_9_x - 2
        p1_5_y = p1_9_y - 2

        p2_5_x = p2_9_x + 2
        p2_5_y = p2_9_y + 2

        # maxpool 2 x 2
        p1_4_x = 2 * p1_5_x
        p1_4_y = 2 * p1_5_y

        p2_4_x = 2 * p2_5_x + 1
        p2_4_y = 2 * p2_5_y + 1

        # 2 conv of 3 x 3
        p1_0_x = p1_4_x - 2
        p1_0_y = p1_4_y - 2

        p2_0_x = p2_4_x + 2
        p2_0_y = p2_4_y + 2

        p1_0_x = np.clip(p1_0_x, 0, self.img_wid - 1)
        p1_0_y = np.clip(p1_0_y, 0, self.img_hei - 1)

        p2_0_x = np.clip(p2_0_x, 0, self.img_wid - 1)
        p2_0_y = np.clip(p2_0_y, 0, self.img_hei - 1)

        return p1_0_x, p1_0_y, p2_0_x, p2_0_y


if __name__ == "__main__":
    map_obj = Mapping(16, 16, 512, 512)
    p1 = np.array([0, 256])
    p2 = np.array([128, 320])
    # (array([0., 5.]), array([ 6., 12.]))
    ret_points = map_obj.imagebbox_to_feature_vgg16(0, 70, 313, 505)
    print(ret_points)
    # ret_points = map_obj.featurebbox_to_image_vff16(0, 5, 6, 12)
    ret_points = map_obj.featurebbox_to_image_vff16(13, 13, 14, 14)
    print(ret_points)
