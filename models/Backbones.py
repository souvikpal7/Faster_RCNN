import torch
from torchvision.models import vgg16, VGG16_Weights

supported_backbones = ["vgg16"]


def get_backbone(backbone_name):
    assert backbone_name.lower() in supported_backbones, f"{backbone_name} not supported"
    if backbone_name.lower() == "vgg16":
        return VGG16FeatureExt


class VGG16FeatureExt(torch.nn.Module):
    op_chnls = 512

    def __init__(self):
        super(VGG16FeatureExt, self).__init__()
        vgg16_clf = vgg16(weights=VGG16_Weights.DEFAULT)
        # excluding last pooling layer as per paper
        excluding_last_layer = torch.nn.Sequential(*list(vgg16_clf.features.children()))
        self.feature_extractor = excluding_last_layer

    def forward(self, img_batch):
        return self.feature_extractor(img_batch)

    def set_trainable_layers(self, n: int):
        n_layers = len(list(self.feature_extractor.parameters()))
        n_freeze_layers = n_layers - n
        if n_freeze_layers <= 0:
            self.feature_extractor.requires_grad_(True)
        else:
            for lidx, param in enumerate(self.feature_extractor.parameters()):
                if lidx <= n_freeze_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


if __name__ == "__main__":
    f_ext = VGG16FeatureExt()
    t = torch.rand((16, 3, 512, 512))
    print(t.shape)

    t_res = f_ext(t)
    print(t_res.shape)
