"""Net.py - model structure (Chapter 10: 'Net.py defines the structure of the model').

ThaiCharNet
  Backbone  : ImageNet-pretrained torchvision CNN (Transfer Learning, fine-tuned end-to-end)
  Attention : Squeeze-and-Excitation on the last feature map (re-weights channels -> helps the
              network focus on the fine strokes that separate look-alike Thai letters)
  Pooling   : global-average  ++  global-max  (concatenated)
  Neck      : Dropout -> FC(512) -> BatchNorm -> ReLU -> Dropout
  Heads     : (A) 72-way character classifier          <- the task we are graded on
              (B) 5-way glyph-group classifier          <- auxiliary task (Multi-task Learning,
                  consonant / upper-mark / lower-mark / digit / other) sharing the same backbone
"""
import copy

import torch
import torch.nn as nn
import torchvision.models as M

_WEIGHTS = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights", "resnet50": "ResNet50_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights", "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            "convnext_tiny": "ConvNeXt_Tiny_Weights"}
_DIMS = {"efficientnet_b0": 1280, "mobilenet_v3_large": 960, "convnext_tiny": 768}
ARCHS = list(_WEIGHTS)


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(8, c // r)), nn.ReLU(inplace=True),
                                nn.Linear(max(8, c // r), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


def _make_backbone(arch, pretrained, fine_detail):
    if arch not in _WEIGHTS:
        raise ValueError(f"arch must be one of {ARCHS}")
    ctor, weights = getattr(M, arch), None
    if pretrained:
        weights = getattr(M, _WEIGHTS[arch]).DEFAULT
    try:
        m = ctor(weights=weights)
    except Exception as e:  # e.g. no internet to download the ImageNet weights
        print(f"[WARNING] could not load pretrained weights ({e}); using RANDOM init. "
              f"Transfer learning is NOT active!")
        m = ctor(weights=None)
    if arch.startswith("resnet"):
        if fine_detail:                       # keep 2x more spatial detail (slower, sharper strokes)
            m.maxpool = nn.Identity()
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        return body, m.fc.in_features
    if fine_detail and arch in ("efficientnet_b0", "mobilenet_v3_large"):
        m.features[0][0].stride = (1, 1)
    return m.features, _DIMS[arch]


class ThaiCharNet(nn.Module):
    def __init__(self, arch="resnet50", num_classes=72, num_groups=5, pretrained=True, drop=0.3,
                 fine_detail=False, attention=True):
        super().__init__()
        self.backbone, dim = _make_backbone(arch, pretrained, fine_detail)
        self.attn = SEBlock(dim) if attention else nn.Identity()
        self.drop2d = nn.Dropout2d(0.1)                       # drops whole feature maps
        self.pre_drop = nn.Dropout(drop)
        self.neck = nn.Sequential(nn.Linear(dim * 2, 512), nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True), nn.Dropout(drop))
        self.head_cls = nn.Linear(512, num_classes)
        self.head_grp = nn.Linear(512, num_groups)

    def forward(self, x):
        f = self.drop2d(self.attn(self.backbone(x)))
        v = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        h = self.neck(self.pre_drop(v))
        return self.head_cls(h), self.head_grp(h)

    def backbone_params(self):
        return list(self.backbone.parameters())

    def new_params(self):
        ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in ids]


def build_from_config(cfg, pretrained=False):
    return ThaiCharNet(cfg["arch"], cfg["num_classes"], cfg["num_groups"], pretrained,
                       cfg["drop"], cfg["fine_detail"], cfg["attention"])


class EMA:
    """Exponential moving average of the weights - smoother and usually generalises better."""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])
