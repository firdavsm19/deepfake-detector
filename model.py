import torch
import torch.nn as nn
from config import Config
from torchvision import models


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_dim),
            Swish(),
            nn.Linear(reduced_dim, in_channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip          = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        expanded_dim = in_channels * expand_ratio
        reduced_dim  = max(1, int(in_channels * se_ratio))
        padding      = (kernel_size - 1) // 2
        layers       = []

        if expand_ratio != 1:
            layers += [nn.Conv2d(in_channels, expanded_dim, 1, bias=False),
                       nn.BatchNorm2d(expanded_dim, momentum=0.01, eps=1e-3), Swish()]

        layers += [nn.Conv2d(expanded_dim, expanded_dim, kernel_size,
                             stride=stride, padding=padding,
                             groups=expanded_dim, bias=False),
                   nn.BatchNorm2d(expanded_dim, momentum=0.01, eps=1e-3), Swish()]

        layers.append(SqueezeExcitation(expanded_dim, reduced_dim))
        layers += [nn.Conv2d(expanded_dim, out_channels, 1, bias=False),
                   nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)]
        self.block = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        survival  = 1 - self.drop_connect_rate
        mask      = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + survival
        return x * torch.floor(mask) / survival

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = self._drop_connect(out) + x
        return out


class EfficientNetB4Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48, momentum=0.01, eps=1e-3), Swish()
        )
        stage_configs = [
            ( 48,  24, 3, 1, 1, 2),
            ( 24,  32, 3, 2, 6, 4),
            ( 32,  56, 5, 2, 6, 4),
            ( 56, 112, 3, 2, 6, 6),
            (112, 160, 5, 1, 6, 6),
            (160, 272, 5, 2, 6, 8),
            (272, 448, 3, 1, 6, 2),
        ]
        total_blocks = sum(c[5] for c in stage_configs)
        block_idx    = 0
        self.stages  = nn.ModuleList()

        for (in_ch, out_ch, k, s, exp, reps) in stage_configs:
            blocks = []
            for i in range(reps):
                dc = Config.DROP_CONNECT * block_idx / total_blocks
                blocks.append(MBConvBlock(
                    in_ch if i == 0 else out_ch, out_ch, k,
                    stride=s if i == 0 else 1,
                    expand_ratio=exp, drop_connect_rate=dc
                ))
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x   # (B, 448, H, W)


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained EfficientNet-B4 (trained on 1.2M ImageNet images)
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # Replace the final classifier head with our own
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, Config.NUM_CLASSES)
        )

    def forward(self, x):
        return self.backbone(x)