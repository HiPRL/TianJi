import torch
import torch.nn as nn
from drl.base.model import Model
from drl.builder import MODELS
from drl.models.utils import *


@MODELS.register_module()
class AlexNet(Model):
    """AlexNet backbone.

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(
        self, num_classes: int = -1, init_method: str = "kaiming_init", **kwargs
    ):
        super().__init__(init_method=init_method, **kwargs)
        self.num_classes = num_classes

        self.backbone_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone_features(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)

        return x
