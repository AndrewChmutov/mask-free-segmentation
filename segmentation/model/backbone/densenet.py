from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch._prims_common import DeviceLikeType
from torchvision import transforms
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    densenet121,
    densenet161,
)

from segmentation.model.backbone.base import CrackModel


class DenseNetCrackModel(CrackModel):
    TRANSFORM: ClassVar[transforms.Compose] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def __init__(
        self, version: Literal["121"] | Literal["161"],
        path: Path | None = None,
        reuse_weights: bool = True,
        device: DeviceLikeType = "cpu",
    ):
        match version:
            case "121":
                model_cls = densenet121
                weights = DenseNet121_Weights.DEFAULT
            case "161":
                model_cls = densenet161
                weights = DenseNet161_Weights.DEFAULT,

        # Use pretrained weights
        if not reuse_weights:
            weights = None

        # Create model
        model = model_cls(weights=weights)
        num_ftrs = model.classifier.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

        # Criterion and Optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        super().__init__(model, criterion, optimizer, device, path)
