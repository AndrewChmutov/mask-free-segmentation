from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch._prims_common import DeviceLikeType
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from segmentation.model.backbone.base import CrackModel


class ResnetCrackModel(CrackModel):
    TRANSFORM: ClassVar[transforms.Compose] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def __init__(
        self, version: Literal["18"] | Literal["50"],
        path: Path | None = None,
        reuse_weights: bool = True,
        device: DeviceLikeType = "cpu",
    ):
        match version:
            case "18":
                model_cls = resnet18
                weights = ResNet18_Weights.DEFAULT
            case "50":
                model_cls = resnet50
                weights = ResNet50_Weights.DEFAULT

        # Use pretrained weights
        if not reuse_weights:
            weights = None

        # Create model
        model = model_cls(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

        # Criterion and Optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        super().__init__(model, criterion, optimizer, device)
