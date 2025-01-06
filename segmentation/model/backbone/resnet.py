from pathlib import Path
from typing import Literal

import torch
from torch._prims_common import DeviceLikeType
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from segmentation.model.backbone.base import CrackModel


class ResnetCrackModel(CrackModel):
    def __init__(
        self,
        version: Literal["18"] | Literal["50"],
        reuse_weights: bool = True,
        device: DeviceLikeType = "cpu",
        path: Path | None = None,
        load_model: bool = False,
    ) -> None:
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

        super().__init__(model, criterion, optimizer, device, path, load_model)
