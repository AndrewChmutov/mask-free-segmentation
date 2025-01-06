from pathlib import Path
from typing import Literal

import torch
from torch._prims_common import DeviceLikeType
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    densenet121,
    densenet161,
)

from segmentation.model.backbone.base import CrackModel


class DenseNetCrackModel(CrackModel):
    def __init__(
        self,
        version: Literal["121"] | Literal["161"],
        reuse_weights: bool = True,
        device: DeviceLikeType = "cpu",
        path: Path | None = None,
        load_model: bool = False,
    ) -> None:
        match version:
            case "121":
                model_cls = densenet121
                weights = DenseNet121_Weights.DEFAULT
            case "161":
                model_cls = densenet161
                weights = (DenseNet161_Weights.DEFAULT,)

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

        super().__init__(model, criterion, optimizer, device, path, load_model)
