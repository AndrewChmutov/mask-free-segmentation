from pathlib import Path

import torch
from efficientnet_pytorch import EfficientNet
from torch._prims_common import DeviceLikeType

from segmentation.model.backbone.base import CrackModel


class EfficientNetCrackModel(CrackModel):
    def __init__(
        self, version: str,
        reuse_weights: bool = True,
        device: DeviceLikeType = "cpu",
        path: str | Path | None = None,
        load_model: bool = False,
    ):
        # Use pretrained weights
        if not reuse_weights:
            model = EfficientNet.from_name(version)
        else:
            model = EfficientNet.from_pretrained(version)

        num_ftrs = model._fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

        # Criterion and Optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        super().__init__(model, criterion, optimizer, device, path, load_model)
