from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AnalysisModel(ABC):
    def __init__(
        self,
        model,
        percentile: float,
        out_shape: tuple[int, int] = (448, 448),
        device: DeviceLikeType = "cpu",
        path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.percentile = percentile
        self.postprocessors = None
        self.out_shape = out_shape
        self.device = device

    @abstractmethod
    def _process(self, path, inputs, expected) -> Any:
        ...

    def __call__(self, dataset: Dataset, use_tqdm: bool = True) -> Iterator:
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1)

        yield from tqdm(
            [
                self._apply_postprocessor(*self._process(*img))
                for img in dataloader
            ],
            disable=not use_tqdm
        )

    def _get_mask(self, heatmap):
        # Map the ranges
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        # Threshold and scale
        heatmap = (
            heatmap > torch.quantile(heatmap, self.percentile)
        ).type(torch.uint8)

        # PIL input format
        heatmap = torch.permute(heatmap, (1, 2, 0))
        heatmap = heatmap.max(2)[0]

        return heatmap

    def _apply_postprocessor(self, path, predicted, expected):
        for postprocessor in self.postprocessors or []:
            path, predicted, expected = postprocessor(path, predicted, expected)
        return path, predicted, expected

    def with_postprocessor(self, postprocessors: list[Callable]):
        self.postprocessors = postprocessors
        return self

    def collect(self, dataset: Dataset, use_tqdm: bool = True):
        return list(self(dataset, use_tqdm))
