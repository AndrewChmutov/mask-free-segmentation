import logging

import numpy as np
import skimage
import torch
from torch._prims_common import DeviceLikeType
from torchcam.methods import (
    CAM,
    ISCAM,
    SSCAM,
    GradCAM,
    GradCAMpp,
    LayerCAM,
    ScoreCAM,
    SmoothGradCAMpp,
    XGradCAM,
)

from segmentation.model.analysis.base import AnalysisModel

logger = logging.getLogger("torchcam")
logging.getLogger().setLevel(logging.ERROR)


class TorchCamModel(AnalysisModel):
    VERSIONS = [
        "CAM",
        "GradCAM",
        "GradCAMpp",
        "SmoothGradCAMpp",
        "ScoreCAM",
        "SSCAM",
        "ISCAM",
        "XGradCAM",
        "LayerCAM",
    ]

    def __init__(
        self,
        model,
        percentile: float,
        out_shape: tuple[int, int] = (448, 448),
        device: DeviceLikeType = "cpu",
        version: str = "ScoreCAM",
        layer: str | None = None,
    ) -> None:
        super().__init__(model, percentile, out_shape, device)
        self.version_str = version
        match version:
            case "CAM":
                self.version = CAM
            case "GradCAM":
                self.version = GradCAM
            case "GradCAMpp":
                self.version = GradCAMpp
            case "SmoothGradCAMpp":
                self.version = SmoothGradCAMpp
            case "ScoreCAM":
                self.version = ScoreCAM
            case "SSCAM":
                self.version = SSCAM
            case "ISCAM":
                self.version = ISCAM
            case "XGradCAM":
                self.version = XGradCAM
            case "LayerCAM":
                self.version = LayerCAM
            case _:
                raise NotImplementedError
        self.layer = layer

    def _process(self, path, inputs, expected):
        # Predict
        input_tensor = inputs.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        pred_class = output.argmax().item()

        # Skip if not crack
        if pred_class == 0:
            return path, np.zeros(self.out_shape, dtype=np.uint8), expected.squeeze().numpy()

        # Integrate
        with self.version(self.model, target_layer=self.layer) as cam_extractor:
            # Preprocess your data and feed it to the model
            out = self.model(input_tensor)
            # Retrieve the CAM by passing the class index and the model output
            heatmap = cam_extractor(out.squeeze(0).argmax().item(), out)[0].squeeze().cpu().numpy()

        # DEBUG
        # source_image = Image.open(path[0]).convert("RGB")

        # DEBUG

        heatmap = skimage.transform.resize(
            heatmap, self.out_shape, anti_aliasing=False,
        )

        # To mask
        heatmap = self._get_mask(heatmap)
        expected_np = expected.squeeze().cpu().numpy()

        return path, heatmap, expected_np

    def _get_mask(self, heatmap):
        # Map the ranges
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        # Threshold and scale
        heatmap = (
            heatmap > np.percentile(heatmap, self.percentile) * 100
        ).astype(np.uint8)

        return heatmap

    def name(self):
        return f"{self.__class__.__name__} - {self.version_str}"
