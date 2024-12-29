import numpy as np
import skimage
import torch
from captum.attr import IntegratedGradients

from segmentation.model.analysis.base import AnalysisModel


class CaptumModel(AnalysisModel):
    def _process(self, path, inputs, expected):
        # Predict
        input_tensor = inputs.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        pred_class = output.argmax().item()

        # Skip if not crack
        if pred_class == 0:
            return path, np.zeros(self.out_shape), expected.squeeze().numpy()

        # Integrate
        ig = IntegratedGradients(self.model)
        attr = ig.attribute(input_tensor, target=1)
        heatmap = attr.squeeze()
        expected = expected.squeeze()

        # To mask
        heatmap_np = self._get_mask(heatmap).cpu().numpy()
        expected_np = expected.cpu().numpy()
        heatmap_resized = skimage.transform.resize(
            heatmap_np, self.out_shape, anti_aliasing=False
        )

        return path, heatmap_resized, expected_np

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

    @classmethod
    def _default_postprocessors(cls):
        return [cls._morphology]

    @staticmethod
    def _morphology(path, predicted, expected):
        predicted = skimage.morphology.closing(predicted)
        predicted = skimage.morphology.opening(predicted)
        predicted = skimage.morphology.dilation(predicted)
        predicted = predicted > 0
        return path, predicted, expected
