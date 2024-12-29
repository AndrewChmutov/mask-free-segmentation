from pathlib import Path
from typing import Callable, ClassVar, Iterator, Literal

import numpy as np
import skimage
import torch
from captum.attr import IntegratedGradients
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)
from tqdm import tqdm


class CrackModel:
    def __init__(
        self,
        model,
        criterion,
        optimizer: torch.optim.Optimizer,
        device: DeviceLikeType = "cpu",
    ):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        best_model: bool = True,
        val_acc_threshold: float | None = 0.98,
        stagnation_threshold: int | None = 5
    ):
        train_losses = []
        val_losses = []
        val_accs = []

        best_model_dict = None
        best_loss = float("inf")
        no_improvement = 0

        for epoch in range(epochs):
            #########
            # Train #
            #########

            self.model.train()
            train_loss = 0.0
            for _, inputs, expected in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}: Training"
            ):
                train_loss += self._train_batch(inputs, expected)

            # Evaluate
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train Batch Loss: {train_loss}"
            )

            ##############
            # Validation #
            ##############

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for _, inputs, expected in tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{epochs}: Validation"
                ):
                    (
                        current_loss,
                        current_correct,
                        current_total
                    ) = self._val_batch(inputs, expected)

                    val_loss += current_loss
                    correct += current_correct
                    total += current_total

            val_loss /= len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Epoch {epoch + 1}/{epochs} Val Accuracy: {val_acc:.2f}")
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Validation Batch Loss: {val_loss:.2f}"
            )

            ##########
            # Checks #
            ##########

            # Check best loss
            if best_loss > val_loss:
                no_improvement = 0
                best_loss = val_loss
                best_model_dict = self.model.state_dict()

            # Stagnation
            if stagnation_threshold and no_improvement > stagnation_threshold:
                break

            # Enough accuracy
            if val_acc_threshold and val_acc >= val_acc_threshold:
                break

        if best_model and best_model_dict:
            self.model.load_state_dict(best_model_dict)

    def _train_batch(self, inputs, expected):
        inputs, expected = inputs.to(self.device), expected.to(self.device)

        # Make predictions and perform step
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, expected)
        loss.backward()
        self.optimizer.step()

        # Evaluate
        return loss.item()

    def _val_batch(self, inputs, expected):
        inputs, expected = inputs.to(self.device), expected.to(self.device)

        # Make predictions
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Evaluate
        loss = self.criterion(outputs, expected)
        total = expected.size(0)
        correct = (predicted == expected).sum().item()
        return loss, correct, total

    @torch.no_grad
    def evaluate(self, test_loader: DataLoader):
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, inputs, expected in tqdm(test_loader):
                current_loss, current_correct, current_total = self._val_batch(inputs, expected)

                test_loss += current_loss
                correct += current_correct
                total += current_total

        test_loss /= len(test_loader)
        test_acc = correct / total

        return test_loss, test_acc


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

        # Load weights if needed
        if path is not None and Path(path).is_file():
            model.load_state_dict(torch.load(path, weights_only=True))

        # Criterion and Optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        super().__init__(model, criterion, optimizer, device)


class CaptumModel:
    def __init__(
        self,
        model,
        percentile: float,
        out_shape: tuple[int, int] = (448, 448),
        device: DeviceLikeType = "cpu",
    ) -> None:
        self.model = model
        self.percentile = percentile
        self.postprocessors = None
        self.out_shape = out_shape
        self.device = device

    def __call__(self, dataset: Dataset, use_tqdm: bool = True) -> Iterator:
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1)

        for path, input_tensor, expected in tqdm(dataloader, disable=not use_tqdm):
            # Predict
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            pred_class = output.argmax().item()

            # Skip if not crack
            if pred_class == 0:
                yield self._apply_postprocessor(
                    path, np.zeros(self.out_shape), expected.squeeze().numpy()
                )
                continue

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

            yield self._apply_postprocessor(path, heatmap_resized, expected_np)

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
