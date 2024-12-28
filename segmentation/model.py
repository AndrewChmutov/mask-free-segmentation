from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from captum.attr import IntegratedGradients
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)
from tqdm import tqdm


class CrackModel:
    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, criterion, optimizer: torch.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = self.__class__.get_device()

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
            for inputs, expected in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}: Training"
            ):
                train_loss += self._train_batch(inputs, expected)

            # Evaluate
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Average Batch Loss: {train_loss}"
            )

            ##############
            # Validation #
            ##############

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, expected in tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{epochs}: Validation"
                ):
                    current_loss, current_correct, current_total = self._val_batch(inputs, expected)

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
                f"Average Batch Loss: {val_loss:.2f}"
            )

            ##########
            # Checks #
            ##########

            # Check best loss
            if best_loss > val_loss:
                no_improvement = 0
                best_loss = val_loss
                best_model_dict = self.model.state_dict()
            else:
                no_improvement += 1

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
            for inputs, expected in tqdm(test_loader):
                current_loss, current_correct, current_total = self._val_batch(inputs, expected)

                test_loss += current_loss
                correct += current_correct
                total += current_total

        test_loss /= len(test_loader)
        test_acc = correct / total

        return test_loss, test_acc


class ResnetCrackModel(CrackModel):
    def __init__(
        self, version: Literal["18"] | Literal["50"],
        path: Path | None = None,
        reuse_weights: bool = True,
    ):
        match version:
            case "18":
                model_cls = resnet18
                weights = ResNet18_Weights.DEFAULT
            case "50":
                model_cls = resnet50
                weights = ResNet50_Weights.DEFAULT

        if not reuse_weights:
            weights = None

        model = model_cls(weights=weights)

        device = self.__class__.get_device()

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

        if path is not None and Path(path).is_file():
            model.load_state_dict(torch.load(path, weights_only=True))

        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        super().__init__(model, criterion, optimizer)


class CaptumModel:
    def __init__(self, model, percentile: float) -> None:
        self.model = model
        self.percentile = percentile
        self.ig = IntegratedGradients(model)

    def __call__(self, loader: DataLoader) -> Iterator:
        self.model.eval()
        device = CrackModel.get_device()

        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)
            # with torch.no_grad():
            #     outputs = self.model(inputs)
            # pred_classes = outputs.argmax()

            print("inputs shape", inputs.shape)
            classes = torch.ones(inputs.shape[0]).to(device)
            print("classes of interest shape (all ones)", classes.shape)
            attr = self.ig.attribute(inputs, target=1)  # Annotate Crack
            heatmaps = attr.squeeze().cpu().detach().numpy()
            print("heatmaps shape", heatmaps.shape)

            # Map the ranges
            heatmaps_mins = heatmaps.min(0)
            heatmaps_maxs = heatmaps.max(0)
            heatmaps = (heatmaps - heatmaps_mins) / (heatmaps_maxs - heatmaps_mins)
            print("heatmaps transformed shape", heatmaps.shape)

            bitmap = (heatmaps > np.percentile(heatmaps, 99.5, axis=0)).astype(np.uint8)
            print("bitmaps shape", bitmap.shape)
            bitmap = (resize(bitmap, (448, 448), anti_aliasing=False, order=0) > 0.5).astype(np.uint8)
            print("bitmaps resized shape", bitmap.shape)
            yield bitmap
