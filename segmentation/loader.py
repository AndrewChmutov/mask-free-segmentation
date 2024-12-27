from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, override

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class CrackDataset(ABC, Dataset):
    def __init__(self, root_dir: str, **kwargs):
        self.root_dir = root_dir
        self.image_paths = []
        self.image_paths = list((Path(root_dir) / "images").iterdir())

    def __len__(self):
        return len(self.image_paths)

    def _get_x(self, path: Path) -> Any:
        return Image.open(path).convert("RGB")

    @abstractmethod
    def _get_y(self, path: Path) -> Any:
        ...

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        return self._get_x(path), self._get_y(path)


class WithTransform(CrackDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "transform" in kwargs
        self.transform = kwargs["transform"]

    @override
    def _get_x(self, path: Path):
        image = super()._get_x(path)
        if self.transform:
            image = self.transform(image)
        return image


class CrackDataset3(WithTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = list(map(self._get_y, self.image_paths))

    @override
    def _get_y(self, path: Path):
        return 0 if path.name.startswith("noncrack") else 1


class CrackDataset4(CrackDataset):
    def get_y(self, path: Path) -> Any:
        path = path.parent / "masks"
        return self._get_x(path)


@dataclass
class DataResult:
    dataset: CrackDataset
    test_dataset: CrackDataset

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _load3(
    transform: Callable | None = None,
    val_size: float = 0.3,
    batch_size: int = 32
) -> DataResult:
    dataset = CrackDataset3(root_dir="data/train", transform=transform)

    indices = np.arange(len(dataset))
    labels = dataset.labels

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = CrackDataset3(root_dir="data/test", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return DataResult(
        dataset=dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def _load4(
    transform: Callable | None = None,
    val_size: float = 0.3,
    batch_size: int = 32
) -> DataResult:
    dataset = CrackDataset3(root_dir="data/train", transform=transform)

    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)
    test_dataset = CrackDataset3(root_dir="data/test", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return DataResult(
        dataset=dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def load(
    version: Literal["3"] | Literal["4"] | Literal["5"],
    **kwargs,
) -> DataResult:
    match version:
        case "3" | "5":
            return _load3(**kwargs)
        case "4":
            return _load4(**kwargs)
