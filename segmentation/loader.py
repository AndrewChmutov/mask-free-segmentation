from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class CrackDataset(ABC, Dataset):
    def __init__(self, root_dir: str | Path, **kwargs):
        self.root_dir = root_dir
        self.image_paths = list((Path(root_dir) / "images").iterdir())

    def __len__(self):
        return len(self.image_paths)

    def _transform(self, path: Path):
        return Image.open(path).convert("RGB")

    def _get_x(self, i: int) -> Any:
        return self._transform(self.image_paths[i])

    @abstractmethod
    def _get_y(self, i: int) -> Any:
        ...

    def __getitem__(self, idx):
        return self._get_x(idx), self._get_y(idx)


class WithTransform(CrackDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "transform" in kwargs
        self.transform = kwargs["transform"]

    def _transform(self, path):
        image = super()._transform(path)
        if self.transform:
            image = self.transform(image)
        return image


class CrackDataset3(WithTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pictures = list(map(super()._get_x, range(len(self.image_paths))))
        self.labels = list(map(self._get_y, range(len(self.image_paths))))

    def _get_x(self, i: int):
        return self.pictures[i]

    def _get_y(self, i: int):
        return 0 if self.image_paths[i].name.startswith("noncrack") else 1


class CrackDataset4(CrackDataset):
    def get_y(self, i: int) -> Any:
        path = self.image_paths[i].parent / "masks"
        return self._transform(path)


@dataclass
class DataResult:
    dataset: CrackDataset
    test_dataset: CrackDataset
    train_dataset: Subset
    val_dataset: Subset

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _load3(
    transform: Callable | None = None,
    val_size: float = 0.3,
    batch_size: int = 32,
    random_state: int = 17
) -> DataResult:
    dataset = CrackDataset3(root_dir="../project2/data/train", transform=transform)

    indices = np.arange(len(dataset))
    labels = dataset.labels

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=random_state
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = CrackDataset3(root_dir="../project2/data/test", transform=transform)

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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def _load4(
    transform: Callable | None = None,
    val_size: float = 0.3,
    batch_size: int = 32,
    random_state: int = 17,
) -> DataResult:
    dataset = CrackDataset3(root_dir="../project2/data/train", transform=transform)

    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_state
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)
    test_dataset = CrackDataset3(root_dir="../project2/data/test", transform=transform)

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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
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
