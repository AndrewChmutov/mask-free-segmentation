from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import trange


class CrackDataset(ABC, Dataset):
    def __init__(
        self, root_dir: str | Path, names: list[str] = [], **kwargs
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_paths = list((Path(root_dir) / "images").iterdir())

        if names:
            self.image_paths = [
                path for path in self.image_paths if path in names
            ]

        desc = self.root_dir.relative_to(self.root_dir.parent.parent)
        self.X = [
            self._transform_x(data)
            for data in trange(len(self), desc=f"{desc}: X")
        ]
        self.y = [
            self._transform_y(data)
            for data in trange(len(self), desc=f"{desc}: y")
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _transform(self, path: Path) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")

    def _transform_x(self, i: int) -> Image.Image:
        return self._transform(self.image_paths[i])

    @abstractmethod
    def _transform_y(self, i: int) -> Any: ...

    def _get_x(self, i: int) -> Image.Image:
        return self.X[i]

    def _get_y(self, i: int) -> Any:
        return self.y[i]

    def __getitem__(self, idx: int) -> tuple[str, Any, Any]:
        return str(self.image_paths[idx]), self._get_x(idx), self._get_y(idx)


class WithTransform(CrackDataset):
    def __init__(self, *args, **kwargs) -> None:
        assert "transform" in kwargs
        self.transform = kwargs["transform"]
        super().__init__(*args, **kwargs)

    def _get_x(self, i: int) -> Image.Image:
        return self.transform(self.X[i])


class CrackDataset3(WithTransform):
    def _transform_y(self, i: int) -> Any:
        return 0 if self.image_paths[i].name.startswith("noncrack") else 1


class CrackDataset4(WithTransform):
    def _transform_y(self, i: int) -> Any:
        path = self.root_dir / "masks" / self.image_paths[i].name
        return np.array(Image.open(path).convert("L"))


class FilteredDataset(Subset):
    def __init__(self, dataset: CrackDataset, paths: list[str]) -> None:
        indices = [
            i
            for i, path in enumerate(dataset.image_paths)
            if str(path in paths)
        ]
        super().__init__(dataset, indices)


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
    root_dir: str | Path,
    transform: Callable | None = None,
    val_size: float = 0.3,
    batch_size: int = 32,
    random_state: int = 17,
) -> DataResult:
    root_dir = Path(root_dir)
    dataset = CrackDataset3(root_dir=root_dir / "train", transform=transform)

    indices = np.arange(len(dataset))
    labels = dataset.y

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=random_state
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = CrackDataset3(
        root_dir=root_dir / "test", transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
    root_dir: str | Path,
    transform: Callable | None = None,
    val_size: float = 0.3,
    batch_size: int = 32,
    random_state: int = 17,
) -> DataResult:
    root_dir = Path(root_dir)
    dataset = CrackDataset3(root_dir=root_dir / "train", transform=transform)

    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_state
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)
    test_dataset = CrackDataset3(root_dir="data/test", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
    root_dir: str | Path,
    version: Literal["3"] | Literal["4"] | Literal["5"],
    **kwargs,
) -> DataResult:
    match version:
        case "3" | "5":
            return _load3(root_dir, **kwargs)
        case "4":
            return _load4(root_dir, **kwargs)
