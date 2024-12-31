import os
from pathlib import Path


def get_model_path(name: str, load: bool) -> str:
    path = f"models/{name}.pth"
    if not Path(path).is_file() and "KAGGLE_KERNEL_RUN_TYPE" in os.environ and load:
        path = f"/kaggle/input/{name}/pytorch/default/{name}.pth"
    assert Path(path).is_file(), "Couldn't find a model"
    return path