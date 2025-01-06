import os
from pathlib import Path


def get_model_path(name: str, load: bool) -> str:
    path = f"models/{name}.pth"
    if (
        not Path(path).is_file()
        and "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        and load
    ):
        path = f"/kaggle/input/{name}/pytorch/default/1/{name}.pth"
    assert Path(path).is_file() or not load, f"Couldn't find a model in {path}"
    return path
