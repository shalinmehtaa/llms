import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Union, BinaryIO, IO

PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]

def save_checkpoint(model: Module, optimizer: Optimizer, iteration: int, out: PathOrFile) -> None:
    """Save model/optimizer states and iteration to a file path or binary file-like object."""
    obj = {
        "iteration": int(iteration),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pytorch_version": torch.__version__,
    }
    torch.save(obj, out)

def load_checkpoint(src: PathOrFile, model: Module, optimizer: Optimizer, device: Union[str, torch.device]) -> int:
    """Load checkpoint from a path or binary file-like object and restore states. Returns iteration number."""
    device = torch.device(device)

    # Explicitly disable weights-only safety gate introduced in PyTorch 2.6
    ckpt = torch.load(src, map_location=device, weights_only=False)

    model.to(device)
    model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                state[k] = v.to(device)

    return int(ckpt["iteration"])
