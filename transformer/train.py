import torch
import numpy as np
from torch import Tensor
from typing import Tuple
from jaxtyping import Float
from numpy.typing import NDArray


def get_batch(x: NDArray,
              batch_size: int, 
              context_length: int, 
              device: str) -> Tuple[Float[Tensor, "batch_size seq_len"], 
                                    Float[Tensor, "batch_size seq_len"]]:
    """Sample a batch of contiguous token sequences from a 1D numpy array (or np.memmap)."""
    if x.ndim != 1:
        raise ValueError("Expected a 1D array of token IDs.")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"Expected integer dtype for token IDs, got {x.dtype}.")
    n = x.shape[0]
    if n <= context_length:
        raise ValueError(
            f"Dataset too small: n={n} must be > context_length={context_length}."
        )

    # Sample start indices so that i+context_length is within bounds for inputs
    starts = np.random.randint(0, n - context_length, size=batch_size, dtype=np.int64)

    # Build index matrix for inputs and targets using broadcasting
    offsets = np.arange(context_length, dtype=np.int64)[None, :]  # (1, m)
    idx_in = starts[:, None] + offsets                            # (B, m)
    idx_tg = idx_in + 1                                           # (B, m)

    # Fancy-index into x; works for memmap and in-memory arrays
    inputs_np  = x[idx_in]
    targets_np = x[idx_tg]

    # Convert to torch.long on the provided device
    inputs  = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
    
    return inputs, targets
