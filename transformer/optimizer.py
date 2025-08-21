import math
import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Tuple


def cross_entropy(logits: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    """Implement numerically stable cross-entropy loss from scratch"""
    # Numerical stability: subtract max logit per batch element
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - logits_max
    log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=-1))
    # Select logits corresponding to the target indices (shape: batch_size seq_len)
    target_logits = logits_shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Compute loss
    nll = log_sum_exp - target_logits
    return nll.mean()


def learning_rate_scheduler(t, lr_max, lr_min, t_warmup, t_cos):
    """Cosine annealing learning rate schedule"""
    if t <= t_warmup:
        lr = (t / t_warmup) * lr_max
    elif t_warmup < t < t_cos:
        lr = lr_min + 0.5 * (1 + math.cos(((t - t_warmup) / (t_cos - t_warmup)) * math.pi)) * (lr_max - lr_min)
    else:
        lr = lr_min
    return lr


def gradient_clipping(params, max_l2_norm: float, eps: Optional[float] = 1e-6):
    """Implement gradient clipping"""
    total_norm_sq = torch.tensor(0.0, device=params[0].device)
    for p in params:
        if p.grad is None:
            continue
        total_norm_sq += p.grad.detach().float().pow(2).sum()
    total_norm = total_norm_sq.sqrt()

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale)
    return total_norm


class AdamW(torch.optim.Optimizer):
    """Implement AdamW optimizer from scratch"""
    def __init__(self, 
                 params: Float[Tensor, "..."],
                 lr: float, 
                 betas: Tuple[float],
                 weight_decay: float,
                 eps: Optional[float] = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be > 0.")
        self.eps = eps
        defaults = {
            "lr": lr, 
            "betas": betas,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # Get the optimizer params
            lr = group["lr"] 
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Get state associated with p
                state = self.state[p] 
                # Get current state values
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                # Get the gradient for p
                grad = p.grad.data 
                # Compute moments and lr
                m_t = beta_1 * m + (1 - beta_1) * grad
                v_t = beta_2 * v + (1 - beta_2) * grad**2
                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))
                
                # Update parameter in place with weight decay
                p.data -= lr_t * (m_t / (torch.sqrt(v_t) + self.eps)) 
                p.data -= lr * weight_decay * p.data
                
                # Update state dict
                state["t"] = t + 1 
                state["m"] = m_t
                state["v"] = v_t
                
        return loss
    