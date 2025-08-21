import os
import csv
import time
import torch
import argparse
import numpy as np
from typing import Optional
from transformer.model import Transformer
from transformer.optimizer import (
    AdamW, 
    cross_entropy, 
    learning_rate_scheduler, 
    gradient_clipping
)
from transformer.loader import get_batch, load_bin_array
from transformer.checkpoint import save_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_tokens", type=str, required=True, help="Path to training tokens .bin")
    p.add_argument("--valid_tokens", type=str, required=True, help="Path to validation tokens .bin")
    p.add_argument("--dtype", type=str, default="uint16", help="Token dtype for .bin (default: uint16)")

    # Model
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--theta", type=float, default=10_000.0)

    # Optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--weight_decay", type=float, default=0.1)

    # Logging / eval / ckpt
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=50, help="Num validation batches per eval")
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")

    # System
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1709)

    # Performance
    p.add_argument("--compile", action="store_true", help="Enable torch.compile for the model")
    p.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model: Transformer,
             x_valid: np.ndarray,
             batch_size: int,
             context_length: int,
             device: str,
             num_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb = get_batch(x_valid, batch_size, context_length, device)
        logits = model(xb)
        loss = cross_entropy(logits, yb).item()
        losses.append(loss)
    model.train()
    return float(sum(losses) / max(1, len(losses)))


HEADER = f"{'step':>8} | {'lr':>9} | {'loss':>9} | {'ema':>9} | {'val':>9} | {'g_norm':>7} | {'tok/s':>10} | {'time':>8}"

def format_row(step: int, lr: float, loss_item: float, ema: float, val_loss: float, grad_norm: float, tps: float, elapsed: float) -> str:
	return f"{step:8d} | {lr:9.5g} | {loss_item:9.4f} | {ema:9.4f} | {val_loss:9.4f} | {float(grad_norm):7.3f} | {tps:10,.0f} | {elapsed:8.2f}s"


def main():
    args = parse_args()
    set_seed(args.seed)

    if "cuda" in args.device:
        torch.set_float32_matmul_precision('high')

    os.makedirs(args.outdir, exist_ok=True)
    metrics_path = os.path.join(args.outdir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    metrics_f = open(metrics_path, "a", newline="")
    metrics_writer = csv.writer(metrics_f)
    if write_header:
        metrics_writer.writerow(["step","lr","train_loss","ema_loss","val_loss","grad_norm","tok_per_s","elapsed_s"])

    # Load datasets (.bin memmap)
    x_train = load_bin_array(args.train_tokens, dtype=args.dtype)
    x_valid = load_bin_array(args.valid_tokens, dtype=args.dtype)

    assert np.issubdtype(x_train.dtype, np.integer), f"train dtype must be integer, got {x_train.dtype}"
    assert np.issubdtype(x_valid.dtype, np.integer), f"valid dtype must be integer, got {x_valid.dtype}"

    device = torch.device(args.device)

    # Build model
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device,
        dtype=None,
    )
    model.to(device)

    # Optional compile
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    # Optimizer (custom AdamW; stateful and checkpointable)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay)

    start_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        # Restore states to the requested device (also moves optimizer states)
        start_step = load_checkpoint(args.resume, model, optimizer, device=device)
        print(f"Resumed from {args.resume} at step={start_step}")

    # Training loop
    model.train()
    tokens_per_step = args.batch_size * args.context_length
    t0 = time.time()
    running_loss: Optional[float] = None
    printed_header = False

    for step in range(start_step, args.max_steps):
        # LR schedule (cosine with warmup)
        lr = learning_rate_scheduler(
            t=step,
            lr_max=args.lr,
            lr_min=args.min_lr,
            t_warmup=max(1, args.warmup_steps),
            t_cos=max(args.warmup_steps + 1, args.max_steps),
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Sample batch
        xb, yb = get_batch(x_train, args.batch_size, args.context_length, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            logits = model(xb)
            loss = cross_entropy(logits, yb)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        grad_norm = gradient_clipping(list(model.parameters()), args.clip_grad)

        # Step
        optimizer.step()

        # Logging
        loss_item = float(loss.item())
        running_loss = loss_item if running_loss is None else (0.95 * running_loss + 0.05 * loss_item)

        if (step + 1) % args.log_interval == 0 or step == start_step:
            elapsed = time.time() - t0
            toks = tokens_per_step * (args.log_interval if (step + 1) % args.log_interval == 0 else 1)
            tps = toks / max(1e-6, elapsed)
            # eval synced with logging
            val_loss = evaluate(model, x_valid, args.batch_size, args.context_length, device=str(device), num_batches=args.eval_batches)
            if not printed_header or ((step + 1) % (args.log_interval * 20) == 0):
                print(HEADER)
                printed_header = True
            print(format_row(step + 1, lr, loss_item, running_loss, val_loss, grad_norm, tps, elapsed))
            metrics_writer.writerow([step + 1, lr, loss_item, running_loss, val_loss, float(grad_norm), tps, elapsed])
            metrics_f.flush()
            t0 = time.time()

        # Periodic save
        if (step + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.outdir, f"step_{step+1}.pt")
            save_checkpoint(model, optimizer, iteration=step+1, out=ckpt_path)
            print(f"[ckpt] saved to {ckpt_path}")

    # Final save
    final_ckpt = os.path.join(args.outdir, f"final_step_{args.max_steps}.pt")
    save_checkpoint(model, optimizer, iteration=args.max_steps, out=final_ckpt)
    print(f"[ckpt] final saved to {final_ckpt}")
    metrics_f.close()


if __name__ == "__main__":
    main()
