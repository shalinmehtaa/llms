## Scratchformer

A Transformer implementation from scratch in PyTorch along with a custom BPE tokenizer.

### Components

- **Architecture**: Multi-head attention with RoPE, FFN with SwiGLU activations, (pre-)RMSNorm
- **Tokenizer**: Byte-pair encoding with pretokenization and special token handling
- **Training**: Distributed training with FSDP, gradient accumulation, mixed precision

### Setup

```bash
uv sync
```

### Train BPE Tokenizer

```bash
uv run -m tokenization.bpe.train \
  --file_path data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 50304 \
  --special_tokens_path data/special_tokens.txt \
  --tokenizer_out_path tokenization/tokenizers/tinystories-tokenizer-50304.json
```

### Tokenize Dataset

```bash
uv run -m tokenization.bpe.tokenizer \
  --dataset_path data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer_path tokenization/tokenizers/tinystories-tokenizer-50304.json \
  --out_path data/inputs/tinystories-train-tokens.bin \
  --append_eot
```

### Train Model

Single GPU:
```bash
uv run -m transformer.train \
  --train_tokens data/inputs/tinystories-train-tokens.bin \
  --valid_tokens data/inputs/tinystories-valid-tokens.bin \
  --vocab_size 50304 \
  --context_length 512 \
  --num_layers 8 \
  --d_model 768 \
  --num_heads 16 \
  --d_ff 2048 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --clip_grad 1.0 \ 
  --max_steps 20000 --warmup_steps 200 \
  --log_interval 10 --save_interval 1000 \
  --compile \
  --outdir checkpoints \
```

Multi-GPU with FSDP:
```bash
source .venv/bin/activate

torchrun --nproc_per_node=4 -m transformer.train \
  --train_tokens data/inputs/tinystories-train-tokens.bin \
  --valid_tokens data/inputs/tinystories-valid-tokens.bin \
  --vocab_size 50304 \
  --context_length 1024 \
  --num_layers 24 \
  --d_model 1024 \
  --num_heads 16 \
  --d_ff 2752 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-4 \
  --min_lr 3e-5
  --max_steps 10000 --warmup_steps 100 \
  --fsdp \
  --log_interval 10 --save_interval 2000 \
  --outdir checkpoints \
  --weights_only
```
