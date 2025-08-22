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
python -m tokenization.bpe.train \
  --file_path data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 50304 \
  --special_tokens_path data/special_tokens.txt \
  --tokenizer_out_path data/tokenizers/tinystories-tokenizer-50304.json
```

### Tokenize Dataset

```bash
python -m tokenization.bpe.tokenizer \
  --dataset_path data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer_path data/tokenizers/tinystories-tokenizer-50304.json \
  --out_path data/inputs/tinystories-train-tokens.npy \
  --append_eot
```

### Train Model

Single GPU:
```bash
python -m transformer.train \
  --train_tokens data/inputs/tinystories-train-tokens.npy \
  --valid_tokens data/inputs/tinystories-valid-tokens.npy \
  --vocab_size 50304 \
  --context_length 512 \
  --num_layers 8 \
  --d_model 768 \
  --num_heads 16 \
  --d_ff 2048 \
  --batch_size 64 \
  --lr 2e-4 \
  --max_steps 50000
```

Multi-GPU with FSDP:
```bash
torchrun --nproc_per_node=4 -m transformer.train \
  --train_tokens data/inputs/tinystories-train-tokens.npy \
  --valid_tokens data/inputs/tinystories-valid-tokens.npy \
  --vocab_size 50304 \
  --context_length 1024 \
  --num_layers 24 \
  --d_model 1024 \
  --num_heads 16 \
  --d_ff 2752 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-4 \
  --max_steps 100000 \
  --fsdp \
  --compile
```
