## Scratchformer

A Transformer implementation from scratch with a custom BPE tokenizer.

- **Transformer model**: Multi-head attention, SwiGLU, RMSNorm, RoPE
- **BPE tokenizer**: Custom implementation with pretokenization and special token support
- **Training setup**: Infrastructure for distributed training

### Install

```bash
uv sync
```
- Main dependencies: PyTorch, einops, jaxtyping

### Usage

```python
from transformer.model import Transformer
from tokenization.bpe.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="data/tokenizers/vocab.json",
    merges_filepath="data/tokenizers/merges.txt"
)

# Create model
model = Transformer(
    vocab_size=len(tokenizer.vocab),
    context_length=2048,
    num_layers=12,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    theta=10000.0
)

# Use it
tokens = tokenizer.encode("Hello world")
output = model(torch.tensor([tokens]))
```
