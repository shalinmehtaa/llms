import json
import base64
import argparse
from pathlib import Path
from typing import (
    Dict, 
    List, 
    Tuple, 
    Optional, 
    Self,
    Iterable,
    Iterator
)
import numpy as np
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from ..pretokenization import pretokenize_chunk

class Tokenizer:

    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[Tuple[bytes, bytes]], 
                 special_tokens: Optional[List[str]] = None):
        """Initialize tokenizer with vocab, merges and special tokens"""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else list()

         # Append special tokens to the vocab if not already there
        existing_values = set(self.vocab.values())
        next_id = max(self.vocab.keys()) + 1
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in existing_values:
                self.vocab[next_id] = special_token_bytes
                existing_values.add(special_token_bytes)
                next_id += 1

        # Fast lookup from bytes token to id
        self._token_to_id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Map raw bytes of special token text -> its id for single-token emission
        self._special_token_bytes_to_id: Dict[bytes, int] = {}
        for s in self.special_tokens:
            b = s.encode("utf-8")
            tok_id = self._token_to_id.get(b)
            if tok_id is not None:
                self._special_token_bytes_to_id[b] = tok_id

        # Rank of each merge pair (lower index = higher priority)
        self._merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_file(cls, tokenizer_filepath: str) -> Self:
        with open(tokenizer_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "vocab" not in data or "merges" not in data:
            raise ValueError("Invalid tokenizer file: expected keys 'vocab' and 'merges'.")

        enc = data.get("byte_encoding", "base64").lower()
        if enc != "base64":
            raise ValueError(f"Unsupported byte_encoding: {enc!r}; expected 'base64'.")

        vocab_items = data["vocab"].items()
        vocab: Dict[int, bytes] = {int(id_): base64.b64decode(tok) for tok, id_ in vocab_items}

        merges_raw = data["merges"]
        merges: List[Tuple[bytes, bytes]] = []
        for pair in merges_raw:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid merge pair: {pair!r}")
            left = base64.b64decode(pair[0])
            right = base64.b64decode(pair[1])
            merges.append((left, right))

        special_tokens = data.get("special_tokens") or []
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> Iterator[int]:
        # Emit special token as a single id if matched exactly
        st_id = self._special_token_bytes_to_id.get(pretoken)
        if st_id is not None:
            yield st_id
            return
        
        tokens: List[bytes] = [bytes([b]) for b in pretoken]

        # Greedy BPE: repeatedly merge the lowest-rank adjacent pair
        if len(tokens) >= 2:
            while True:
                best_pair: Optional[Tuple[bytes, bytes]] = None
                best_rank: Optional[int] = None
                # Find best-ranked adjacent pair present
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    rank = self._merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break

                # Merge all occurrences of best_pair, left-to-right
                i = 0
                out: List[bytes] = []
                while i < len(tokens):
                    if (
                        i + 1 < len(tokens)
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]
                    ):
                        out.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        out.append(tokens[i])
                        i += 1
                tokens = out

        for tok in tokens:
            yield self._token_to_id[tok]

    def encode(self, text: str) -> List[int]:
        pretokens: List[bytes] = pretokenize_chunk(chunk=text, 
                                                   special_tokens=self.special_tokens,
                                                   training=False)
        ids: List[int] = []
        for pretoken in pretokens:
            ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for pretoken in pretokenize_chunk(chunk=text, 
                                              special_tokens=self.special_tokens,
                                              training=False):
                yield from self._encode_pretoken(pretoken)

    def decode(self, ids: List[int]) -> str:
        byte_stream = b"".join(self.vocab[id] for id in ids)
        return byte_stream.decode("utf-8", errors="replace")


def _iter_texts(dataset_path: str) -> Iterator[str]:
    """Yield text lines from a file or all .txt files in a directory, lazily."""
    p = Path(dataset_path)
    if p.is_dir():
        files = sorted(fp for fp in p.rglob("*.txt") if fp.is_file())
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    yield line
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                yield line


def _read_special_tokens(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_special_id(tokenizer: "Tokenizer", token_text: str) -> int:
    ids = tokenizer.encode(token_text)
    if len(ids) != 1:
        raise ValueError(f"Special token {token_text!r} did not resolve to a single id.")
    return ids[0]


def _ensure_uint16_capacity(tokenizer: "Tokenizer"):
    max_id = max(tokenizer.vocab.keys())
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer vocab ids exceed uint16 capacity (max id {max_id}). "
            f"Use a smaller vocab or a wider dtype."
        )


def _tokenize_to_bin(tokenizer: "Tokenizer",
                     dataset_path: str,
                     out_path: str,
                     append_eot: Optional[str],
                     buffer_size: int = 1_000_000,
                     show_progress: bool = True) -> int:
    """Stream token ids to a raw .bin file as uint16; returns total token count."""
    eot_id: Optional[int] = None
    if append_eot:
        eot_id = _resolve_special_id(tokenizer, append_eot)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    buf: List[int] = []

    itr = _iter_texts(dataset_path)
    if show_progress:
        itr = tqdm(itr, desc="Tokenizing (bin)", unit="lines")

    with open(out_path, "wb") as f:
        for text in itr:
            for tok in tokenizer.encode_iterable([text]):
                buf.append(tok)
                if len(buf) >= buffer_size:
                    np.asarray(buf, dtype=np.uint16).tofile(f)
                    total += len(buf)
                    buf.clear()
            if eot_id is not None:
                buf.append(eot_id)
                if len(buf) >= buffer_size:
                    np.asarray(buf, dtype=np.uint16).tofile(f)
                    total += len(buf)
                    buf.clear()

        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
            buf.clear()

    return total


def _count_tokens(tokenizer: "Tokenizer",
                  dataset_path: str,
                  append_eot: Optional[str],
                  show_progress: bool = True) -> tuple[int, Optional[int]]:
    total = 0
    eot_id: Optional[int] = None
    if append_eot:
        eot_id = _resolve_special_id(tokenizer, append_eot)

    itr = _iter_texts(dataset_path)
    if show_progress:
        itr = tqdm(itr, desc="Counting (npy pass 1/2)", unit="lines")

    for text in itr:
        c = 0
        for _ in tokenizer.encode_iterable([text]):
            c += 1
        total += c
        if eot_id is not None:
            total += 1

    return total, eot_id


def _tokenize_to_npy(tokenizer: "Tokenizer",
                     dataset_path: str,
                     out_path: str,
                     append_eot: Optional[str],
                     buffer_size: int = 1_000_000,
                     show_progress: bool = True) -> int:
    """Write token ids to a .npy file (uint16) via two passes using a memmap."""
    total, eot_id = _count_tokens(tokenizer, dataset_path, append_eot, show_progress)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mm = open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total,))
    pos = 0
    buf: List[int] = []

    itr = _iter_texts(dataset_path)
    if show_progress:
        itr = tqdm(itr, desc="Writing (npy pass 2/2)", unit="lines")

    for text in itr:
        for tok in tokenizer.encode_iterable([text]):
            buf.append(tok)
            if len(buf) >= buffer_size:
                n = len(buf)
                mm[pos:pos + n] = np.asarray(buf, dtype=np.uint16)
                pos += n
                buf.clear()
        if eot_id is not None:
            buf.append(eot_id)
            if len(buf) >= buffer_size:
                n = len(buf)
                mm[pos:pos + n] = np.asarray(buf, dtype=np.uint16)
                pos += n
                buf.clear()

    if buf:
        n = len(buf)
        mm[pos:pos + n] = np.asarray(buf, dtype=np.uint16)
        pos += n
        buf.clear()

    mm.flush()
    return total


def _default_out_path(dataset_path: str, out_format: str) -> str:
    base = Path(dataset_path)
    if base.is_dir():
        name = base.name
    else:
        name = base.stem
    ext = ".npy" if out_format == "npy" else ".bin"
    return str(Path("data") / f"{name}-tokens{ext}")


def _write_meta(meta_path: str, meta: dict):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset with a given tokenizer JSON (containing vocab and merges).")
    parser.add_argument(
        "--dataset_path", "-d", type=str, required=True,
        help="Path to the input dataset (a .txt file or a directory containing .txt files)."
    )
    parser.add_argument(
        "--tokenizer_path", "-t", type=str, required=True,
        help="Path to the tokenizer JSON (vocab+merges in one file)."
    )
    parser.add_argument(
        "--special_tokens_path", "-sp", type=str, required=False,
        help="Optional path to a file with special tokens (one per line)."
    )
    parser.add_argument(
        "--out_path", "-o", type=str, required=False,
        help="Output file path. Defaults to data/<dataset_name>-tokens.(bin|npy) based on --format."
    )
    parser.add_argument(
        "--format", "-f", type=str, choices=["bin", "npy"], default="bin",
        help="Serialization format: 'bin' (streaming raw uint16) or 'npy' (memmapped two-pass)."
    )
    parser.add_argument(
        "--append_eot", action="store_true",
        help="Append an end-of-text special token after each input line."
    )
    parser.add_argument(
        "--eot_token", type=str, default="<|endoftext|>",
        help="Which special token string to append when --append_eot is set."
    )
    parser.add_argument(
        "--no_tqdm", action="store_true",
        help="Disable progress bars."
    )
    args = parser.parse_args()

    # Load tokenizer
    special_tokens = _read_special_tokens(args.special_tokens_path)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    # If explicit special tokens are provided, override
    if special_tokens is not None:
        tokenizer.special_tokens = special_tokens
        # rebuild special bytes->id map
        tokenizer._special_token_bytes_to_id = {}
        for s in tokenizer.special_tokens:
            b = s.encode("utf-8")
            tok_id = tokenizer._token_to_id.get(b)
            if tok_id is not None:
                tokenizer._special_token_bytes_to_id[b] = tok_id

    _ensure_uint16_capacity(tokenizer)

    out_path = args.out_path or _default_out_path(args.dataset_path, args.format)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    show_progress = not args.no_tqdm

    # Invoke the actual tokenization on the inner functions
    if args.format == "bin":
        total = _tokenize_to_bin(
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            out_path=out_path,
            append_eot=(args.eot_token if args.append_eot else None),
            show_progress=show_progress,
        )
    else:
        total = _tokenize_to_npy(
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            out_path=out_path,
            append_eot=(args.eot_token if args.append_eot else None),
            show_progress=show_progress,
        )

    meta = {
        "total_tokens": int(total),
        "dtype": "uint16",
        "format": args.format,
        "dataset_path": args.dataset_path,
        "tokenizer_path": args.tokenizer_path,
        "special_tokens_path": args.special_tokens_path,
        "append_eot": bool(args.append_eot),
        "eot_token": args.eot_token if args.append_eot else None,
    }
    _write_meta(out_path + ".meta.json", meta)
    print(f"Wrote {total} tokens to {out_path} (uint16). Meta: {out_path}.meta.json")