import json
from typing import (
    Dict, 
    List, 
    Tuple, 
    Optional, 
    Self,
    Iterable,
    Iterator
)
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
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath: str,
                   special_tokens: Optional[List[str]] = None) -> Self:
        
        with open(vocab_filepath, "r", encoding="utf-8") as vocab_file:
            vocab: Dict[str, int] = json.load(vocab_file)
            vocab: Dict[int, bytes] = {id_: token.encode("utf-8") for token, id_ in vocab.items()}

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as merges_file:
            for raw in merges_file:
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                try:
                    left_str, right_str = line.rsplit(" ", 1)
                except ValueError:
                    raise ValueError(f"Invalid merge line (expected two tokens): {repr(line)}")
                merges.append((left_str.encode("utf-8"), right_str.encode("utf-8")))

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

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizers/TinyStoriesV2-vocab.json",
        merges_filepath="data/tokenizers/TinyStoriesV2-merges.txt",
        special_tokens_path="data/special_tokens.txt"
    )
    print(tokenizer.encode("Hello how are you?"))
