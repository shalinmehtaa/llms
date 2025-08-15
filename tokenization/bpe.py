from typing import List, Dict, Tuple, Optional
from .pretokenization import pretokenize

class BPETokenizer:
    """An efficient implementation of the Byte Pair Encoding tokenization algorithm"""
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        num_special_tokens = len(self.special_tokens)
        self.num_merges = self.vocab_size - (256 + num_special_tokens)

        if self.num_merges < 0:
            raise ValueError("vocab_size must be greater than 256 + the number of special tokens")

    def _get_pair_counts(self, pretokens: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Get byte pair counts for each pretoken during a first pass of the data"""
        pair_counts = dict()
        # Working with integers here
        for pretoken in pretokens:
            for left_id, right_id in zip(pretoken[:-1], pretoken[1:]):
                pair_counts[(left_id, right_id)] = pair_counts.get((left_id, right_id), 0) + 1

        return pair_counts
    
    def _get_incremental_pair_counts(self,
                                     pair_counts: Dict[Tuple[int, int], int],
                                     old_pretokens: List[List[int]],
                                     new_pretokens: List[List[int]],
                                     max_count_pair: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """Incrementally updated pair counts by only modifying counts for pretokens with the merged pair."""
        # Creat copy to store new counts
        updated_counts = pair_counts.copy()

        for old_pretoken, new_pretoken in zip(old_pretokens, new_pretokens):
            # Pretoken hasn't changed, skip updating counts
            if old_pretoken == new_pretoken:
                continue
            
            # Remove counts for merged pair based if found in old pretokens
            for left_id, right_id in zip(old_pretoken[:-1], old_pretoken[1:]):
                pair = (left_id, right_id)
                if pair in updated_counts:
                    updated_counts[pair] -= 1
                    if updated_counts[pair] == 0:
                        del updated_counts[pair]

            # Add counts for merged token if found in new pretokens
            for left_id, right_id in zip(new_pretoken[:-1], new_pretoken[1:]):
                pair = (left_id, right_id)
                updated_counts[pair] = \
                    updated_counts.get(pair, 0) + 1

        return updated_counts
            
    def _get_pair_with_max_count(self, 
                                pair_counts: Dict[Tuple[int, int], int],
                                vocab: Dict[int, bytes]) -> Tuple[int, int]:
        max_count = max(pair_counts.values())
        max_count_pairs = [
            pair for pair, count in pair_counts.items() if count == max_count
        ]

        # Return the only pair with max count if no ties
        if len(max_count_pairs) == 1:
            return max_count_pairs[0]
        
        # If there are ties, choose the lexicographically greater pair based on byte representation
        def pair_to_bytes(pair):
            return (vocab[pair[0]], vocab[pair[1]])
    
        # Find the pair with lexicographically maximum byte representation
        max_count_pair = max(max_count_pairs, key=pair_to_bytes)
        
        return max_count_pair

    def _apply_merge(self, 
                     pretokens: List[List[int]], 
                     next_token_id: int,
                     max_count_pair: Tuple[int, int]) -> List[List[int]]:
        """Recursively merge all occurrences of the pair."""
        def merge_recursively(pretoken: List[int]):
            # Nothing to merge
            if len(pretoken) < 2:
                return pretoken
            
            # Working with integers here
            for i in range(len(pretoken) - 1):
                if pretoken[i] == max_count_pair[0] and pretoken[i+1] == max_count_pair[1]:
                    before = pretoken[:i]
                    after = pretoken[i+2:]
                    return before + [next_token_id] + merge_recursively(after)
                
            return pretoken # List of integers

        return [merge_recursively(pretoken) for pretoken in pretokens]
    

    def train(self, 
              pretokens: List[List[int]]) -> Tuple[Dict[int, bytes], List[bytes]]:

        # Initialize vocabulary
        vocab = {i: bytes([i]) for i in range(256)}

        # Add special tokens to the vocabulary
        next_token_id = len(vocab)
        for special_token in self.special_tokens:
            vocab[next_token_id] = special_token.encode("utf-8")
            next_token_id +=1
        
        # Get first pass byte pair counts across all text
        pair_counts = self._get_pair_counts(pretokens)

        # Initialize merges list
        merges = list()

        for _ in range(self.num_merges):

            max_count_pair = self._get_pair_with_max_count(
                pair_counts=pair_counts,
                vocab=vocab
            )

            # Add pair to merges as bytes
            merges.append((
                vocab[max_count_pair[0]],
                vocab[max_count_pair[1]]
            ))

            # Merge the bytes to mint a new token and add to vocab
            merged_token = vocab[max_count_pair[0]] + vocab[max_count_pair[1]]
            vocab[next_token_id] = merged_token
            
            old_pretokens = pretokens
            pretokens  = self._apply_merge(
                pretokens=pretokens,
                next_token_id=next_token_id,
                max_count_pair=max_count_pair
            )

            pair_counts = self._get_incremental_pair_counts(
                pair_counts=pair_counts,
                old_pretokens=old_pretokens,
                new_pretokens=pretokens,
                max_count_pair=max_count_pair
            )

            # Increment the token ID
            next_token_id += 1

        return vocab, merges
    

def main(file_path: str,
         special_tokens: List[str],
         vocab_size: int,
         max_workers: Optional[int] = 8,
         num_chunks: Optional[int] = 8):
    """Pretokenize in chunks and then train a BPE tokenizer"""
    pretokens = pretokenize(
        file_path=file_path,
        max_workers=max_workers,
        num_chunks=num_chunks
    )
    # Convert to integers for easier downstream processing
    pretokens = [list(pretoken) for pretoken in pretokens]

    bpe = BPETokenizer(vocab_size, special_tokens)
    vocab, merges = bpe.train(pretokens)
    
    return vocab, merges


if __name__ == "__main__":

    with open("data/special_tokens.txt", "r") as special_tokens_file:
        special_tokens = [line.strip() for line in special_tokens_file]

    vocab, merges = main("data/TinyStoriesV2-GPT4-valid.txt", 
                         special_tokens=special_tokens, 
                         vocab_size=300)
