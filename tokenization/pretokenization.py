import regex
import argparse
import multiprocessing as mp

from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor
from .pretokenization_chunking import find_chunk_boundaries

# GPT-2 pretokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = regex.compile(PAT)

def get_pretokenization_tasks(file_path: str,
                              num_chunks: int,                
                              special_tokens: List[str]) -> List[str]:
    """Find chunk boundaries and generate pretokenization tasks"""
    file_path = Path(file_path)
    # find_chunk_boundaries only supports passing in a single special token (in bytes)
    split_special_token = special_tokens[0].encode("utf-8")
    with open(file_path, "rb") as file:
        # Use the function provided by the Stanford lecturers to find boundaries
        chunk_boundaries = find_chunk_boundaries(file, 
                                                 num_chunks, 
                                                 split_special_token)

        tasks = list()
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start)
            # Convert chunk to str since will be working with regex downstream
            chunk = chunk.decode("utf-8")
            tasks.append(chunk)

        return tasks


def pretokenize_chunk(chunk: str,
                      special_tokens: str) -> List[bytes]:
    """Pretokenize a chunk of text"""
    # Working with strings since regex and other operations below work better with strings
    assert type(chunk) == str, "chunk must be str type"
    assert all(isinstance(token, str) for token in special_tokens), "all special tokens must be str type"
    
    # Start by splitting on special tokens
    pattern = ("|".join(map(regex.escape, special_tokens)))
    segments = regex.split(pattern, chunk)

    pretokens = list()
    for segment in segments:
        for match in regex.finditer(PAT_COMPILED, segment):
            # Return pretokens as bytes
            pretokens.append(match.group().encode("utf-8"))

    return pretokens


def pretokenize(file_path: str,
                special_tokens_path: Optional[str] = None,
                num_chunks: Optional[int] = None,
                max_workers: Optional[int] = None) -> List[bytes]:
    """Split a given text file into chunks and pretokenize in parallel"""
    # Set default values if none provided
    if num_chunks is None:
        num_chunks = 8

    if max_workers is None:
        max_workers = mp.cpu_count() - 1

    special_tokens = ["<|endoftext|>"] # Default
    if special_tokens_path is not None:
        try:
            with open(special_tokens_path, "r") as special_tokens_file:
                special_tokens = [line.strip() for line in special_tokens_file]
        except Exception as e:
            print(f"Warning: Could not read special tokens from {special_tokens_path}: {e}")

    # Generate tasks to parallelise
    tasks = get_pretokenization_tasks(
        file_path=file_path,
        num_chunks=num_chunks,
        special_tokens=special_tokens
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(pretokenize_chunk, task, special_tokens) for task in tasks
        ]
        results = [future.result() for future in futures]

    pretokens = [
        pretoken for chunk_pretokens in results for pretoken in chunk_pretokens
    ]

    return pretokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", 
        "-f",
        type=str,
        required=True,
        help="Path to the input text file to be pretokenized"
    )
    parser.add_argument(
        "--special_tokens_path", 
        "-sp",
        type=str,
        required=False,
        help="Path to the text file with special tokens (one per line)"
    )
    parser.add_argument(
        "--max_workers", 
        "-w",
        type=int,
        required=False,
        help="Number of concurrent worker processes to use for pretokenization"
    )
    parser.add_argument(
        "--num_chunks", 
        "-n", 
        type=int,
        required=False,
        help="Number of chunks to split the input text into"
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        required=False,
        help="File path to save down outputs to. Must include file name."
    )
    args = parser.parse_args()

    main_args = vars(args).copy()
    main_args.pop("save_path")
    pretokens = pretokenize(**main_args)

    # Save outputs to file if path is provided
    if hasattr(args, "save_path") and args.save_path is not None:
        with open(args.save_path, "w", encoding="utf-8") as save_file:
            save_file.writelines(f"{pretoken}\n" for pretoken in pretokens)
