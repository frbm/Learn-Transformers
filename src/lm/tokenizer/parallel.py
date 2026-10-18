"""Tokenise on all cores. This module imports only the tokeniser.

On Windows (spawn) and macOS, each worker re-imports the module of the function it runs. A worker function in a
module that imports torch makes every worker import torch: seconds of start-up per worker. Keep workers here.
"""
import multiprocessing as mp
from collections.abc import Iterable, Iterator

import numpy as np

from lm.tokenizer.bpe import Tokenizer

EOT = "<|endoftext|>"
_TOK: Tokenizer | None = None


def _init(path: str) -> None:
    # One tokeniser per worker, loaded once. Its pretoken cache then lives in that worker for the whole run.
    global _TOK
    _TOK = Tokenizer.load(path)


def _encode_doc(text: str) -> np.ndarray:
    return np.asarray(_TOK.encode(text), np.uint16)


def _encode_batch(docs: list[str]) -> np.ndarray:
    eot = _TOK.special[EOT]
    ids: list[int] = []
    for d in docs:
        ids.extend(_TOK.encode(d))
        ids.append(eot)
    return np.asarray(ids, dtype=np.uint16)


def encode_docs(tok_path: str, docs: Iterable[str], workers: int, chunksize: int = 32) -> Iterator[np.ndarray]:
    """uint16 ids per document, in order (imap). The input is consumed lazily: nothing waits for the whole corpus."""
    with mp.Pool(workers, _init, (tok_path,)) as pool:
        yield from pool.imap(_encode_doc, docs, chunksize=chunksize)


def encode_batches(tok_path: str, batches: Iterable[list[str]], workers: int) -> Iterator[np.ndarray]:
    """One flat uint16 array per batch of documents, an eot after each document, in order."""
    with mp.Pool(workers, _init, (tok_path,)) as pool:
        yield from pool.imap(_encode_batch, batches)
