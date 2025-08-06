import json
import os
import pickle
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import lmdb
import ray
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


@contextmanager
def suppress_output(is_main_worker):
    """A context manager that suppresses stdout and stderr for non-main workers."""
    if is_main_worker:
        yield
        return

    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class OptimizedLMDBReader:
    """Optimized LMDB reader with prefetching and better memory management."""

    def __init__(self, mdb_file_path: str, prefetch_size: int = 10000):
        self.mdb_file_path = mdb_file_path
        self.prefetch_size = prefetch_size
        self._env = None
        self._lock = threading.Lock()

    def _get_env(self):
        """Thread-safe LMDB environment getter with optimized settings."""
        if self._env is None:
            with self._lock:
                if self._env is None:
                    self._env = lmdb.open(
                        self.mdb_file_path,
                        readonly=True,
                        lock=False,
                        readahead=True,  # Enable readahead for better performance
                        meminit=False,
                        max_readers=128,  # Increase max readers for better concurrency
                    )
        return self._env

    def get_stats(self):
        """Get LMDB statistics."""
        env = self._get_env()
        return env.stat()

    def prefetch_batch_generator(self, start_idx: int = 0, batch_size: int = 1000):
        """Generator that yields batches of data with prefetching."""
        env = self._get_env()
        with env.begin() as txn:
            cursor = txn.cursor()

            # Seek to start position
            if start_idx > 0:
                cursor.set_key(str(start_idx).encode("utf-8"))
            else:
                cursor.first()

            batch = []
            for key, value in cursor:
                try:
                    # Try JSON first (faster), fallback to pickle
                    try:
                        entry = json.loads(value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        entry = pickle.loads(value)

                    if (
                        isinstance(entry, dict)
                        and "sequence" in entry
                        and isinstance(entry.get("sequence"), str)
                    ):
                        batch.append({"sequence": entry["sequence"]})

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                except Exception:
                    # Log error but continue processing
                    continue

            # Yield remaining batch
            if batch:
                yield batch


def _dataset_generator_optimized(
    mdb_file_path: str, batch_size: int = 1000
) -> Iterator[Dict[str, Any]]:
    """
    Optimized generator that yields data entries from an LMDB file with batching and prefetching.
    """
    try:
        rank = ray.train.get_context().get_world_rank()
    except (ValueError, AttributeError):
        rank = 0

    # Suppress all output from non-main workers
    with suppress_output(is_main_worker=(rank == 0)):
        reader = OptimizedLMDBReader(mdb_file_path)
        stats = reader.get_stats()
        num_examples = stats["entries"]

        with tqdm(
            total=num_examples,
            desc="Loading dataset (optimized)",
            disable=(rank != 0),
        ) as pbar:
            for batch in reader.prefetch_batch_generator(batch_size=batch_size):
                for item in batch:
                    yield item
                pbar.update(len(batch))


def _optimized_preprocess_function(
    examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int = 512
) -> Dict[str, List[Any]]:
    """Optimized preprocessing function with dynamic padding and better memory usage."""
    sequences = examples["sequence"]

    # Add space prefix for ESM tokenization and filter by length
    processed_sequences = []
    for seq in sequences:
        # Pre-filter sequences that are too long to avoid tokenization overhead
        if len(seq) <= max_length - 2:  # Account for special tokens
            processed_sequences.append(" " + seq)
        else:
            # Truncate sequence if too long
            processed_sequences.append(" " + seq[: max_length - 2])

    # Use dynamic padding for better memory efficiency
    tokenized = tokenizer(
        processed_sequences,
        padding="longest",  # Use dynamic padding instead of max_length
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors=None,  # Keep as lists for now
    )

    return tokenized


def load_and_preprocess_data(
    lmdb_path: str,
    tokenizer: AutoTokenizer,
    cache_dir: str,
    max_length: int = 512,  # Reduced from 700 for better performance
    batch_size: int = 1000,
    preprocessing_num_proc: int = None,
) -> Dataset:
    """
    Optimized data loading and preprocessing with better caching and parallel processing.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)

    # Check for cached dataset
    if os.path.exists(cache_dir):
        print(f"Loading tokenized dataset from cache: {cache_dir}")
        return Dataset.load_from_disk(cache_dir)

    try:
        rank = ray.train.get_context().get_world_rank()
    except (ValueError, AttributeError):
        rank = 0

    # Determine number of processes for preprocessing
    if preprocessing_num_proc is None:
        try:
            # 使用 Ray 分配的 CPU 资源
            ray_cpus = int(
                ray.get_runtime_context().get_assigned_resources().get("CPU", 0)
            )
            preprocessing_num_proc = ray_cpus if ray_cpus > 0 else 12
        except Exception:
            preprocessing_num_proc = 12
        preprocessing_num_proc = 12

    print(f"Using {preprocessing_num_proc} processes for preprocessing")

    # Load dataset with optimized generator
    with suppress_output(is_main_worker=(rank == 0)):
        start_time = time.time()
        train_dataset = Dataset.from_generator(
            _dataset_generator_optimized,
            gen_kwargs={"mdb_file_path": lmdb_path, "batch_size": batch_size},
        )
        load_time = time.time() - start_time
        if rank == 0:
            print(f"Dataset loading completed in {load_time:.2f} seconds")

    # Optimized preprocessing with better parameters
    start_time = time.time()
    tokenized_train_dataset = train_dataset.map(
        lambda examples: _optimized_preprocess_function(
            examples, tokenizer, max_length
        ),
        batched=True,
        batch_size=2000,  # Larger batch size for better throughput
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dataset (optimized)",
        disable_nullable=(rank != 0),
        num_proc=preprocessing_num_proc,
        writer_batch_size=1000,  # Optimize write batch size
    )

    tokenization_time = time.time() - start_time
    if rank == 0:
        print(f"Tokenization completed in {tokenization_time:.2f} seconds")

    # Save to cache on main process only
    if rank == 0:
        print(f"Saving tokenized dataset to cache: {cache_dir}")
        tokenized_train_dataset.save_to_disk(cache_dir, num_proc=preprocessing_num_proc)

    return tokenized_train_dataset


# Legacy function for backward compatibility
def _dataset_generator(mdb_file_path: str) -> Iterator[Dict[str, Any]]:
    """Legacy generator function - use _dataset_generator_optimized instead."""
    return _dataset_generator_optimized(mdb_file_path)
