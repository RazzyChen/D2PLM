#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
#                    High-Performance FASTA to LMDB Converter
#
# DESCRIPTION:
#   This script converts large FASTA formatted sequence files into an LMDB
#   (Lightning Memory-Mapped Database). It is designed for extreme performance
#   and memory efficiency, making it suitable for processing terabyte-scale
#   datasets that do not fit into RAM.
#
#   It employs a multi-process producer-consumer architecture to achieve true
#   parallelism and uses memory-mapped files (`mmap`) by default for the
#   fastest possible I/O. A fallback to the robust Bio.SeqIO parser is
#   available for compatibility with non-standard FASTA files.
#
# INPUT:
#   - A FASTA file (--fasta_file) containing biological sequences.
#
# OUTPUT:
#   - An LMDB database directory (--lmdb_file) where each sequence is stored
#     with its index as the key.
#   - A metadata file (.meta.json) in the same location as the LMDB, used
#     for caching the total sequence count to speed up subsequent runs.
#
# USAGE:
#
#   1. Default High-Performance (mmap) Mode:
#      python fasta2lmdb.py \
#          --fasta_file /path/to/uniref50.fasta \
#          --lmdb_file /path/to/uniref50_lmdb \
#          --processes 24
#
#   2. Robust Fallback (SeqIO) Mode:
#      (Use if the default mode fails due to unusual FASTA formatting)
#      python fasta2lmdb.py \
#          --fasta_file /path/to/special.fasta \
#          --lmdb_file /path/to/special_lmdb \
#          --use_seqio
#
# ==============================================================================

import argparse
import json
import mmap
import multiprocessing as mp
import os
import queue
import shutil

import lmdb
from Bio import SeqIO
from tqdm import tqdm

# ==============================================================================
#  Parser functions (producer part, unchanged)
# ==============================================================================


def get_sequence_count_mmap(fasta_file: str, lmdb_file: str) -> int:
    """Fast sequence counting using mmap"""
    meta_file = os.path.splitext(lmdb_file)[0] + ".meta.json"
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            if (
                meta_data.get("fasta_file") == fasta_file
                and meta_data.get("parser_used") == "mmap"
            ):
                print(
                    f"Loaded sequence count from cache (mmap): {meta_data['total_sequences']}"
                )
                return meta_data["total_sequences"]
        except (json.JSONDecodeError, KeyError):
            pass

    print("Counting total sequences with mmap (fast)...")
    with open(fasta_file, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Fixed: Use manual counting instead of non-existent count method
            count = 0
            pos = 0
            while True:
                pos = mm.find(b">", pos)
                if pos == -1:
                    break
                count += 1
                pos += 1
            total_sequences = count

    print(f"Found {total_sequences} sequences.")
    with open(meta_file, "w") as f:
        json.dump(
            {
                "fasta_file": fasta_file,
                "total_sequences": total_sequences,
                "parser_used": "mmap",
            },
            f,
        )
    return total_sequences


def mmap_fasta_parser(mm: mmap.mmap):
    positions = []
    pos = mm.find(b">")
    while pos != -1:
        positions.append(pos)
        pos = mm.find(b">", pos + 1)

    for i in range(len(positions)):
        start_pos = positions[i]
        end_pos = positions[i + 1] if i + 1 < len(positions) else len(mm)
        record_bytes = mm[start_pos:end_pos]
        header_end_pos = record_bytes.find(b"\n")
        header_bytes = (
            record_bytes[1:header_end_pos] if header_end_pos != -1 else record_bytes[1:]
        )
        seq_bytes = record_bytes[header_end_pos + 1 :] if header_end_pos != -1 else b""
        record_id = header_bytes.split(b" ", 1)[0].decode(errors="ignore")
        sequence = seq_bytes.replace(b"\n", b"").decode(errors="ignore")
        yield record_id, sequence


def read_fasta_in_chunks_mmap(fasta_file: str, chunk_size: int):
    chunk = []
    idx = 0
    with open(fasta_file, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for record_id, sequence in mmap_fasta_parser(mm):
                chunk.append((idx, record_id, sequence))
                idx += 1
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


# --- SeqIO version ---


def get_sequence_count_seqio(fasta_file: str, lmdb_file: str) -> int:
    """Count sequences using Bio.SeqIO"""
    meta_file = os.path.splitext(lmdb_file)[0] + ".meta.json"
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            if (
                meta_data.get("fasta_file") == fasta_file
                and meta_data.get("parser_used") == "seqio"
            ):
                print(
                    f"Loaded sequence count from cache (SeqIO): {meta_data['total_sequences']}"
                )
                return meta_data["total_sequences"]
        except (json.JSONDecodeError, KeyError):
            pass

    print("Counting total sequences with Bio.SeqIO (slower but robust)...")
    total_sequences = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    print(f"Found {total_sequences} sequences.")
    with open(meta_file, "w") as f:
        json.dump(
            {
                "fasta_file": fasta_file,
                "total_sequences": total_sequences,
                "parser_used": "seqio",
            },
            f,
        )
    return total_sequences


def read_fasta_in_chunks_seqio(fasta_file: str, total_sequences: int, chunk_size: int):
    chunk = []
    idx = 0
    with tqdm(total=total_sequences, desc="Reading FASTA (SeqIO)", unit="seq") as pbar:
        for record in SeqIO.parse(fasta_file, "fasta"):
            chunk.append((idx, record.id, str(record.seq)))
            idx += 1
            pbar.update(1)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


# ==============================================================================
#  New architecture: Producer -> Worker -> Writer
# ==============================================================================


def producer(
    task_queue: mp.Queue,
    fasta_file: str,
    chunk_size: int,
    num_workers: int,
    use_seqio: bool,
    total_sequences: int,  # Only needed when use_seqio=True
):
    """
    Producer: Reads the FASTA file, creates data chunks, and puts them into the task queue.
    Basically the same as the original version, just sends a termination signal to 'num_workers'.
    """
    parser_mode = "SeqIO" if use_seqio else "mmap"
    print(f"[Producer] Starting to read FASTA (mode: {parser_mode})...")

    if use_seqio:
        chunk_generator = read_fasta_in_chunks_seqio(
            fasta_file, total_sequences, chunk_size
        )
    else:
        chunk_generator = read_fasta_in_chunks_mmap(fasta_file, chunk_size)

    for i, chunk in enumerate(chunk_generator):
        task_queue.put((i, chunk))

    print("[Producer] All chunks have been put into the queue.")
    # Send termination signals to all worker processes
    for _ in range(num_workers):
        task_queue.put(None)
    print("[Producer] Sent termination signals to workers. Exiting.")


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue):
    """
    Worker: Gets data chunks from the task queue, processes them (encodes),
    and then puts the list of prepared key-value pairs into the result queue.
    **This process does not perform any disk write operations.**
    """
    while True:
        task = task_queue.get()
        if task is None:
            # Job done, exit the loop
            result_queue.put(None)  # Send termination signal to writer
            break

        _chunk_id, chunk_data = task

        # Prepare the data to be written in memory
        records_to_write = []
        for idx, record_id, sequence in chunk_data:
            key = f"{idx}".encode("utf-8")
            value = json.dumps({"id": record_id, "sequence": sequence}).encode("utf-8")
            records_to_write.append((key, value))

        # Put the processed data chunk into the result queue
        result_queue.put(records_to_write)


def writer_process(
    result_queue: mp.Queue, lmdb_file: str, total_sequences: int, num_workers: int
):
    """
    Writer: A dedicated process responsible for getting data from the result queue
    and batch writing it to the final LMDB file.
    **This is the only process that interacts with the final LMDB file.**
    """
    print("[Writer] Process started, waiting for data...")
    # Open the final LMDB file with optimized parameters
    env = lmdb.open(
        lmdb_file,
        map_size=27 * 1024 * 1024 * 1024,  # 27GB
        writemap=True,
        meminit=False,
        map_async=True,
    )

    processed_count = 0
    workers_done = 0
    with tqdm(total=total_sequences, desc="Writing to LMDB", unit="seq") as pbar:
        while workers_done < num_workers:
            try:
                results_chunk = result_queue.get(timeout=120)  # Set a timeout just in case
                if results_chunk is None:
                    workers_done += 1
                    continue

                # Batch write data using a transaction
                with env.begin(write=True) as txn:
                    # Use putmulti for best performance
                    cursor = txn.cursor()
                    cursor.putmulti(results_chunk)

                # Update the progress bar
                update_amount = len(results_chunk)
                processed_count += update_amount
                pbar.update(update_amount)
            except queue.Empty:
                print(
                    "[Writer] Warning: Result queue was empty for 120 seconds. Check for worker errors."
                )
                # If all sequences have been processed, it's safe to exit
                if processed_count >= total_sequences:
                    break

    # Make sure the progress bar reaches 100%
    if pbar.n < total_sequences:
        pbar.n = total_sequences
        pbar.refresh()

    env.close()
    print("\n[Writer] All data has been written to the database. Exiting.")


# ==============================================================================
#  Main coordination function
# ==============================================================================


def fasta_to_lmdb(
    fasta_file, lmdb_file, num_processes=None, chunk_size=50000, use_seqio=False
):
    """Main coordination function, using Producer -> Worker -> Writer architecture."""
    if num_processes is None:
        # Leave one core for the producer and writer
        num_processes = max(1, mp.cpu_count() - 1)

    # If the lmdb file already exists, delete it first to avoid appending
    if os.path.exists(lmdb_file):
        shutil.rmtree(lmdb_file)
        print(f"Removed existing LMDB database: {lmdb_file}")

    parser_mode = "SeqIO" if use_seqio else "mmap (default)"
    print("--- FASTA to LMDB Conversion (Optimized Architecture) ---")
    print(f"Worker processes: {num_processes}")
    print(f"Sequence chunk size: {chunk_size}")
    print(f"Parser mode: {parser_mode}")
    print("-------------------------------------------------------")

    # 1. Count
    if use_seqio:
        total_sequences = get_sequence_count_seqio(fasta_file, lmdb_file)
    else:
        total_sequences = get_sequence_count_mmap(fasta_file, lmdb_file)

    if total_sequences == 0:
        print("No sequences found. Exiting.")
        return

    # 2. Create processes and queues
    with mp.Manager() as manager:
        task_queue = manager.Queue(maxsize=num_processes * 2)
        result_queue = manager.Queue(maxsize=num_processes * 2)

        # 3. Start all processes
        # Start the producer
        producer_proc = mp.Process(
            target=producer,
            args=(
                task_queue,
                fasta_file,
                chunk_size,
                num_processes,
                use_seqio,
                total_sequences,
            ),
        )
        producer_proc.start()

        # Start the single writer
        writer_proc = mp.Process(
            target=writer_process,
            args=(result_queue, lmdb_file, total_sequences, num_processes),
        )
        writer_proc.start()

        # Start the worker pool
        worker_procs = []
        for _ in range(num_processes):
            p = mp.Process(target=worker_process, args=(task_queue, result_queue))
            p.start()
            worker_procs.append(p)

        # 4. Wait for all processes to complete
        producer_proc.join()
        print("[Main] Producer has finished.")

        for p in worker_procs:
            p.join()
        print("[Main] All workers have finished.")

        writer_proc.join()
        print("[Main] Writer has finished.")

        print(
            f"\nFASTA file {fasta_file} successfully converted to LMDB database: {lmdb_file}"
        )
        print(f"Processed {total_sequences} sequences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert FASTA to LMDB using an optimized Producer-Worker-Writer model."
    )
    parser.add_argument(
        "--fasta_file", type=str, required=True, help="Input FASTA file path"
    )
    parser.add_argument(
        "--lmdb_file",
        type=str,
        required=True,
        help="Output LMDB file path (will be overwritten if exists)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to CPU count - 1.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=50000, help="Number of sequences per chunk"
    )
    parser.add_argument(
        "--use_seqio",
        action="store_true",
        help="Use the Bio.SeqIO parser. Slower but potentially more robust for non-standard FASTA files. Default is the fast mmap parser.",
    )

    args = parser.parse_args()

    fasta_to_lmdb(
        args.fasta_file, args.lmdb_file, args.processes, args.chunk_size, args.use_seqio
    )
