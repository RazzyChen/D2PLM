import argparse
import json
import os
import pickle
import random
import subprocess

import lmdb
from tqdm import tqdm


# Copied from model/dataloader/DataPipe.py for self-containment
class OptimizedLMDBReader:
    """Optimized LMDB reader."""

    def __init__(self, mdb_file_path: str):
        self.mdb_file_path = mdb_file_path
        self._env = None


# Copied from model/dataloader/DataPipe.py for self-containment
class OptimizedLMDBReader:
    """Optimized LMDB reader."""

    def __init__(self, mdb_file_path: str):
        self.mdb_file_path = mdb_file_path
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.mdb_file_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=128,
            )
        return self._env

    def get_sequences(self):
        """Generator that yields all sequences from the LMDB file."""
        env = self._get_env()
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    try:
                        entry = json.loads(value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        entry = pickle.loads(value)

                    if (
                        isinstance(entry, dict)
                        and "sequence" in entry
                        and isinstance(entry.get("sequence"), str)
                    ):
                        # The key is often the sequence ID
                        yield key.decode("utf-8"), entry["sequence"]
                except Exception:
                    continue


def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")


def write_fasta(sequences, fasta_path):
    """Writes a list of (id, sequence) tuples to a FASTA file."""
    with open(fasta_path, "w") as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")


def check_gpu_available():
    """Check if CUDA/GPU is available for MMseqs2."""
    try:
        # Check if nvidia-smi is available and working
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✓ nvidia-smi detected - GPU available")
            # Additional check: try to see if there are available GPUs
            lines = result.stdout.strip().split("\n")
            gpu_found = any("NVIDIA" in line for line in lines)
            if gpu_found:
                print("✓ NVIDIA GPU(s) detected and accessible")
                return True
            else:
                print("⚠ nvidia-smi works but no GPUs found")
                return False
        else:
            print("⚠ nvidia-smi failed - no GPU available")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"⚠ GPU check failed: {str(e)} - falling back to CPU")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ESM model training. This includes splitting, homology reduction, and creating final LMDB datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_lmdb_path",
        type=str,
        required=True,
        help="Path to the input raw LMDB dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save all processed files and final LMDBs.",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.005,
        help="Ratio of the dataset to use for validation (as in ESM-2 paper).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting."
    )
    parser.add_argument(
        "--min_seq_id",
        type=float,
        default=0.5,
        help="Minimum sequence identity for mmseqs2 search.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU-only mode even if GPU is available.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of CPU threads to use for MMseqs2 operations.",
    )

    args = parser.parse_args()

    print("--- Starting Dataset Preparation ---")
    print(f"Args: {args}")

    # Automatic GPU detection (default behavior)
    use_gpu = False
    if args.force_cpu:
        print("🖥️  CPU-only mode forced by user")
        use_gpu = False
    else:
        print("🔍 Checking GPU availability...")
        gpu_available = check_gpu_available()
        if gpu_available:
            print("🚀 GPU mode enabled - using hardware acceleration")
            use_gpu = True
        else:
            print("💻 GPU not available - using CPU mode")
            use_gpu = False

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # --- 1. Load and Split Data ---
    print("\n[Step 1/6] Loading sequences from raw LMDB and splitting...")
    reader = OptimizedLMDBReader(args.raw_lmdb_path)
    all_sequences = list(tqdm(reader.get_sequences(), desc="Loading sequences"))

    random.seed(args.seed)
    random.shuffle(all_sequences)

    val_size = int(len(all_sequences) * args.val_split_ratio)
    val_set = all_sequences[:val_size]
    train_set = all_sequences[val_size:]

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    # --- 2. Write to Intermediate FASTA Files ---
    print("\n[Step 2/6] Writing to intermediate FASTA files...")
    initial_train_fasta = os.path.join(tmp_dir, "initial_train.fasta")
    val_fasta = os.path.join(tmp_dir, "validation.fasta")
    write_fasta(train_set, initial_train_fasta)
    write_fasta(val_set, val_fasta)
    print(f"Initial training FASTA: {initial_train_fasta}")
    print(f"Validation FASTA: {val_fasta}")

    # --- 3. Create MMseqs2 Databases ---
    print("\n[Step 3/6] Creating MMseqs2 databases...")

    if use_gpu:
        print("Creating databases with GPU optimization...")
        # For GPU mode: create regular DB first, then convert to GPU format
        val_db_path = os.path.join(tmp_dir, "val_db")
        train_db_path = os.path.join(tmp_dir, "train_db")
        val_db_gpu_path = os.path.join(tmp_dir, "val_db_gpu")
        train_db_gpu_path = os.path.join(tmp_dir, "train_db_gpu")

        print("  1/4: Creating standard databases...")
        run_command(
            [
                "mmseqs",
                "createdb",
                val_fasta,
                val_db_path,
                "--threads",
                str(args.threads),
            ]
        )
        run_command(
            [
                "mmseqs",
                "createdb",
                initial_train_fasta,
                train_db_path,
                "--threads",
                str(args.threads),
            ]
        )

        print("  2/4: Converting to GPU-compatible format...")
        run_command(["mmseqs", "makepaddedseqdb", val_db_path, val_db_gpu_path])
        run_command(["mmseqs", "makepaddedseqdb", train_db_path, train_db_gpu_path])

        print("  3/4: Creating memory-optimized indices...")
        run_command(
            [
                "mmseqs",
                "createindex",
                train_db_gpu_path,
                tmp_dir,
                "--index-subset",
                "2",  # Omit k-mer index to save memory
            ]
        )
        run_command(
            ["mmseqs", "createindex", val_db_gpu_path, tmp_dir, "--index-subset", "2"]
        )

        print("  4/4: Cleaning up intermediate files...")
        # Remove standard databases to save space (optional)
        # run_command(["mmseqs", "rmdb", val_db_path])
        # run_command(["mmseqs", "rmdb", train_db_path])

        # Use GPU databases for search
        search_train_db = train_db_gpu_path
        search_val_db = val_db_gpu_path
        print(f"GPU databases ready: {search_train_db}, {search_val_db}")

    else:
        print("Creating databases for CPU-only mode...")
        val_db_path = os.path.join(tmp_dir, "val_db")
        train_db_path = os.path.join(tmp_dir, "train_db")

        run_command(
            [
                "mmseqs",
                "createdb",
                val_fasta,
                val_db_path,
                "--threads",
                str(args.threads),
            ]
        )
        run_command(
            [
                "mmseqs",
                "createdb",
                initial_train_fasta,
                train_db_path,
                "--threads",
                str(args.threads),
            ]
        )

        # Use regular databases for CPU search
        search_train_db = train_db_path
        search_val_db = val_db_path
        print(f"CPU databases ready: {search_train_db}, {search_val_db}")

    # --- 4. Homology Reduction with MMseqs2 ---
    print("\n[Step 4/6] Performing homology reduction with MMseqs2...")
    search_result_path = os.path.join(tmp_dir, "search_result_db")

    # Build search command based on GPU availability
    mmseqs_search_cmd = [
        "mmseqs",
        "search",
        search_train_db,
        search_val_db,
        search_result_path,
        tmp_dir,
        "--min-seq-id",
        str(args.min_seq_id),
        "--alignment-mode",
        "3",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--threads",
        str(args.threads),
    ]

    if use_gpu:
        mmseqs_search_cmd.extend(
            [
                "--gpu",
                "1",
                "--prefilter-mode",
                "1",  # Use ungappedprefilter for GPU
                "--max-seqs",
                "300",  # Consistent with your original ungappedprefilter call
            ]
        )
    else:
        mmseqs_search_cmd.extend(
            [
                "-s",
                "7",  # Sensitivity parameter (only works in CPU mode)
            ]
        )

    run_command(mmseqs_search_cmd)

    # Convert results to TSV
    tsv_result_path = os.path.join(tmp_dir, "search_result.tsv")
    run_command(
        [
            "mmseqs",
            "createtsv",
            search_train_db,
            search_val_db,
            search_result_path,
            tsv_result_path,
        ]
    )

    # Parse results and identify sequences to remove
    sequences_to_remove = set()
    if os.path.exists(tsv_result_path):
        with open(tsv_result_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 0:
                    query_id = parts[0]
                    sequences_to_remove.add(query_id)

    print(
        f"Found {len(sequences_to_remove)} training sequences with homology to validation set. Removing them."
    )

    # --- 5. Filter Training Set ---
    print("\n[Step 5/6] Filtering training set...")
    final_train_set = [item for item in train_set if item[0] not in sequences_to_remove]
    final_train_fasta = os.path.join(args.output_dir, "train_final.fasta")
    write_fasta(final_train_set, final_train_fasta)

    final_val_fasta = os.path.join(args.output_dir, "validation_final.fasta")
    write_fasta(val_set, final_val_fasta)

    print(f"Final filtered training set size: {len(final_train_set)}")
    print(f"Removed {len(train_set) - len(final_train_set)} sequences due to homology")
    print(f"Final training FASTA saved to: {final_train_fasta}")
    print(f"Final validation FASTA saved to: {final_val_fasta}")

    # --- 6. Convert to Final LMDB Datasets ---
    print("\n[Step 6/6] Converting final FASTA files to LMDB datasets...")
    final_train_lmdb = os.path.join(args.output_dir, "train_lmdb")
    final_val_lmdb = os.path.join(args.output_dir, "validation_lmdb")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    converter_script = os.path.join(script_dir, "fasta2lmdb_optimized.py")

    run_command(
        [
            "python",
            converter_script,
            "--fasta_file",
            final_train_fasta,
            "--lmdb_path",
            final_train_lmdb,
        ]
    )
    run_command(
        [
            "python",
            converter_script,
            "--fasta_file",
            final_val_fasta,
            "--lmdb_path",
            final_val_lmdb,
        ]
    )

    print("\n--- Dataset Preparation Complete! ---")
    print(f"Final Training LMDB path: {final_train_lmdb}")
    print(f"Final Validation LMDB path: {final_val_lmdb}")
    print(f"Mode used: {'GPU-accelerated' if use_gpu else 'CPU-only'}")
    print(
        "You can now update your train_config.yaml to point to these new LMDB directories."
    )

    # Optional cleanup
    print(f"\nTemporary files are stored in: {tmp_dir}")
    print("You can delete this directory to save space after verifying the results.")


if __name__ == "__main__":
    main()

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.mdb_file_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=128,
            )
        return self._env

    def get_sequences(self):
        """Generator that yields all sequences from the LMDB file."""
        env = self._get_env()
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    try:
                        entry = json.loads(value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        entry = pickle.loads(value)

                    if (
                        isinstance(entry, dict)
                        and "sequence" in entry
                        and isinstance(entry.get("sequence"), str)
                    ):
                        # The key is often the sequence ID
                        yield key.decode("utf-8"), entry["sequence"]
                except Exception:
                    continue


def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")


def write_fasta(sequences, fasta_path):
    """Writes a list of (id, sequence) tuples to a FASTA file."""
    with open(fasta_path, "w") as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ESM model training. This includes splitting, homology reduction, and creating final LMDB datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_lmdb_path",
        type=str,
        required=True,
        help="Path to the input raw LMDB dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save all processed files and final LMDBs.",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.005,
        help="Ratio of the dataset to use for validation (as in ESM-2 paper).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting."
    )
    parser.add_argument(
        "--min_seq_id",
        type=float,
        default=0.5,
        help="Minimum sequence identity for mmseqs2 search.",
    )

    args = parser.parse_args()

    print("--- Starting Dataset Preparation ---")
    print(f"Args: {args}")

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # --- 1. Load and Split Data ---
    print("\n[Step 1/5] Loading sequences from raw LMDB and splitting...")
    reader = OptimizedLMDBReader(args.raw_lmdb_path)
    all_sequences = list(tqdm(reader.get_sequences(), desc="Loading sequences"))

    random.seed(args.seed)
    random.shuffle(all_sequences)

    val_size = int(len(all_sequences) * args.val_split_ratio)
    val_set = all_sequences[:val_size]
    train_set = all_sequences[val_size:]

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    # --- 2. Write to Intermediate FASTA Files ---
    print("\n[Step 2/5] Writing to intermediate FASTA files...")
    initial_train_fasta = os.path.join(tmp_dir, "initial_train.fasta")
    val_fasta = os.path.join(tmp_dir, "validation.fasta")
    write_fasta(train_set, initial_train_fasta)
    write_fasta(val_set, val_fasta)
    print(f"Initial training FASTA: {initial_train_fasta}")
    print(f"Validation FASTA: {val_fasta}")

    # --- 3. Homology Reduction with MMseqs2 ---
    print("\n[Step 3/5] Performing homology reduction with MMseqs2...")
    val_db_path = os.path.join(tmp_dir, "val_db")
    run_command(["mmseqs", "createdb", val_fasta, val_db_path])

    train_db_path = os.path.join(tmp_dir, "train_db")
    run_command(["mmseqs", "createdb", initial_train_fasta, train_db_path])

    search_result_path = os.path.join(tmp_dir, "search_result_db")
    mmseqs_search_cmd = [
        "mmseqs",
        "search",
        train_db_path,
        val_db_path,
        search_result_path,
        tmp_dir,
        "--min-seq-id",
        str(args.min_seq_id),
        "--alignment-mode",
        "3",
        "-s",
        "7",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
    ]
    run_command(mmseqs_search_cmd)

    tsv_result_path = os.path.join(tmp_dir, "search_result.tsv")
    run_command(
        [
            "mmseqs",
            "createtsv",
            train_db_path,
            val_db_path,
            search_result_path,
            tsv_result_path,
        ]
    )

    sequences_to_remove = set()
    with open(tsv_result_path, "r") as f:
        for line in f:
            query_id = line.split("\t")[0]
            sequences_to_remove.add(query_id)

    print(
        f"Found {len(sequences_to_remove)} training sequences with homology to validation set. Removing them."
    )

    # --- 4. Filter Training Set ---
    print("\n[Step 4/5] Filtering training set...")
    final_train_set = [item for item in train_set if item[0] not in sequences_to_remove]
    final_train_fasta = os.path.join(args.output_dir, "train_final.fasta")
    write_fasta(final_train_set, final_train_fasta)

    final_val_fasta = os.path.join(args.output_dir, "validation_final.fasta")
    write_fasta(val_set, final_val_fasta)

    print(f"Final filtered training set size: {len(final_train_set)}")
    print(f"Final training FASTA saved to: {final_train_fasta}")
    print(f"Final validation FASTA saved to: {final_val_fasta}")

    # --- 5. Convert to Final LMDB Datasets ---
    print("\n[Step 5/5] Converting final FASTA files to LMDB datasets...")
    final_train_lmdb = os.path.join(args.output_dir, "train_lmdb")
    final_val_lmdb = os.path.join(args.output_dir, "validation_lmdb")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    converter_script = os.path.join(script_dir, "fasta2lmdb_optimized.py")

    run_command(
        [
            "python",
            converter_script,
            "--fasta_file",
            final_train_fasta,
            "--lmdb_path",
            final_train_lmdb,
        ]
    )
    run_command(
        [
            "python",
            converter_script,
            "--fasta_file",
            final_val_fasta,
            "--lmdb_path",
            final_val_lmdb,
        ]
    )

    print("\n--- Dataset Preparation Complete! ---")
    print(f"Final Training LMDB path: {final_train_lmdb}")
    print(f"Final Validation LMDB path: {final_val_lmdb}")
    print(
        "You can now update your train_config.yaml to point to these new LMDB directories."
    )


if __name__ == "__main__":
    main()
