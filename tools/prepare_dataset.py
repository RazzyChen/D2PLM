import os
import subprocess
import argparse
import random
import lmdb
import json
import pickle
from tqdm import tqdm

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
                        yield key.decode('utf-8'), entry["sequence"]
                except Exception:
                    continue

def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")

def write_fasta(sequences, fasta_path):
    """Writes a list of (id, sequence) tuples to a FASTA file."""
    with open(fasta_path, 'w') as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ESM model training. This includes splitting, homology reduction, and creating final LMDB datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raw_lmdb_path", type=str, required=True, help="Path to the input raw LMDB dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all processed files and final LMDBs.")
    parser.add_argument("--val_split_ratio", type=float, default=0.005, help="Ratio of the dataset to use for validation (as in ESM-2 paper).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--min_seq_id", type=float, default=0.5, help="Minimum sequence identity for mmseqs2 search.")

    args = parser.parse_args()

    print(f"--- Starting Dataset Preparation ---")
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
        "mmseqs", "search", train_db_path, val_db_path, search_result_path, tmp_dir,
        "--min-seq-id", str(args.min_seq_id),
        "--alignment-mode", "3", "-s", "7", "-c", "0.8", "--cov-mode", "0"
    ]
    run_command(mmseqs_search_cmd)

    tsv_result_path = os.path.join(tmp_dir, "search_result.tsv")
    run_command(["mmseqs", "createtsv", train_db_path, val_db_path, search_result_path, tsv_result_path])

    sequences_to_remove = set()
    with open(tsv_result_path, 'r') as f:
        for line in f:
            query_id = line.split('\t')[0]
            sequences_to_remove.add(query_id)
            
    print(f"Found {len(sequences_to_remove)} training sequences with homology to validation set. Removing them.")

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

    run_command(["python", converter_script, "--fasta_file", final_train_fasta, "--lmdb_path", final_train_lmdb])
    run_command(["python", converter_script, "--fasta_file", final_val_fasta, "--lmdb_path", final_val_lmdb])

    print("\n--- Dataset Preparation Complete! ---")
    print(f"Final Training LMDB path: {final_train_lmdb}")
    print(f"Final Validation LMDB path: {final_val_lmdb}")
    print("You can now update your train_config.yaml to point to these new LMDB directories.")

if __name__ == "__main__":
    main()
