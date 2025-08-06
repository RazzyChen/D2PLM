import argparse
import os

from tqdm import tqdm


def filter_fasta_by_length(input_file, output_file, max_length=1024, min_length=0):
    """
    Filters a FASTA file to include only sequences with length within the specified range.

    Args:
        input_file (str): Path to the input FASTA file.
        output_file (str): Path to the output FASTA file.
        max_length (int): Maximum sequence length to include (inclusive).
        min_length (int): Minimum sequence length to include (inclusive).
    """
    print(f"Starting to filter '{input_file}'...")
    print(
        f"Sequences with length between {min_length} and {max_length} residues will be saved to '{output_file}'."
    )

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        header = None
        sequence = []
        total_sequences = 0
        kept_sequences = 0

        # Get file size for progress bar
        file_size = os.path.getsize(input_file)

        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Processing"
        ) as pbar:
            for line in infile:
                pbar.update(len(line.encode("utf-8")))
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    total_sequences += 1
                    # Process the previous sequence
                    if header and min_length <= len("".join(sequence)) <= max_length:
                        outfile.write(header + "\n")
                        outfile.write("".join(sequence) + "\n")
                        kept_sequences += 1

                    # Start a new sequence
                    header = line
                    sequence = []
                else:
                    sequence.append(line)

            # Don't forget to process the last sequence in the file
            if header and min_length <= len("".join(sequence)) <= max_length:
                outfile.write(header + "\n")
                outfile.write("".join(sequence) + "\n")
                kept_sequences += 1

    print("\nFiltering complete.")
    print(f"Total sequences processed: {total_sequences}")
    print(f"Sequences kept: {kept_sequences}")
    print(f"Filtered file saved to: '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a FASTA file by sequence length."
    )
    parser.add_argument(
        "input_fasta",
        type=str,
        help="Path to the input FASTA file (e.g., UniRef50.fasta)",
    )
    parser.add_argument(
        "output_fasta",
        type=str,
        help="Path to save the filtered output FASTA file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length to keep (inclusive, default: 1024)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimum sequence length to keep (inclusive, default: 0)",
    )
    args = parser.parse_args()

    filter_fasta_by_length(
        args.input_fasta, args.output_fasta, args.max_length, args.min_length
    )
