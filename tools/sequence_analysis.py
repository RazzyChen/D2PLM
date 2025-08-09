import argparse
import os
import sys
import warnings
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def analyze_sequences(df: pd.DataFrame, pbar: tqdm = None) -> Dict[str, Any]:
    """Analyze sequences in the DataFrame."""
    sequences = df["sequence"].tolist()
    lengths = []
    min_seq = ""
    max_seq = ""
    min_len = float('inf')
    max_len = 0
    
    for seq in sequences:
        seq_len = len(seq)
        lengths.append(seq_len)
        
        if seq_len < min_len:
            min_len = seq_len
            min_seq = seq
            
        if seq_len > max_len:
            max_len = seq_len
            max_seq = seq
            
        if pbar:
            pbar.update(1)
            
    return {
        "count": len(sequences),
        "lengths": lengths,
        "mean_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "variance": np.var(lengths),
        "min_length": min_len,
        "max_length": max_len,
        "min_sequence": min_seq,
        "max_sequence": max_seq
    }


def analyze_length_distribution(lengths: List[int]) -> Dict[str, Any]:
    """Analyze the distribution of sequence lengths."""
    # Calculate percentiles
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = np.percentile(lengths, percentiles)
    
    # Create length interval statistics
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, float('inf')]
    bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-2)]
    bin_labels.append(f"≥{bins[-2]}")
    
    hist, _ = np.histogram(lengths, bins=bins)
    distribution = dict(zip(bin_labels, hist))
    
    # Calculate percentages
    total_sequences = len(lengths)
    distribution_percentage = {
        bin_label: (count/total_sequences * 100) 
        for bin_label, count in distribution.items()
    }
    
    return {
        "percentiles": dict(zip(percentiles, percentile_values)),
        "distribution": distribution,
        "distribution_percentage": distribution_percentage
    }


def plot_length_histogram(lengths: List[int], title: str) -> None:
    """Plot histogram of sequence lengths and save as PNG."""
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.title(title)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()


def count_tokens_esm2(sequences: List[str], pbar: tqdm = None) -> int:
    """Count total number of tokens using ESM2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    total_tokens = 0
    for seq in sequences:
        tokens = tokenizer.encode(seq)
        total_tokens += len(tokens) - 2
        if pbar:
            pbar.update(1)
    return total_tokens


def print_sequence_stats(stats: Dict[str, Any], tokens: int, dataset_name: str = "") -> None:
    """Print sequence statistics in a formatted way."""
    if dataset_name:
        print(f"\nAnalyzing Dataset: {dataset_name}")
    
    print("\nBasic Statistics:")
    print(f"Number of sequences: {stats['count']:,}")
    print(f"Mean sequence length: {stats['mean_length']:.2f}")
    print(f"Median sequence length: {stats['median_length']:.2f}")
    print(f"Variance of sequence length: {stats['variance']:.2f}")
    print(f"Shortest sequence length: {stats['min_length']}")
    print(f"Longest sequence length: {stats['max_length']}")
    print(f"Total number of tokens (ESM2 tokenizer): {tokens:,}")


def print_length_distribution(stats: Dict[str, Any]) -> None:
    """Print length distribution statistics."""
    print("\nPercentile Distribution:")
    print(f"{'Percentile':>10} {'Length':>10}")
    print("-" * 22)
    for percentile, value in stats["percentiles"].items():
        print(f"{percentile:>10}th {value:>10.1f}")
    
    print("\nLength Range Distribution:")
    print(f"{'Range':<12} {'Count':>10} {'Percentage':>12}")
    print("-" * 34)
    for bin_label in stats["distribution"].keys():
        count = stats["distribution"][bin_label]
        percentage = stats["distribution_percentage"][bin_label]
        print(f"{bin_label:<12} {count:>10,} {percentage:>11.2f}%")


def main(csv_files: List[str]) -> None:
    """Main function to process CSV files and analyze sequences."""
    print("Starting sequence analysis...")
    
    if len(csv_files) == 1:
        df = load_data(csv_files[0])

        with tqdm(total=len(df) * 2, desc="Processing") as pbar:
            stats = analyze_sequences(df, pbar)
            tokens = count_tokens_esm2(df["sequence"].tolist(), pbar)

        # 打印基本统计信息
        print_sequence_stats(stats, tokens, csv_files[0])
        
        # 分析和打印长度分布
        distribution_stats = analyze_length_distribution(stats["lengths"])
        print_length_distribution(distribution_stats)
        
        # 绘制分布图
        plot_length_histogram(stats["lengths"], "Sequence Length Distribution")

    else:
        # 处理多个文件
        all_data = pd.DataFrame()
        
        for i, file in enumerate(csv_files, 1):
            df = load_data(file)
            
            with tqdm(total=len(df) * 2, desc=f"Processing file {i}/{len(csv_files)}") as pbar:
                stats = analyze_sequences(df, pbar)
                tokens = count_tokens_esm2(df["sequence"].tolist(), pbar)
            
            print_sequence_stats(stats, tokens, file)
            distribution_stats = analyze_length_distribution(stats["lengths"])
            print_length_distribution(distribution_stats)
            plot_length_histogram(stats["lengths"], f"Dataset {i} Length Distribution")
            
            df["dataset"] = f"Dataset {i}"
            all_data = pd.concat([all_data, df], ignore_index=True)

        # 保存并分析合并数据
        all_data.to_csv("combined_datasets.csv", index=False)
        print("\nAnalyzing Combined Dataset...")
        
        combined_stats = analyze_sequences(all_data)
        combined_tokens = count_tokens_esm2(all_data["sequence"].tolist())
        print_sequence_stats(combined_stats, combined_tokens, "Combined Dataset")
        
        combined_distribution = analyze_length_distribution(combined_stats["lengths"])
        print_length_distribution(combined_distribution)
        plot_length_histogram(combined_stats["lengths"], "Combined Dataset Distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze protein sequence datasets with detailed length distribution."
    )
    parser.add_argument("csv_files", nargs="+", help="CSV file(s) to analyze")
    args = parser.parse_args()

    main(args.csv_files)