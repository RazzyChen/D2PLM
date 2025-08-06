#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Benchmarking Tool for ESM2AE

This script provides comprehensive benchmarking capabilities to measure:
1. Data loading performance (LMDB reading, tokenization)
2. Model inference performance 
3. Training throughput
4. Memory usage optimization
5. I/O bottleneck identification

Usage:
    python tools/performance_benchmark.py --config train_config/trainer_config.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, EsmConfig
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone.esm2ae import ESM2AE
from model.dataloader.DataPipe import load_and_preprocess_data, OptimizedLMDBReader


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite for ESM2AE."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Benchmarking on device: {self.device}")
    
    def get_memory_stats(self):
        """Get current memory statistics."""
        stats = {
            "cpu_memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_memory_percent": psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_memory_cached_mb": torch.cuda.memory_cached() / 1024 / 1024,
            })
        
        return stats
    
    def benchmark_lmdb_reading(self, lmdb_path: str, num_samples: int = 10000):
        """Benchmark LMDB reading performance."""
        print(f"Benchmarking LMDB reading from: {lmdb_path}")
        
        if not os.path.exists(lmdb_path):
            print(f"LMDB path {lmdb_path} does not exist, skipping benchmark")
            return None
        
        # Test optimized vs standard reading
        results = {}
        
        # Optimized reading
        reader = OptimizedLMDBReader(lmdb_path)
        start_time = time.time()
        count = 0
        
        for batch in reader.prefetch_batch_generator(batch_size=1000):
            count += len(batch)
            if count >= num_samples:
                break
        
        optimized_time = time.time() - start_time
        optimized_throughput = count / optimized_time
        
        results = {
            "samples_read": count,
            "optimized_time_seconds": optimized_time,
            "optimized_throughput_samples_per_sec": optimized_throughput,
            "memory_stats": self.get_memory_stats()
        }
        
        print(f"LMDB Reading Results:")
        print(f"  Samples read: {count:,}")
        print(f"  Time: {optimized_time:.2f} seconds")
        print(f"  Throughput: {optimized_throughput:.2f} samples/sec")
        
        return results
    
    def benchmark_tokenization(self, sequences: list, tokenizer, max_lengths: list = [256, 512, 700]):
        """Benchmark tokenization performance across different sequence lengths."""
        print("Benchmarking tokenization performance...")
        
        results = {}
        
        for max_length in max_lengths:
            print(f"Testing max_length: {max_length}")
            
            # Test with different padding strategies
            strategies = ["longest", "max_length"]
            
            for strategy in strategies:
                start_time = time.time()
                memory_before = self.get_memory_stats()
                
                # Add space prefix for ESM tokenization
                processed_sequences = [" " + seq[:max_length-2] for seq in sequences]
                
                tokenized = tokenizer(
                    processed_sequences,
                    padding=strategy,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                tokenization_time = time.time() - start_time
                memory_after = self.get_memory_stats()
                
                throughput = len(sequences) / tokenization_time
                memory_used = memory_after["cpu_memory_mb"] - memory_before["cpu_memory_mb"]
                
                key = f"max_length_{max_length}_padding_{strategy}"
                results[key] = {
                    "time_seconds": tokenization_time,
                    "throughput_samples_per_sec": throughput,
                    "memory_used_mb": memory_used,
                    "output_shape": list(tokenized["input_ids"].shape),
                    "total_tokens": tokenized["input_ids"].numel()
                }
                
                print(f"  {strategy}: {throughput:.2f} samples/sec, {memory_used:.2f} MB")
        
        return results
    
    def benchmark_model_inference(self, model, tokenizer, batch_sizes: list = [1, 4, 8, 16], 
                                 sequence_lengths: list = [256, 512]):
        """Benchmark model inference performance."""
        print("Benchmarking model inference...")
        
        model.eval()
        results = {}
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                print(f"Testing batch_size: {batch_size}, seq_length: {seq_len}")
                
                # Create dummy input
                dummy_sequences = ["A" * seq_len] * batch_size
                tokenized = tokenizer(
                    [" " + seq for seq in dummy_sequences],
                    padding="longest",
                    truncation=True,
                    max_length=seq_len,
                    return_tensors="pt"
                ).to(self.device)
                
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(**tokenized)
                
                # Benchmark
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                memory_before = self.get_memory_stats()
                start_time = time.time()
                
                num_iterations = 20
                for _ in range(num_iterations):
                    with torch.no_grad():
                        outputs = model(**tokenized)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start_time
                memory_after = self.get_memory_stats()
                
                avg_time_per_batch = inference_time / num_iterations
                throughput = batch_size / avg_time_per_batch
                
                key = f"batch_{batch_size}_seqlen_{seq_len}"
                results[key] = {
                    "avg_time_per_batch_ms": avg_time_per_batch * 1000,
                    "throughput_samples_per_sec": throughput,
                    "memory_before_mb": memory_before.get("gpu_memory_allocated_mb", 0),
                    "memory_after_mb": memory_after.get("gpu_memory_allocated_mb", 0),
                    "memory_increase_mb": memory_after.get("gpu_memory_allocated_mb", 0) - 
                                        memory_before.get("gpu_memory_allocated_mb", 0),
                }
                
                print(f"  {avg_time_per_batch*1000:.2f} ms/batch, {throughput:.2f} samples/sec")
        
        return results
    
    def benchmark_data_loading_pipeline(self, data_path: str, tokenizer, cache_dir: str):
        """Benchmark the complete data loading pipeline."""
        print("Benchmarking complete data loading pipeline...")
        
        # Test different configurations
        configs = [
            {"max_length": 256, "batch_size": 1000, "num_proc": 4},
            {"max_length": 512, "batch_size": 1000, "num_proc": 4},
            {"max_length": 512, "batch_size": 2000, "num_proc": 8},
        ]
        
        results = {}
        
        for i, config in enumerate(configs):
            print(f"Testing configuration {i+1}: {config}")
            
            # Clear cache for fair comparison
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
            
            start_time = time.time()
            memory_before = self.get_memory_stats()
            
            try:
                dataset = load_and_preprocess_data(
                    data_path,
                    tokenizer,
                    cache_dir,
                    **config
                )
                
                loading_time = time.time() - start_time
                memory_after = self.get_memory_stats()
                
                # Test DataLoader performance
                dataloader = DataLoader(
                    dataset,
                    batch_size=16,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                dataloader_start = time.time()
                batch_count = 0
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 10:  # Test first 10 batches
                        break
                
                dataloader_time = time.time() - dataloader_start
                
                key = f"config_{i+1}"
                results[key] = {
                    "config": config,
                    "dataset_size": len(dataset),
                    "loading_time_seconds": loading_time,
                    "loading_throughput_samples_per_sec": len(dataset) / loading_time,
                    "memory_used_mb": memory_after["cpu_memory_mb"] - memory_before["cpu_memory_mb"],
                    "dataloader_time_per_10_batches": dataloader_time,
                    "dataloader_throughput_batches_per_sec": 10 / dataloader_time,
                }
                
                print(f"  Dataset size: {len(dataset):,}")
                print(f"  Loading time: {loading_time:.2f}s")
                print(f"  DataLoader throughput: {10/dataloader_time:.2f} batches/sec")
                
            except Exception as e:
                print(f"  Error in configuration {i+1}: {e}")
                results[key] = {"error": str(e), "config": config}
        
        return results
    
    def generate_report(self, output_dir: str = "benchmark_results"):
        """Generate a comprehensive performance report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate plots
        self._plot_results(output_dir)
        
        # Generate text report
        self._generate_text_report(output_dir)
        
        print(f"Benchmark results saved to: {output_dir}")
    
    def _plot_results(self, output_dir: str):
        """Generate performance plots."""
        plt.style.use('seaborn-v0_8')
        
        # Plot tokenization performance
        if "tokenization" in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            tokenization_data = self.results["tokenization"]
            
            # Throughput comparison
            throughputs = []
            labels = []
            for key, data in tokenization_data.items():
                if "throughput_samples_per_sec" in data:
                    throughputs.append(data["throughput_samples_per_sec"])
                    labels.append(key.replace("_", "\n"))
            
            ax1.bar(range(len(throughputs)), throughputs)
            ax1.set_xlabel("Configuration")
            ax1.set_ylabel("Throughput (samples/sec)")
            ax1.set_title("Tokenization Throughput")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            
            # Memory usage comparison
            memory_usage = []
            for key, data in tokenization_data.items():
                if "memory_used_mb" in data:
                    memory_usage.append(data["memory_used_mb"])
            
            ax2.bar(range(len(memory_usage)), memory_usage)
            ax2.set_xlabel("Configuration")
            ax2.set_ylabel("Memory Used (MB)")
            ax2.set_title("Tokenization Memory Usage")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "tokenization_performance.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot inference performance
        if "model_inference" in self.results:
            inference_data = self.results["model_inference"]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            batch_sizes = []
            throughputs_256 = []
            throughputs_512 = []
            
            for key, data in inference_data.items():
                if "batch_" in key and "seqlen_256" in key:
                    batch_size = int(key.split("_")[1])
                    batch_sizes.append(batch_size)
                    throughputs_256.append(data["throughput_samples_per_sec"])
                
                if "batch_" in key and "seqlen_512" in key:
                    batch_size = int(key.split("_")[1])
                    if batch_size in batch_sizes:
                        idx = batch_sizes.index(batch_size)
                        if len(throughputs_512) <= idx:
                            throughputs_512.extend([0] * (idx - len(throughputs_512) + 1))
                        throughputs_512[idx] = data["throughput_samples_per_sec"]
            
            x = np.arange(len(batch_sizes))
            width = 0.35
            
            ax.bar(x - width/2, throughputs_256, width, label='Seq Length 256', alpha=0.8)
            ax.bar(x + width/2, throughputs_512, width, label='Seq Length 512', alpha=0.8)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title('Model Inference Throughput by Batch Size')
            ax.set_xticks(x)
            ax.set_xticklabels(batch_sizes)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "inference_performance.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self, output_dir: str):
        """Generate a text summary report."""
        report_path = os.path.join(output_dir, "performance_report.txt")
        
        with open(report_path, "w") as f:
            f.write("ESM2AE Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Benchmark Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n\n")
            
            # System information
            f.write("System Information:\n")
            f.write(f"  CPU Count: {psutil.cpu_count()}\n")
            f.write(f"  Total Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB\n")
            if torch.cuda.is_available():
                f.write(f"  GPU: {torch.cuda.get_device_name()}\n")
                f.write(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB\n")
            f.write("\n")
            
            # Detailed results
            for benchmark_name, results in self.results.items():
                f.write(f"{benchmark_name.upper()} RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for subkey, subvalue in value.items():
                                f.write(f"    {subkey}: {subvalue}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    def run_full_benchmark(self, lmdb_path: str = None, num_samples: int = 10000):
        """Run the complete benchmark suite."""
        print("Starting comprehensive performance benchmark...")
        
        # Initialize model and tokenizer for benchmarks
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        config = EsmConfig.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model = ESM2AE(config).to(self.device)
        
        # 1. LMDB Reading Benchmark
        if lmdb_path and os.path.exists(lmdb_path):
            self.results["lmdb_reading"] = self.benchmark_lmdb_reading(lmdb_path, num_samples)
        
        # 2. Tokenization Benchmark
        # Generate sample sequences of varying lengths
        sample_sequences = [
            "MKFLILLFNILCLFPVLAADNHGTTGS" * (i + 1) for i in range(100)
        ]
        
        self.results["tokenization"] = self.benchmark_tokenization(
            sample_sequences, tokenizer
        )
        
        # 3. Model Inference Benchmark
        self.results["model_inference"] = self.benchmark_model_inference(
            model, tokenizer
        )
        
        # 4. Data Loading Pipeline Benchmark (if LMDB exists)
        if lmdb_path and os.path.exists(lmdb_path):
            cache_dir = "./benchmark_cache"
            self.results["data_loading_pipeline"] = self.benchmark_data_loading_pipeline(
                lmdb_path, tokenizer, cache_dir
            )
        
        print("Benchmark completed!")
        return self.results


def main():
    parser = argparse.ArgumentParser(description="ESM2AE Performance Benchmark")
    parser.add_argument("--lmdb_path", type=str, help="Path to LMDB dataset")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to benchmark")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark(args.lmdb_path, args.num_samples)
    
    # Generate report
    benchmark.generate_report(args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if "tokenization" in results:
        print("Tokenization Performance:")
        for key, data in results["tokenization"].items():
            if "throughput_samples_per_sec" in data:
                print(f"  {key}: {data['throughput_samples_per_sec']:.2f} samples/sec")
    
    if "model_inference" in results:
        print("\nModel Inference Performance:")
        for key, data in results["model_inference"].items():
            if "throughput_samples_per_sec" in data:
                print(f"  {key}: {data['throughput_samples_per_sec']:.2f} samples/sec")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()