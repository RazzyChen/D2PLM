#!/usr/bin/env python3
"""
D2PLM Inference Script for Protein Sequence Embeddings

Extract CLS token embeddings from trained D2PLM models (Diffusion or Flow Matching).
The CLS embedding contains aggregated contextual information from the entire protein sequence.

Usage:
    python inference.py --model_path /path/to/model --fasta_file sequences.fasta
    python inference.py --model_path /path/to/model --sequences "MKWV,MGAS"
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# SafeTensors for loading model weights
try:
    from safetensors.torch import load_file as load_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️  safetensors not available, install with: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

# BioPython for FASTA parsing
try:
    from Bio import SeqIO

    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("⚠️  BioPython not available, install with: pip install biopython")
    BIOPYTHON_AVAILABLE = False

from transformers import AutoTokenizer

from model.backbone.dit_config import DITConfig
from model.backbone.dit_model import DITModel


class D2PLMEmbeddingExtractor:
    """
    Extract protein sequence embeddings from trained D2PLM models.

    This class provides a unified interface for extracting CLS token embeddings
    from either Diffusion or Flow Matching trained models. The CLS embedding
    captures the aggregated contextual representation of the entire protein sequence.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ("cuda", "cpu", or "auto")
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)

        print(f"🚀 Loading D2PLM model from: {self.model_path}")
        print(f"📱 Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode

        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {self._count_parameters():.1f}M")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the ESM tokenizer."""
        # Try to load from model directory first, fallback to original ESM
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("📝 Loaded tokenizer from model directory")
        except:
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            print("📝 Loaded original ESM tokenizer")

        return tokenizer

    def _load_model(self) -> DITModel:
        """Load the trained D2PLM model with safetensors support."""
        # Load model configuration
        config_path = self.model_path / "config.json"
        if config_path.exists():
            config = DITConfig.from_json_file(str(config_path))
        else:
            # Fallback: try to infer from model files
            config = self._infer_config()

        # Create model
        model = DITModel(config)

        # Load trained weights with safetensors support
        model_file, is_safetensor = self._find_model_file()

        if is_safetensor and SAFETENSORS_AVAILABLE:
            print(f"📦 Loading safetensors weights: {model_file.name}")
            state_dict = load_safetensors(str(model_file))
        else:
            print(f"📦 Loading PyTorch weights: {model_file.name}")
            state_dict = torch.load(
                model_file, map_location=self.device, weights_only=True
            )

            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)

        return model

    def _find_model_file(self) -> Tuple[Path, bool]:
        """Find the model weights file and return whether it's safetensors."""
        # Prioritize safetensors files
        safetensor_patterns = [
            "model.safetensors",
            "pytorch_model.safetensors",
        ]

        pytorch_patterns = [
            "pytorch_model.bin",
            "model.bin",
            "checkpoint.pt",
            "best_model.pt",
        ]

        # First try safetensors
        for pattern in safetensor_patterns:
            model_file = self.model_path / pattern
            if model_file.exists():
                print(f"📦 Found safetensors weights: {pattern}")
                return model_file, True

        # Then try PyTorch files
        for pattern in pytorch_patterns:
            model_file = self.model_path / pattern
            if model_file.exists():
                print(f"📦 Found PyTorch weights: {pattern}")
                return model_file, False

        # Search for any safetensors files first
        safetensor_matches = list(self.model_path.glob("*.safetensors"))
        if safetensor_matches:
            print(f"📦 Found safetensors weights: {safetensor_matches[0].name}")
            return safetensor_matches[0], True

        # Then search for PyTorch files
        for ext in ["*.bin", "*.pt"]:
            matches = list(self.model_path.glob(ext))
            if matches:
                print(f"📦 Found PyTorch weights: {matches[0].name}")
                return matches[0], False

        raise FileNotFoundError(f"No model weights found in {self.model_path}")

    def _infer_config(self) -> DITConfig:
        """Infer model configuration from available information."""
        print("⚠️  No config.json found, using default D2PLM configuration")
        return DITConfig(
            vocab_size=33,
            max_position_embeddings=1024,
            hidden_size=512,
            num_hidden_layers=32,
            num_attention_heads=16,
            intermediate_size=2048,
            time_embedding_dim=256,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=getattr(self.tokenizer, "mask_token_id", 32),
            cls_token_id=getattr(self.tokenizer, "cls_token_id", 0),
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _count_parameters(self) -> float:
        """Count total trainable parameters in millions."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6

    @torch.no_grad()
    def extract_embeddings(
        self, sequences: List[str], batch_size: int = 8, max_length: int = 1024
    ) -> Dict[str, np.ndarray]:
        """
        Extract CLS token embeddings for protein sequences.

        Args:
            sequences: List of protein sequences (amino acid strings)
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Dictionary with:
                - 'embeddings': CLS token embeddings [num_sequences, hidden_size]
                - 'sequences': Original input sequences
                - 'sequence_lengths': Actual sequence lengths (excluding special tokens)
        """
        print(f"🧬 Processing {len(sequences)} protein sequences...")

        all_embeddings = []
        sequence_lengths = []

        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]

            # Tokenize batch
            encoded = self.tokenizer(
                batch_seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Get sequence lengths (excluding padding)
            batch_lengths = (
                attention_mask.sum(dim=1).cpu().numpy() - 2
            )  # -2 for CLS and EOS
            sequence_lengths.extend(batch_lengths.tolist())

            # Forward pass through model
            # Use dummy timesteps for inference (not used in embedding extraction)
            dummy_timesteps = torch.zeros(input_ids.shape[0], device=self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                timesteps=dummy_timesteps,
            )

            # Extract CLS token embeddings (position 0)
            cls_embeddings = outputs.last_hidden_state[
                :, 0, :
            ]  # [batch_size, hidden_size]
            all_embeddings.append(cls_embeddings.cpu().numpy())

            print(
                f"📊 Processed batch {i // batch_size + 1}/{(len(sequences) - 1) // batch_size + 1}"
            )

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)

        print(f"✅ Extracted embeddings shape: {embeddings.shape}")

        results = {
            "embeddings": embeddings,
            "sequences": sequences,
            "sequence_lengths": sequence_lengths,
            "hidden_size": embeddings.shape[1],
            "model_path": str(self.model_path),
        }

        return results


def load_sequences_from_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """
    Load protein sequences from FASTA file using BioPython.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Tuple of (sequences, sequence_ids) where:
        - sequences: List of amino acid sequences
        - sequence_ids: List of sequence identifiers from FASTA headers
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError(
            "BioPython is required for FASTA parsing. Install with: pip install biopython"
        )

    sequences = []
    sequence_ids = []

    print(f"🧬 Parsing FASTA file: {fasta_path}")

    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence = str(record.seq).upper()
            # Validate protein sequence (basic check)
            if sequence and all(c in "ACDEFGHIKLMNPQRSTVWYX*" for c in sequence):
                sequences.append(sequence)
                sequence_ids.append(record.id)
            else:
                print(f"⚠️  Skipping invalid sequence: {record.id}")

        print(f"✅ Loaded {len(sequences)} valid protein sequences")
        return sequences, sequence_ids

    except Exception as e:
        raise ValueError(f"Error parsing FASTA file {fasta_path}: {e}")


def load_sequences_from_text(text_sequences: str) -> Tuple[List[str], List[str]]:
    """
    Load protein sequences from comma-separated text.

    Args:
        text_sequences: Comma-separated protein sequences

    Returns:
        Tuple of (sequences, sequence_ids)
    """
    sequences = [s.strip().upper() for s in text_sequences.split(",") if s.strip()]
    sequence_ids = [f"seq_{i + 1}" for i in range(len(sequences))]

    # Basic validation
    valid_sequences = []
    valid_ids = []

    for seq, seq_id in zip(sequences, sequence_ids):
        if seq and all(c in "ACDEFGHIKLMNPQRSTVWYX*" for c in seq):
            valid_sequences.append(seq)
            valid_ids.append(seq_id)
        else:
            print(f"⚠️  Skipping invalid sequence: {seq}")

    print(f"✅ Loaded {len(valid_sequences)} valid sequences from text input")
    return valid_sequences, valid_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract protein embeddings from D2PLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract embeddings for downstream training
    python inference_embeddings.py --model_path /path/to/model --fasta_file sequences.fasta --output protein_embeddings.pt
    
    # From command line sequences
    python inference_embeddings.py --model_path /path/to/model --sequences "MKWV,MGAS" --output embeddings.pt
    
    # Use GPU with larger batch size  
    python inference_embeddings.py --model_path /path/to/model --fasta_file seqs.fasta --device cuda --batch_size 32 --output embeddings.pt
        """,
    )

    # Model and input
    parser.add_argument(
        "--model_path", required=True, help="Path to trained D2PLM model directory"
    )
    parser.add_argument(
        "--fasta_file", help="FASTA file containing protein sequences (preferred)"
    )
    parser.add_argument(
        "--sequences",
        help="Comma-separated protein sequences (alternative to --fasta_file)",
    )

    # Processing options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: cuda/cpu/mps/auto (default: auto)",
    )

    # Output options
    parser.add_argument(
        "--output",
        help="Output file path for embeddings (saves as .pt for downstream training)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.fasta_file and not args.sequences:
        parser.error("Either --fasta_file or --sequences must be provided")

    if args.fasta_file and args.sequences:
        parser.error("Cannot specify both --fasta_file and --sequences")

    # Load sequences
    if args.fasta_file:
        if not os.path.exists(args.fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {args.fasta_file}")
        sequences, sequence_ids = load_sequences_from_fasta(args.fasta_file)
    else:
        sequences, sequence_ids = load_sequences_from_text(args.sequences)

    if not sequences:
        raise ValueError("No valid sequences provided")

    print(f"🧬 Loaded {len(sequences)} valid protein sequences")

    # Initialize extractor
    extractor = D2PLMEmbeddingExtractor(model_path=args.model_path, device=args.device)

    # Extract embeddings
    results = extractor.extract_embeddings(
        sequences=sequences, batch_size=args.batch_size, max_length=args.max_length
    )

    # Results already contain all needed data

    # Save results for downstream training
    if args.output:
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for downstream training
        training_data = {
            "embeddings": torch.tensor(results["embeddings"], dtype=torch.float32),
            "sequence_ids": sequence_ids,
            "sequences": sequences,
            "sequence_lengths": torch.tensor(
                results["sequence_lengths"], dtype=torch.long
            ),
            "hidden_size": results["hidden_size"],
            "model_path": str(results["model_path"]),
            "metadata": {
                "num_sequences": len(sequences),
                "avg_length": float(np.mean(results["sequence_lengths"])),
                "min_length": int(min(results["sequence_lengths"])),
                "max_length": int(max(results["sequence_lengths"])),
                "extraction_device": str(extractor.device),
            },
        }

        # Save as PyTorch format (best for downstream training)
        torch.save(training_data, output_path)
        print(f"💾 Saved embeddings for downstream training: {output_path}")
        print(f"📊 Data shape: {training_data['embeddings'].shape}")
        print("🔧 Ready for adding classification/regression heads!")
    else:
        # Print summary
        print("\n📊 Embedding Extraction Summary:")
        print(f"Number of sequences: {len(sequences)}")
        print(f"Embedding dimension: {results['hidden_size']}")
        print(
            f"Average sequence length: {np.mean(results['sequence_lengths']):.1f} amino acids"
        )
        print(
            f"Min/Max sequence length: {min(results['sequence_lengths'])}/{max(results['sequence_lengths'])}"
        )

        print("\n🔍 First few sequences:")
        for i in range(min(3, len(sequences))):
            seq_preview = (
                sequences[i][:50] + "..." if len(sequences[i]) > 50 else sequences[i]
            )
            print(
                f"  {sequence_ids[i]}: {seq_preview} (length: {results['sequence_lengths'][i]})"
            )

        print("\n📊 First embedding (first 10 dimensions):")
        print(f"  {results['embeddings'][0][:10]}")

        print("\n💡 To save for downstream training, use: --output embeddings.pt")


if __name__ == "__main__":
    main()
