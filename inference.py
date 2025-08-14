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
from model.backbone.flow_matching_scheduler import DiscreteAbsorbingFlowMatchingScheduler


class D2PLMEmbeddingExtractor:
    """
    Extract protein sequence embeddings from trained D2PLM models.

    This class provides a unified interface for extracting CLS token embeddings
    from either Diffusion or Flow Matching trained models. The CLS embedding
    captures the aggregated contextual representation of the entire protein sequence.
    
    For Flow Matching models, supports both embedding extraction and sequence generation.
    """

    def __init__(self, model_path: str, device: str = "auto", model_type: str = "auto"):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ("cuda", "cpu", or "auto")
            model_type: Model type ("diffusion", "flow_matching", or "auto")
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.model_type = self._detect_model_type(model_type)

        print(f"🚀 Loading D2PLM model from: {self.model_path}")
        print(f"📱 Using device: {self.device}")
        print(f"🔧 Model type: {self.model_type}")

        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode
        
        # Load flow matching scheduler if needed
        self.flow_scheduler = None
        if self.model_type == "flow_matching":
            self.flow_scheduler = self._load_flow_scheduler()

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

    def _detect_model_type(self, model_type: str) -> str:
        """Detect model type from configuration files."""
        if model_type != "auto":
            return model_type
            
        # Check for flow matching configuration
        fm_config_path = self.model_path / "fm_config.yaml"
        if fm_config_path.exists():
            print("🔍 Detected Flow Matching model (fm_config.yaml found)")
            return "flow_matching"
            
        # Default to diffusion if no specific indicators found
        print("🔍 Defaulting to Diffusion model")
        return "diffusion"

    def _load_flow_scheduler(self) -> DiscreteAbsorbingFlowMatchingScheduler:
        """Load Flow Matching scheduler from configuration."""
        try:
            import yaml
            fm_config_path = self.model_path / "fm_config.yaml"
            
            if fm_config_path.exists():
                with open(fm_config_path, 'r') as f:
                    fm_config = yaml.safe_load(f)
                
                # Extract flow matching parameters
                fm_params = fm_config.get('flow_matching', {})
                
                scheduler = DiscreteAbsorbingFlowMatchingScheduler(
                    vocab_size=fm_config['model']['vocab_size'],
                    absorbing_token_id=getattr(self.tokenizer, 'mask_token_id', 32),
                    num_flow_steps=fm_params.get('num_flow_steps', 100),
                    flow_schedule=fm_params.get('flow_schedule', 'cosine'),
                    min_flow_time=fm_params.get('min_flow_time', 1e-5),
                    max_flow_time=fm_params.get('max_flow_time', 1.0),
                )
                
                print(f"📅 Flow scheduler loaded: {fm_params.get('num_flow_steps', 100)} steps, {fm_params.get('flow_schedule', 'cosine')} schedule")
                return scheduler
            else:
                print("⚠️  fm_config.yaml not found, using default flow scheduler")
                return DiscreteAbsorbingFlowMatchingScheduler(
                    vocab_size=33,
                    absorbing_token_id=getattr(self.tokenizer, 'mask_token_id', 32)
                )
                
        except ImportError:
            print("⚠️  PyYAML not available for flow matching config, using defaults")
            return DiscreteAbsorbingFlowMatchingScheduler(
                vocab_size=33,
                absorbing_token_id=getattr(self.tokenizer, 'mask_token_id', 32)
            )

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
            "model_type": self.model_type,
        }

        return results

    @torch.no_grad()
    def generate_sequences(
        self, 
        num_sequences: int = 1, 
        max_length: int = 512, 
        temperature: float = 1.0,
        num_steps: int = None,
        cls_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
    ) -> Dict[str, List[str]]:
        """
        Generate protein sequences using Flow Matching (only available for flow_matching models).
        
        Args:
            num_sequences: Number of sequences to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more diverse)
            num_steps: Number of flow steps (uses scheduler default if None)
            cls_token_id: CLS token ID (default: 0)
            eos_token_id: EOS token ID (default: 2)  
            pad_token_id: Padding token ID (default: 1)
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        if self.model_type != "flow_matching":
            raise ValueError("Sequence generation is only available for Flow Matching models")
            
        if self.flow_scheduler is None:
            raise RuntimeError("Flow scheduler not loaded")
            
        print(f"🧬 Generating {num_sequences} protein sequences...")
        print(f"📏 Max length: {max_length}, Temperature: {temperature}")
        
        # Generate sequences
        generated_tokens = self.flow_scheduler.generate(
            model=self.model,
            shape=(num_sequences, max_length),
            device=self.device,
            temperature=temperature,
            num_steps=num_steps
        )
        
        # Process generated tokens to create valid protein sequences
        generated_sequences = []
        for i in range(num_sequences):
            # Convert tokens to sequence
            tokens = generated_tokens[i].cpu().tolist()
            
            # Add CLS token at the beginning if not present
            if tokens[0] != cls_token_id:
                tokens = [cls_token_id] + tokens
                
            # Find a reasonable place to add EOS token (avoid very short sequences)
            # Look for a natural ending point or use a reasonable fraction of max_length
            min_length = min(50, max_length // 4)  # At least 50 AA or 1/4 of max length
            
            # Try to find a good stopping point after min_length
            eos_pos = len(tokens) - 1
            for j in range(min_length, len(tokens)):
                # Simple heuristic: stop at natural boundary tokens or when sequence becomes repetitive
                if j >= len(tokens) - 10:  # Near the end
                    eos_pos = j
                    break
                    
            # Insert EOS token and truncate
            if eos_pos < len(tokens) - 1:
                tokens = tokens[:eos_pos] + [eos_token_id]
            else:
                tokens.append(eos_token_id)
            
            # Convert to string
            try:
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=False)
                # Remove special tokens for clean sequence  
                sequence_clean = self.tokenizer.decode(tokens, skip_special_tokens=True)
                generated_sequences.append(sequence_clean)
            except:
                # Fallback: convert tokens manually if tokenizer fails
                # This is a simplified conversion - you might need to adjust based on your tokenizer
                sequence_clean = ""
                for token_id in tokens[1:-1]:  # Skip CLS and EOS
                    if token_id not in [cls_token_id, eos_token_id, pad_token_id]:
                        # Simple mapping - this might need adjustment for your specific tokenizer
                        if token_id < len("ACDEFGHIKLMNPQRSTVWY"):
                            sequence_clean += "ACDEFGHIKLMNPQRSTVWY"[token_id]
                generated_sequences.append(sequence_clean)
        
        print(f"✅ Generated {len(generated_sequences)} sequences")
        
        # Show some examples
        print("\n🔍 Generated sequence examples:")
        for i, seq in enumerate(generated_sequences[:3]):
            preview = seq[:50] + "..." if len(seq) > 50 else seq
            print(f"  Sequence {i+1}: {preview} (length: {len(seq)})")
            
        return {
            "sequences": generated_sequences,
            "num_sequences": len(generated_sequences),
            "generation_params": {
                "temperature": temperature,
                "max_length": max_length,
                "num_steps": num_steps or self.flow_scheduler.num_flow_steps,
                "model_type": self.model_type
            }
        }


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
        description="Extract protein embeddings or generate sequences from D2PLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract embeddings for downstream training
    python inference.py --model_path /path/to/model --fasta_file sequences.fasta --output protein_embeddings.pt
    
    # From command line sequences  
    python inference.py --model_path /path/to/model --sequences "MKWV,MGAS" --output embeddings.pt
    
    # Generate new sequences (Flow Matching models only)
    python inference.py --model_path /path/to/flow_matching_model --generate --num_sequences 10 --max_length 200
    
    # Generate with custom parameters
    python inference.py --model_path /path/to/flow_matching_model --generate --num_sequences 5 --temperature 1.2 --num_steps 50
        """,
    )

    # Model and input
    parser.add_argument(
        "--model_path", required=True, help="Path to trained D2PLM model directory"
    )
    parser.add_argument(
        "--fasta_file", help="FASTA file containing protein sequences (for embedding extraction)"
    )
    parser.add_argument(
        "--sequences",
        help="Comma-separated protein sequences (alternative to --fasta_file)",
    )
    
    # Mode selection
    parser.add_argument(
        "--generate", 
        action="store_true",
        help="Generate new sequences instead of extracting embeddings (Flow Matching only)"
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
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "diffusion", "flow_matching"],
        help="Model type (default: auto-detect)",
    )
    
    # Generation-specific options
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate (generation mode only, default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of flow steps for generation (uses model default if not specified)",
    )

    # Output options
    parser.add_argument(
        "--output",
        help="Output file path for embeddings (.pt) or generated sequences (.fasta)",
    )

    args = parser.parse_args()

    # Validate inputs based on mode
    if args.generate:
        # Generation mode
        if args.fasta_file or args.sequences:
            print("⚠️  In generation mode, --fasta_file and --sequences are ignored")
        print(f"🚀 Generation mode: creating {args.num_sequences} new sequences")
    else:
        # Embedding extraction mode  
        if not args.fasta_file and not args.sequences:
            parser.error("For embedding extraction, either --fasta_file or --sequences must be provided")
        if args.fasta_file and args.sequences:
            parser.error("Cannot specify both --fasta_file and --sequences")
        
        # Load sequences for embedding extraction
        if args.fasta_file:
            if not os.path.exists(args.fasta_file):
                raise FileNotFoundError(f"FASTA file not found: {args.fasta_file}")
            sequences, sequence_ids = load_sequences_from_fasta(args.fasta_file)
        else:
            sequences, sequence_ids = load_sequences_from_text(args.sequences)

        if not sequences:
            raise ValueError("No valid sequences provided")
        
        print(f"🧬 Loaded {len(sequences)} valid protein sequences for embedding extraction")

    # Initialize extractor
    extractor = D2PLMEmbeddingExtractor(
        model_path=args.model_path, 
        device=args.device,
        model_type=args.model_type
    )

    if args.generate:
        # Generation mode
        results = extractor.generate_sequences(
            num_sequences=args.num_sequences,
            max_length=args.max_length,
            temperature=args.temperature,
            num_steps=args.num_steps,
        )
        
        # Save or display results
        if args.output:
            # Save as FASTA file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for i, seq in enumerate(results["sequences"]):
                    f.write(f">generated_sequence_{i+1}\n{seq}\n")
            
            print(f"💾 Saved generated sequences: {output_path}")
            print(f"📊 Generated {len(results['sequences'])} sequences")
        else:
            # Print generated sequences
            print("\n🧬 Generated sequences:")
            for i, seq in enumerate(results["sequences"]):
                print(f">generated_sequence_{i+1}")
                print(seq)
            
            print(f"\n💡 To save sequences, use: --output generated_sequences.fasta")
            
    else:
        # Embedding extraction mode
        results = extractor.extract_embeddings(
            sequences=sequences, 
            batch_size=args.batch_size, 
            max_length=args.max_length
        )

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
                "model_type": results["model_type"],
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
