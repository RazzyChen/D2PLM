# model/backbone/flow_matching_scheduler.py
# Discrete Flow Matching Scheduler with Absorbing States for Protein Sequences
# Based on "Discrete Flow Matching" (arXiv:2407.15595v2)

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class DiscreteAbsorbingFlowMatchingScheduler:
    """
    Discrete Flow Matching Scheduler with Absorbing States for protein sequence generation.
    
    Based on the Discrete Flow Matching paper (arXiv:2407.15595v2), this scheduler implements 
    flow matching on discrete sequences using absorbing states (MASK tokens) instead of 
    traditional diffusion noise schedules.
    
    Key advantages over diffusion:
    - Learns flexible probability paths (not fixed noise schedules)
    - Uses flow matching objective for better sample quality
    - Fewer generation steps required
    - More stable training dynamics
    
    Args:
        vocab_size: Size of the vocabulary (number of tokens)
        absorbing_token_id: Token ID for the absorbing state (typically MASK token)
        num_flow_steps: Number of flow steps for generation (default: 100)
        flow_schedule: Type of flow schedule ('linear', 'cosine', 'sigmoid')
        min_flow_time: Minimum flow time (default: 1e-5)
        max_flow_time: Maximum flow time (default: 1.0)
    """
    
    def __init__(
        self,
        vocab_size: int,
        absorbing_token_id: int,
        num_flow_steps: int = 100,
        flow_schedule: str = 'cosine',
        min_flow_time: float = 1e-5,
        max_flow_time: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.absorbing_token_id = absorbing_token_id
        self.num_flow_steps = num_flow_steps
        self.flow_schedule = flow_schedule
        self.min_flow_time = min_flow_time
        self.max_flow_time = max_flow_time
        
        # Create flow time schedule
        self.flow_times = self._create_flow_schedule()
        
    def _create_flow_schedule(self) -> torch.Tensor:
        """Create flow time schedule based on the specified type."""
        if self.flow_schedule == 'linear':
            return torch.linspace(self.min_flow_time, self.max_flow_time, self.num_flow_steps)
        elif self.flow_schedule == 'cosine':
            # Cosine schedule for smoother interpolation
            t = torch.linspace(0, 1, self.num_flow_steps)
            return self.min_flow_time + (self.max_flow_time - self.min_flow_time) * (1 - torch.cos(t * math.pi / 2))
        elif self.flow_schedule == 'sigmoid':
            # Sigmoid schedule for more gradual changes
            t = torch.linspace(-6, 6, self.num_flow_steps)
            sigmoid_t = torch.sigmoid(t)
            return self.min_flow_time + (self.max_flow_time - self.min_flow_time) * sigmoid_t
        else:
            raise ValueError(f"Unknown flow schedule: {self.flow_schedule}")
    
    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random flow times for training.
        Compatible with existing diffusion interface.
        """
        # Sample uniformly from flow time range
        flow_times = torch.rand(batch_size, device=device)
        flow_times = self.min_flow_time + flow_times * (self.max_flow_time - self.min_flow_time)
        return flow_times
    
    def add_noise(
        self,
        clean_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        mask_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flow Matching forward process with proper BOS/EOS token handling.
        
        For ESM protein sequences: [CLS] M K W V ... Y S [EOS]
        - Only corrupt amino acid tokens (positions 1 to -2)
        - Preserve CLS and EOS tokens to distinguish full vs cropped proteins
        - This is crucial for the model to understand protein boundaries
        
        Args:
            clean_tokens: Clean token sequences [batch_size, seq_len]
            timesteps: Flow times [batch_size] 
            mask_token_id: The absorbing token ID (ESM <mask>)
            
        Returns:
            noisy_tokens: Interpolated sequences [batch_size, seq_len]
            corruption_mask: Mask indicating which tokens were corrupted [batch_size, seq_len]
        """
        batch_size, seq_len = clean_tokens.shape
        device = clean_tokens.device
        
        # Expand flow times for broadcasting
        flow_times_expanded = rearrange(timesteps, 'batch -> batch 1')
        
        # Create corruption probability based on flow time
        # At t=0: no corruption, At t=1: full corruption to absorbing state
        corruption_prob = flow_times_expanded.expand(batch_size, seq_len)
        
        # Sample which tokens to corrupt
        corruption_mask = torch.rand(batch_size, seq_len, device=device) < corruption_prob
        
        # CRITICAL: Protect BOS (CLS) and EOS tokens from corruption
        # ESM format: [CLS] M K W V ... Y S [EOS]
        #            pos 0   1 2 3 4     n-1 n
        if seq_len > 2:  # Ensure we have enough tokens to protect boundaries
            # Never corrupt the CLS token (position 0)
            corruption_mask[:, 0] = False
            
            # Find actual sequence length (excluding padding) and protect EOS
            # Assume pad_token_id = 1 for ESM tokenizer
            attention_mask = (clean_tokens != 1)
            for i in range(batch_size):
                actual_seq_len = attention_mask[i].sum().item()
                if actual_seq_len > 1:
                    # Never corrupt the EOS token at the end of actual sequence
                    corruption_mask[i, actual_seq_len-1] = False
        
        # Create noisy tokens by replacing corrupted positions with absorbing token
        noisy_tokens = clean_tokens.clone()
        noisy_tokens[corruption_mask] = mask_token_id
        
        return noisy_tokens, corruption_mask
    
    @staticmethod
    def compute_flow_matching_loss(
        model_logits: torch.Tensor,
        clean_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        corruption_mask: torch.Tensor,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss for discrete sequences.
        
        Implements the discrete flow matching objective from the paper.
        
        Args:
            model_logits: Model predictions [batch_size, seq_len, vocab_size]
            clean_tokens: Original clean tokens [batch_size, seq_len]
            timesteps: Flow times [batch_size]
            corruption_mask: Mask of corrupted positions [batch_size, seq_len]
            pad_token_id: Padding token ID to ignore in loss
            
        Returns:
            Flow matching loss
        """
        # Create target distribution for flow matching
        # The target is the conditional expectation of the clean data
        target_distribution = torch.zeros_like(model_logits)
        
        # For corrupted positions, target should predict the original token
        batch_indices = torch.arange(clean_tokens.size(0)).unsqueeze(1)
        seq_indices = torch.arange(clean_tokens.size(1)).unsqueeze(0)
        
        # Set target probabilities for clean tokens at corrupted positions
        target_distribution[batch_indices, seq_indices, clean_tokens] = corruption_mask.float()
        
        # Convert logits to log probabilities
        log_probs = F.log_softmax(model_logits, dim=-1)
        
        # Compute cross-entropy loss only on corrupted positions
        ce_loss = -torch.sum(target_distribution * log_probs, dim=-1)
        
        # Apply corruption mask and padding mask
        loss_mask = corruption_mask.float()
        if pad_token_id is not None:
            loss_mask = loss_mask * (clean_tokens != pad_token_id).float()
        
        # Average loss over corrupted positions
        masked_loss = ce_loss * loss_mask
        
        # Normalize by number of corrupted tokens
        total_corrupted = loss_mask.sum()
        if total_corrupted > 0:
            return masked_loss.sum() / total_corrupted
        else:
            return torch.tensor(0.0, device=model_logits.device)
    
    def sample_step(
        self,
        model_logits: torch.Tensor,
        current_tokens: torch.Tensor,
        flow_time: float,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Single sampling step for generation.
        
        Args:
            model_logits: Model predictions [batch_size, seq_len, vocab_size]
            current_tokens: Current token state [batch_size, seq_len]
            flow_time: Current flow time
            temperature: Sampling temperature
            
        Returns:
            Updated token sequences
        """
        # Apply temperature scaling
        scaled_logits = model_logits / temperature
        
        # Sample from the predicted distribution
        probs = F.softmax(scaled_logits, dim=-1)
        
        # For positions that are currently absorbing tokens, sample new tokens
        absorbing_mask = (current_tokens == self.absorbing_token_id)
        
        # Sample new tokens only for absorbing positions
        new_tokens = current_tokens.clone()
        if absorbing_mask.any():
            # Sample from predicted distribution
            sampled_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size), 
                num_samples=1
            ).view(current_tokens.shape)
            
            # Update only absorbing positions
            new_tokens[absorbing_mask] = sampled_tokens[absorbing_mask]
        
        return new_tokens
    
    def generate(
        self,
        model,
        shape: Tuple[int, int],
        device: torch.device,
        temperature: float = 1.0,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequences using flow matching.
        
        Args:
            model: The trained model
            shape: (batch_size, seq_len) shape of sequences to generate
            device: Device to generate on
            temperature: Sampling temperature
            num_steps: Number of generation steps (default: self.num_flow_steps)
            
        Returns:
            Generated sequences [batch_size, seq_len]
        """
        if num_steps is None:
            num_steps = self.num_flow_steps
            
        batch_size, seq_len = shape
        
        # Start with all absorbing tokens
        current_tokens = torch.full(
            size=(batch_size, seq_len),
            fill_value=self.absorbing_token_id,
            device=device,
            dtype=torch.long
        )
        
        # Flow from t=1 (absorbing) to t=0 (clean)
        flow_times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        model.eval()
        with torch.no_grad():
            for i in range(num_steps):
                t = flow_times[i]
                t_batch = torch.full((batch_size,), t.item(), device=device)
                
                # Get model predictions
                outputs = model(
                    input_ids=current_tokens,
                    timesteps=t_batch,
                )
                
                # Sample next state
                logits = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
                current_tokens = self.sample_step(
                    logits,
                    current_tokens,
                    t.item(),
                    temperature=temperature
                )
        
        return current_tokens
    
    def save_pretrained(self, save_directory: str):
        """Save scheduler configuration."""
        import os
        import json
        
        config = {
            'vocab_size': self.vocab_size,
            'absorbing_token_id': self.absorbing_token_id,
            'num_flow_steps': self.num_flow_steps,
            'flow_schedule': self.flow_schedule,
            'min_flow_time': self.min_flow_time,
            'max_flow_time': self.max_flow_time,
        }
        
        config_path = os.path.join(save_directory, 'flow_scheduler_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load scheduler from saved configuration."""
        import os
        import json
        
        config_path = os.path.join(load_directory, 'flow_scheduler_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return cls(**config)