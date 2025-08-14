# model/trainer/FMTrainer.py
# Dedicated Flow Matching Trainer - Completely Independent from Diffusion

import torch
import torch.nn as nn
import wandb
from diffusers.training_utils import EMAModel
from transformers import Trainer, PreTrainedTokenizer
from typing import Dict, Any, List, Tuple, Optional, Union
from ..backbone.flow_matching_scheduler import DiscreteAbsorbingFlowMatchingScheduler


class FMDataCollator:
    """
    Data collator specifically designed for Flow Matching training.
    
    This collator handles the flow matching forward process, creating
    interpolated sequences between clean data and absorbing states.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing sequences.
        scheduler (DiscreteAbsorbingFlowMatchingScheduler): Flow matching scheduler instance.
        mask_token_id (int): The token ID used for the absorbing (MASK) state.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, scheduler: DiscreteAbsorbingFlowMatchingScheduler, mask_token_id: int):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.scheduler: DiscreteAbsorbingFlowMatchingScheduler = scheduler
        self.mask_token_id: int = mask_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of features for flow matching training."""
        # Stack tensors from the list of features
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        attention_mask = torch.stack(
            [torch.tensor(f["attention_mask"]) for f in features]
        )

        batch_size = input_ids.shape[0]

        # Sample random flow times for training
        timesteps = self.scheduler.get_timesteps(batch_size, input_ids.device)

        # Apply flow matching forward process (corruption)
        noisy_input_ids, corruption_mask = self.scheduler.add_noise(
            input_ids, timesteps, self.mask_token_id
        )

        return {
            "input_ids": noisy_input_ids,
            "attention_mask": attention_mask,
            "timesteps": timesteps,
            "labels": input_ids,  # Use original sequences as labels
            "corruption_mask": corruption_mask,
        }


class FMTrainer(Trainer):
    """
    Specialized Trainer for Flow Matching with DIT models.
    
    This trainer is exclusively designed for flow matching training and includes:
    1. Flow matching loss computation based on discrete flow matching objective
    2. Exponential Moving Average (EMA) for stable training
    3. Optimized training loop for flow matching dynamics
    
    Args:
        pad_token_id (Optional[int]): The ID of the padding token for loss masking.
        ema_decay (float): The decay factor for the EMA model.
    """
    
    def __init__(self, *args, pad_token_id: Optional[int] = None, ema_decay: float = 0.9999, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id: Optional[int] = pad_token_id
        self.cumulative_tokens = 0
        
        # Initialize EMA model for stable flow matching training
        self.ema_model: EMAModel = EMAModel(
            self.model, decay=ema_decay
        )

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute flow matching loss for discrete sequences.
        
        This implements the flow matching objective specifically for protein sequences,
        using the corruption mask to focus learning on corrupted positions.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        timesteps = inputs["timesteps"]
        labels = inputs["labels"]
        corruption_mask = inputs["corruption_mask"]

        # Model forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )

        # Get model predictions (logits)
        sequence_output = outputs.last_hidden_state
        logits = model.lm_head(sequence_output)

        # Apply flow matching loss computation
        # Note: Flow matching uses direct token prediction (no shifting)
        loss = DiscreteAbsorbingFlowMatchingScheduler.compute_flow_matching_loss(
            model_logits=logits,
            clean_tokens=labels,
            timesteps=timesteps,
            corruption_mask=corruption_mask,
            pad_token_id=self.pad_token_id,
        )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Enhanced training step with EMA updates and WandB logging.
        """
        # Perform the standard training step
        loss = super().training_step(model, inputs)

        # Update EMA model after optimizer step
        self.ema_model.step(model.parameters())

        # --- Custom WandB Logging ---
        if self.is_world_process_zero() and wandb.run:
            # Calculate non-padding tokens in the batch for the current device
            tokens_in_batch = inputs["input_ids"].ne(self.pad_token_id).sum().item()
            
            # Extrapolate to all processes for total tokens in the global batch
            total_tokens_in_global_batch = tokens_in_batch * self.args.world_size
            self.cumulative_tokens += total_tokens_in_global_batch
            
            # Log loss and cumulative tokens to WandB
            wandb.log({
                "train/loss": loss.item(),
                "cumulative_tokens": self.cumulative_tokens
            })
        # --- End Custom WandB Logging ---

        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with EMA weights for better generalization.
        
        For flow matching, we save the EMA-averaged weights as they typically
        provide better sample quality and more stable generation.
        """
        # Copy EMA weights to main model before saving
        self.ema_model.copy_to(self.model.parameters())
        
        # Save the model with EMA weights
        super().save_model(output_dir, _internal_call)
        
        # Restore original weights for continued training
        self.ema_model.restore(self.model.parameters())

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Enhanced evaluation using EMA weights and logs metrics to WandB.
        """
        # Temporarily switch to EMA weights for evaluation
        self.ema_model.copy_to(self.model.parameters())
        
        # Run evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Restore training weights
        self.ema_model.restore(self.model.parameters())
        
        # --- Custom WandB Logging for Evaluation ---
        if self.is_world_process_zero() and wandb.run:
            # Prepare metrics for logging, ensuring they are serializable
            metrics_to_log = {
                f"{metric_key_prefix}/loss": metrics.get(f"{metric_key_prefix}_loss"),
                "cumulative_tokens": self.cumulative_tokens
            }
            # Filter out any potential None values before logging
            metrics_to_log = {k: v for k, v in metrics_to_log.items() if v is not None}
            wandb.log(metrics_to_log)
        # --- End Custom WandB Logging ---
        
        return metrics
