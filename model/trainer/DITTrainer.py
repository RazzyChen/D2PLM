# model/trainer/DITTrainer.py

import torch
import torch.nn as nn
import wandb
from diffusers.training_utils import EMAModel
from transformers import Trainer, PreTrainedTokenizer
from typing import Dict, Any, List, Tuple, Optional, Union

class DITDataCollator:
    """
    Data collator for the DIT model.

    This collator is responsible for processing a batch of data samples. For each
    batch, it performs the following key operations:
    1. Samples a random timestep for each sequence in the batch.
    2. Applies noise to the sequences based on the sampled timesteps, corrupting
       some tokens to the MASK token ID, according to the diffusion schedule.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing sequences.
        scheduler (Any): The diffusion scheduler (e.g., DITDiffusionScheduler) instance.
        mask_token_id (int): The token ID used for the absorbing (MASK) state.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, scheduler: Any, mask_token_id: int):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.scheduler: Any = scheduler
        self.mask_token_id: int = mask_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack tensors from the list of features
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        attention_mask = torch.stack(
            [torch.tensor(f["attention_mask"]) for f in features]
        )

        batch_size = input_ids.shape[0]

        # Sample random timesteps
        timesteps = self.scheduler.get_timesteps(batch_size, input_ids.device)

        # Apply noise (corruption)
        noisy_input_ids, noise_mask = self.scheduler.add_noise(
            input_ids, timesteps, self.mask_token_id
        )

        return {
            "input_ids": noisy_input_ids,
            "attention_mask": attention_mask,
            "timesteps": timesteps,
            "labels": input_ids,  # Use original sequences as labels
            "noise_mask": noise_mask,
        }

class DITTrainer(Trainer):
    """
    A specialized Trainer for DIT models that integrates custom loss calculation
    and Exponential Moving Average (EMA) for model weights.

    This class extends the standard Hugging Face Trainer to:
    1. Implement a custom loss function where the loss is computed only at the
       token positions that were corrupted by noise.
    2. Integrate and manage an EMA model, which maintains a shadow copy of the
       model's weights and updates them with a decay factor after each training step.
    3. Ensure that the EMA weights, not the standard model weights, are saved
       during checkpointing, as they often provide better generalization.

    Args:
        pad_token_id (Optional[int]): The ID of the padding token, used to ignore padded
                                     positions in the loss calculation.
        ema_decay (float): The decay factor for the EMA model. A higher value
                           results in slower, more stable updates.
    """
    def __init__(self, *args, pad_token_id: Optional[int] = None, ema_decay: float = 0.9999, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id: Optional[int] = pad_token_id
        self.cumulative_tokens = 0
        
        # Initialize the EMAModel, which will handle device placement automatically.
        # It creates a shadow copy of the model's parameters for smooth updates.
        self.ema_model: EMAModel = EMAModel(
            self.model, decay=ema_decay
        )

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Overrides the default loss computation for DIT diffusion model.
        
        For protein sequences with format [CLS] M K W V ... Y S [EOS]:
        - We predict next tokens: M→K, K→W, ..., Y→S, S→[EOS]  
        - CLS token doesn't participate in prediction (it's the start marker)
        - Loss is calculated only at positions corrupted by diffusion noise
        - This ensures proper BOS/EOS token handling for full vs cropped proteins
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        timesteps = inputs["timesteps"]
        labels = inputs["labels"]
        noise_mask = inputs["noise_mask"]

        # Model forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )

        sequence_output = outputs.last_hidden_state
        logits = model.lm_head(sequence_output)

        # For protein sequences: [CLS] M K W V ... Y S [EOS]
        # We want to predict: M→K, K→W, ..., Y→S, S→[EOS]
        # So we use positions 0:-1 to predict positions 1: (excluding CLS from being predicted)
        shift_logits = logits[..., :-1, :].contiguous()  # Predictions at pos 0 to n-1
        shift_labels = labels[..., 1:].contiguous()       # Target tokens at pos 1 to n
        
        # Also shift the noise mask to align with the shifted logits and labels
        shift_noise_mask = noise_mask[..., 1:].contiguous()

        # Create loss function that ignores PAD tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        
        # Only compute loss at positions that were corrupted by diffusion
        # This ensures we learn to denoise the corrupted amino acids
        masked_logits = shift_logits.view(-1, shift_logits.size(-1))[shift_noise_mask.view(-1)]
        masked_labels = shift_labels.view(-1)[shift_noise_mask.view(-1)]
        
        loss = loss_fct(masked_logits, masked_labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Overrides the training_step to update EMA weights and log to WandB.
        """
        # Perform the original training step (forward, loss, backward)
        loss = super().training_step(model, inputs)

        # After the gradients have been updated, update the EMA model.
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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Overrides save_model to ensure that the EMA weights are saved.
        """
        # Before saving, copy the averaged EMA parameters to the main model
        self.ema_model.copy_to(self.model.parameters())
        
        # Call the original save method
        super().save_model(output_dir, _internal_call)
        
        # IMPORTANT: Immediately restore the original model parameters after saving
        self.ema_model.restore(self.model.parameters())