# model/trainer/DITTrainer_NVTX.py
# NVTX-annotated version of DITTrainer for comprehensive performance profiling

import torch
import torch.nn as nn
import wandb
import nvtx
from diffusers.training_utils import EMAModel
from transformers import Trainer, PreTrainedTokenizer
from typing import Dict, Any, List, Tuple, Optional, Union
from ..dataloader.AsyncDataCollator import AsyncDataCollator, PipelinedDataLoader


class DITDataCollator:
    """
    NVTX-annotated data collator for the DIT model with comprehensive profiling.

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

    @nvtx.annotate("DITDataCollator.__call__", color="blue")
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack tensors from the list of features  
        with nvtx.annotate("stack_input_tensors", color="cyan"):
            input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
            attention_mask = torch.stack(
                [torch.tensor(f["attention_mask"]) for f in features]
            )

        batch_size = input_ids.shape[0]

        # Sample random timesteps for diffusion process
        with nvtx.annotate("sample_diffusion_timesteps", color="green"):
            timesteps = self.scheduler.get_timesteps(batch_size, input_ids.device)

        # Apply diffusion noise corruption to input sequences
        with nvtx.annotate("apply_diffusion_noise", color="orange"):
            noisy_input_ids, noise_mask = self.scheduler.add_noise(
                input_ids, timesteps, self.mask_token_id
            )

        # Prepare batch dictionary for model consumption
        with nvtx.annotate("prepare_batch_dict", color="purple"):
            batch_dict = {
                "input_ids": noisy_input_ids,
                "attention_mask": attention_mask,
                "timesteps": timesteps,
                "labels": input_ids,  # Use original sequences as labels
                "noise_mask": noise_mask,
            }

        return batch_dict


class DITTrainer(Trainer):
    """
    NVTX-annotated specialized Trainer for DIT models with comprehensive profiling.

    This class extends the standard Hugging Face Trainer to:
    1. Implement a custom loss function where the loss is computed only at the
       token positions that were corrupted by noise.
    2. Integrate and manage an EMA model, which maintains a shadow copy of the
       model's weights and updates them with a decay factor after each training step.
    3. Ensure that the EMA weights, not the standard model weights, are saved
       during checkpointing, as they often provide better generalization.
    4. Provide comprehensive NVTX annotations for detailed performance profiling.

    Args:
        pad_token_id (Optional[int]): The ID of the padding token, used to ignore padded
                                     positions in the loss calculation.
        ema_decay (float): The decay factor for the EMA model. A higher value
                           results in slower, more stable updates.
        enable_async_dataloader (bool): Whether to enable async dataloader with dual CUDA streams.
    """
    def __init__(self, *args, pad_token_id: Optional[int] = None, ema_decay: float = 0.9999, 
                 enable_async_dataloader: bool = True, **kwargs):
        with nvtx.annotate("DITTrainer.__init__", color="red"):
            super().__init__(*args, **kwargs)
            self.pad_token_id: Optional[int] = pad_token_id
            self.cumulative_tokens = 0
            self.enable_async_dataloader = enable_async_dataloader
            
            # Initialize the EMAModel with NVTX annotation
            with nvtx.annotate("EMAModel_init", color="yellow"):
                # Initialize the EMAModel, which will handle device placement automatically.
                # It creates a shadow copy of the model's parameters for smooth updates.
                self.ema_model: EMAModel = EMAModel(
                    self.model, decay=ema_decay
                )

    @nvtx.annotate("compute_loss", color="red")
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch: int = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        NVTX-annotated override of the default loss computation for DIT diffusion model.
        
        For protein sequences with format [CLS] M K W V ... Y S [EOS]:
        - We predict next tokens: M→K, K→W, ..., Y→S, S→[EOS]  
        - CLS token doesn't participate in prediction (it's the start marker)
        - Loss is calculated only at positions corrupted by diffusion noise
        - This ensures proper BOS/EOS token handling for full vs cropped proteins
        """
        with nvtx.annotate("extract_input_tensors", color="cyan"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            timesteps = inputs["timesteps"]
            labels = inputs["labels"]
            noise_mask = inputs["noise_mask"]

        # Model forward pass with detailed profiling
        with nvtx.annotate("dit_model_forward", color="purple"):
            with nvtx.annotate("attention_computation", color="pink"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    timesteps=timesteps,
                )

        # Extract sequence representations and compute logits
        with nvtx.annotate("logits_computation", color="green"):
            sequence_output = outputs.last_hidden_state
            logits = model.lm_head(sequence_output)

        # Compute diffusion-specific loss with comprehensive profiling
        with nvtx.annotate("compute_diffusion_loss", color="yellow"):
            # For protein sequences: [CLS] M K W V ... Y S [EOS]
            # We want to predict: M→K, K→W, ..., Y→S, S→[EOS]
            # So we use positions 0:-1 to predict positions 1: (excluding CLS from being predicted)
            with nvtx.annotate("shift_tensors_for_next_token_prediction", color="orange"):
                shift_logits = logits[..., :-1, :].contiguous()  # Predictions at pos 0 to n-1
                shift_labels = labels[..., 1:].contiguous()       # Target tokens at pos 1 to n
                
                # Also shift the noise mask to align with the shifted logits and labels
                shift_noise_mask = noise_mask[..., 1:].contiguous()

            # Create loss function that ignores PAD tokens
            with nvtx.annotate("create_loss_function", color="red"):
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            
            # Only compute loss at positions that were corrupted by diffusion
            # This ensures we learn to denoise the corrupted amino acids
            with nvtx.annotate("mask_corrupted_positions", color="lime"):
                masked_logits = shift_logits.view(-1, shift_logits.size(-1))[shift_noise_mask.view(-1)]
                masked_labels = shift_labels.view(-1)[shift_noise_mask.view(-1)]
            
            with nvtx.annotate("compute_cross_entropy_loss", color="coral"):
                loss = loss_fct(masked_logits, masked_labels)

        return (loss, outputs) if return_outputs else loss

    @nvtx.annotate("training_step", color="magenta")
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: int = None) -> torch.Tensor:
        """
        NVTX-annotated override of training_step to update EMA weights and log to WandB.
        """
        # Perform the original training step (forward, loss, backward) with profiling
        with nvtx.annotate("super_training_step", color="lime"):
            with nvtx.annotate("forward_pass", color="blue"):
                loss = super().training_step(model, inputs, num_items_in_batch)

        # After the gradients have been updated, update the EMA model.
        with nvtx.annotate("ema_model_update", color="coral"):
            self.ema_model.step(model.parameters())

        # --- Custom WandB Logging with NVTX ---
        with nvtx.annotate("wandb_logging", color="yellow"):
            if self.is_world_process_zero() and wandb.run:
                # Calculate non-padding tokens in the batch for the current device
                with nvtx.annotate("count_non_padding_tokens", color="cyan"):
                    tokens_in_batch = inputs["input_ids"].ne(self.pad_token_id).sum().item()
                
                # Extrapolate to all processes for total tokens in the global batch
                with nvtx.annotate("compute_global_token_count", color="green"):
                    total_tokens_in_global_batch = tokens_in_batch * self.args.world_size
                    self.cumulative_tokens += total_tokens_in_global_batch
                
                # Log loss and cumulative tokens to WandB
                with nvtx.annotate("wandb_log_metrics", color="orange"):
                    wandb.log({
                        "train/loss": loss.item(),
                        "cumulative_tokens": self.cumulative_tokens
                    })
        # --- End Custom WandB Logging ---

        return loss

    @nvtx.annotate("get_train_dataloader", color="blue")
    def get_train_dataloader(self):
        """
        NVTX-annotated override to create async dataloader if enabled.
        """
        with nvtx.annotate("super_get_train_dataloader", color="cyan"):
            train_dataloader = super().get_train_dataloader()
        
        if self.enable_async_dataloader and hasattr(self, 'data_collator'):
            with nvtx.annotate("create_async_dataloader", color="green"):
                # Wrap the existing data collator with async version
                device = self.args.device if hasattr(self.args, 'device') else 'cuda'
                
                with nvtx.annotate("create_async_collator", color="orange"):
                    async_collator = AsyncDataCollator(self.data_collator, device)
                
                # Create pipelined dataloader
                with nvtx.annotate("create_pipelined_dataloader", color="purple"):
                    return PipelinedDataLoader(train_dataloader, async_collator)
        
        return train_dataloader
    
    @nvtx.annotate("evaluate", color="purple")
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        NVTX-annotated enhanced evaluation using EMA weights and logs metrics to WandB.
        """
        # Temporarily switch to EMA weights for evaluation
        with nvtx.annotate("switch_to_ema_weights", color="yellow"):
            self.ema_model.copy_to(self.model.parameters())
        
        # Run evaluation with comprehensive profiling
        with nvtx.annotate("super_evaluate", color="cyan"):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Restore training weights
        with nvtx.annotate("restore_training_weights", color="orange"):
            self.ema_model.restore(self.model.parameters())
        
        # --- Custom WandB Logging for Evaluation with NVTX ---
        with nvtx.annotate("wandb_eval_logging", color="green"):
            if self.is_world_process_zero() and wandb.run:
                # Prepare metrics for logging, ensuring they are serializable
                with nvtx.annotate("prepare_eval_metrics", color="lime"):
                    metrics_to_log = {
                        f"{metric_key_prefix}/loss": metrics.get(f"{metric_key_prefix}_loss"),
                        "cumulative_tokens": self.cumulative_tokens
                    }
                    # Filter out any potential None values before logging
                    metrics_to_log = {k: v for k, v in metrics_to_log.items() if v is not None}
                
                with nvtx.annotate("wandb_log_eval_metrics", color="coral"):
                    wandb.log(metrics_to_log)
        # --- End Custom WandB Logging ---
        
        return metrics

    @nvtx.annotate("save_model", color="red")
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        NVTX-annotated override of save_model to ensure that the EMA weights are saved.
        """
        # Before saving, copy the averaged EMA parameters to the main model
        with nvtx.annotate("copy_ema_weights_for_saving", color="yellow"):
            self.ema_model.copy_to(self.model.parameters())
        
        # Call the original save method with profiling
        with nvtx.annotate("super_save_model", color="cyan"):
            super().save_model(output_dir, _internal_call)
        
        # IMPORTANT: Immediately restore the original model parameters after saving
        with nvtx.annotate("restore_training_weights_after_save", color="orange"):
            self.ema_model.restore(self.model.parameters())