# model/trainer/DITTrainer.py

import torch
import torch.nn as nn
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
    A specialized Trainer for DIT models that supports both Diffusion and Flow Matching,
    integrating custom loss calculation and Exponential Moving Average (EMA) for model weights.

    This class extends the standard Hugging Face Trainer to:
    1. Support both diffusion and flow matching training paradigms
    2. Implement custom loss functions for both approaches
    3. Integrate and manage an EMA model for stable training
    4. Ensure that the EMA weights are saved during checkpointing

    Args:
        pad_token_id (Optional[int]): The ID of the padding token, used to ignore padded
                                     positions in the loss calculation.
        ema_decay (float): The decay factor for the EMA model. A higher value
                           results in slower, more stable updates.
        use_flow_matching (bool): Whether to use flow matching instead of diffusion.
    """
    def __init__(self, *args, pad_token_id: Optional[int] = None, ema_decay: float = 0.9999, use_flow_matching: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id: Optional[int] = pad_token_id
        self.use_flow_matching: bool = use_flow_matching
        
        # Initialize the EMAModel, which will handle device placement automatically.
        # It creates a shadow copy of the model's parameters for smooth updates.
        self.ema_model: EMAModel = EMAModel(
            self.model, decay=ema_decay
        )

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Overrides the default loss computation to support both diffusion and flow matching.
        
        For diffusion: The loss is calculated only at positions corrupted by noise.
        For flow matching: Uses the flow matching objective for discrete sequences.
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

        if self.use_flow_matching:
            # Flow matching loss computation
            # Import here to avoid circular imports
            from ..backbone.flow_matching_scheduler import DiscreteAbsorbingFlowMatchingScheduler
            
            # For flow matching, we don't shift tokens - direct prediction
            loss = DiscreteAbsorbingFlowMatchingScheduler.compute_flow_matching_loss(
                model_logits=logits,
                clean_tokens=labels,
                timesteps=timesteps,
                corruption_mask=noise_mask,
                pad_token_id=self.pad_token_id,
            )
        else:
            # Traditional diffusion loss computation
            # Align logits and labels for loss calculation (shift them)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Also shift the noise mask to align with the shifted logits and labels
            shift_noise_mask = noise_mask[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            
            # Flatten tensors and use the noise mask as an index to select tokens for loss calculation
            masked_logits = shift_logits.view(-1, shift_logits.size(-1))[shift_noise_mask.view(-1)]
            masked_labels = shift_labels.view(-1)[shift_noise_mask.view(-1)]
            
            loss = loss_fct(masked_logits, masked_labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Overrides the training_step to update EMA weights after each optimizer step.
        """
        # Perform the original training step (forward, loss, backward)
        loss = super().training_step(model, inputs)

        # After the gradients have been updated (optimizer.step() is called internally),
        # update the EMA model with the new model parameters.
        self.ema_model.step(model.parameters())

        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Overrides save_model to ensure that the EMA weights are saved.
        """
        # Before saving, copy the averaged EMA parameters to the main model
        self.ema_model.copy_to(self.model.parameters())
        
        # Call the original save method
        super().save_model(output_dir, _internal_call)
        
        # IMPORTANT: Immediately restore the original model parameters after saving
        # to continue training from the non-averaged weights. Failing to do so
        # would disrupt the optimizer's state and the learning trajectory.
        self.ema_model.restore(self.model.parameters())