claude.md: D2PLM Project Refactoring Plan (FSDP + Accelerate) v7
Note to the Claude Agent: The code presented in this document serves as a high-level refactoring plan and conceptual guide. It is your responsibility to adapt the specific function calls, class initializations, and method overrides to match the exact APIs of the libraries present in the local project environment. All comments, docstrings, and variable names must be implemented in English, including comprehensive type hinting as demonstrated.

This document outlines the migration of the D2PLM project from its current Ray + DeepSpeed architecture to a more modern and native PyTorch FSDP + Hugging Face Accelerate workflow. This refactoring is a comprehensive upgrade of performance and engineering practices.

Core Refactoring Objectives
Deprecate Ray and DeepSpeed: Adopt accelerate launch as the standard distributed launcher. This strategic shift simplifies the project's dependency stack, significantly reducing environmental configuration complexity. By aligning with the native PyTorch ecosystem, the project gains better long-term maintainability and easier access to the latest PyTorch features and community support.

Embrace FSDP: Utilize PyTorch's native FullyShardedDataParallel (FSDP) for memory optimization. The specific sharding strategy will be dynamically configured via the accelerate config file, allowing for flexible adaptation to different hardware environments without code changes. As a first-party solution, FSDP offers a seamless and predictable integration compared to third-party libraries.

Optimize Data Pipeline: Implement an overlapping data load and computation pipeline to maximize hardware utilization. This is a critical step to mitigate the I/O bottleneck, where the GPU might sit idle waiting for data. By creating an asynchronous data pipeline, we ensure that the expensive GPU resources are constantly fed, leading to a direct increase in training throughput.

Integrate EMA: Incorporate Exponential Moving Average (EMA) using the diffusers.training_utils.EMAModel. EMA acts as a temporal smoothing mechanism for model weights during training. It often helps the model converge to a more robust and generalizable solution in the loss landscape, which is particularly beneficial for the stability and final performance of generative models like diffusion transformers.

Modularize and Decouple: Make the training script capable of dynamically loading Hydra configurations and encapsulate the core Trainer logic into a separate, reusable module. This separation of concerns makes the main training script a high-level orchestrator, while the complex implementation details are neatly organized, improving code readability, testability, and reusability for future projects.

Step 1: Modify Configuration File (train_config/train_config.yaml)
The core configuration file will be streamlined to focus exclusively on model and training hyperparameters. This decouples the scientific aspects of the model (architecture, learning rates) from the engineering aspects of its execution (distributed strategy), making experiments more portable and easier to manage.

# DIT (Diffusion Transformer) Training Configuration - v7 (FSDP + Accelerate)

defaults:
  - _self_

# 1. Model Configuration (Unchanged)
model:
  tokenizer: "facebook/esm2_tt33_650M_UR50D"
  vocab_size: 33
  max_position_embeddings: 1024
  hidden_size: 512
  num_hidden_layers: 32
  num_attention_heads: 16
  intermediate_size: 2048
  time_embedding_dim: 256
  hidden_dropout_prob: 0
  attention_probs_dropout_prob: 0
  layer_norm_eps: 1e-5
  initializer_range: 0.02

# 2. Diffusion Scheduler Configuration (Unchanged)
diffusion:
  num_train_timesteps: 1000
  beta_start: 1e-4
  beta_end: 0.02
  beta_schedule: "linear"
  prediction_type: "epsilon"
  steps_offset: 0

# 3. Data Configuration (Unchanged)
data:
  train_lmdb_path: "/workspace/d2plm/prepared_dataset/train_lmdb"
  val_lmdb_path: "/workspace/d2plm/prepared_dataset/validation_lmdb"
  train_cache_dir: "/workspace/d2plm/data_cache/train"
  val_cache_dir: "/workspace/d2plm/data_cache/validation"
  max_length: 1024
  batch_size: 8 # Note: This is per_device_train_batch_size

# 4. Core Training Configuration (for TrainingArguments)
training:
  max_steps: 160000
  learning_rate: 4e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 48
  
  # Optimizer and LR Scheduler
  optim: "adamw_torch_fused"
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1

  # Saving, Logging, and Evaluation Frequency
  save_strategy: "steps"
  save_steps: 500
  logging_steps: 125
  evaluation_strategy: "steps"
  eval_steps: 500
  
  # Checkpoint Management
  save_total_limit: 50
  load_best_model_at_end: false
  ckpt: null # Retain for checkpoint resumption

# 5. System and Environment Configuration
system:
  device: "cuda"
  seed: 42
  mixed_precision: "fp16"
  dataloader_prefetch_factor: 128
  dataloader_num_workers: 23

# 6. Weight Output and Logging Configuration
weight:
  output_dir: "/workspace/d2plm/weight"

logging:
  report_to: "wandb"
  logging_strategy: "steps"

# WandB specific configuration
wabdb:
  project: "D2PLM_DiT_FSDP"
  entity: null
  run_name_prefix: "D2PLM" # For dynamic run name generation

checkpointing:
  save_strategy: "steps"


Step 2: Modify Data Loader (model/dataloader/DataPipe.py)
The data loader modifications remain the same, using standard environment variables to determine the main process, thus ensuring compatibility with any standard launcher.

Step 3: Modularize Trainer Logic (New File: model/trainer/DITTrainer.py)
To enhance code modularity and maintainability, all Trainer-related classes (DITDataCollator and DITTrainer) will be moved to a new, independent file. This encapsulation ensures that the logic for how the model is trained is self-contained and separated from the main script that orchestrates the overall process.

Agent Action: Create the new file model/trainer/DITTrainer.py with the following content.

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
        
        # Initialize the EMAModel, which will handle device placement automatically.
        # It creates a shadow copy of the model's parameters for smooth updates.
        self.ema_model: EMAModel = EMAModel(
            self.model, decay=ema_decay
        )

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Overrides the default loss computation to fit the specific needs of the DIT model.
        The loss is calculated only at the positions corrupted by the diffusion noise.
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


Step 4: Refactor Main Training Script (train.py)
The main training script, train.py, will now be significantly cleaner, focusing solely on high-level process orchestration.

Agent Action: Overwrite the existing train.py with the following content.

#!/usr/bin/env python3

import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

import hydra
import torch
import wandb
import numpy as np
from accelerate import Accelerator
from model.backbone.diffusion_scheduler import DITDiffusionScheduler
from model.backbone.dit_config import DITConfig
from model.backbone.dit_model import DITModel
from model.dataloader.DataPipe import load_and_preprocess_data
# Import the newly modularized Trainer and DataCollator
from model.trainer.DITTrainer import DITDataCollator, DITTrainer
from omegaconf import DictConfig, OmegaConf
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

# create_dit_model_and_tokenizer, create_diffusion_scheduler, prepare_dataset functions remain here
# ...

def main(cfg: DictConfig) -> None:
    """
    Main training function to orchestrate the D2PLM training process.

    This function handles:
    - Dynamic run name and WandB ID generation.
    - Initialization of the Accelerator for distributed training.
    - Configuration of TrainingArguments.
    - Initialization of WandB (on the main process only).
    - Seeding for reproducibility.
    - Creation of the model, tokenizer, scheduler, and datasets.
    - Instantiation of the custom DITTrainer.
    - Launching the training process.
    - Saving the final model and configuration.

    Args:
        cfg (DictConfig): The configuration object provided by Hydra.
    """
    
    # -- Dynamic Run Name Generation --
    timestamp = (datetime.now() + timedelta(hours=8)).strftime("%H%M%m%d%Y")
    run_name_prefix = cfg.wabdb.get("run_name_prefix", "training_run")
    run_name = f"{run_name_prefix}_{timestamp}"
    print(f"Generated Run Name: {run_name}")

    # -- WandB Run ID Management --
    # Create a unique run ID for WandB to allow for resumption
    OmegaConf.set_struct(cfg, False)
    if "wandb_run_id" not in cfg or cfg.wandb_run_id is None:
        cfg.wandb_run_id = run_name
    OmegaConf.set_struct(cfg, True)
    print(f"WandB Run ID: {cfg.wandb_run_id}")
    
    # Initialize Accelerator to get distributed state information
    accelerator = Accelerator()

    # 1. Configure TrainingArguments
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_args = TrainingArguments(
        output_dir=cfg.weight.output_dir,
        per_device_train_batch_size=cfg.data.batch_size,
        per_device_eval_batch_size=cfg.data.batch_size,
        fp16=(cfg.system.mixed_precision == "fp16"),
        dataloader_num_workers=cfg.system.dataloader_num_workers,
        dataloader_pin_memory=True, # Enable for async data transfer
        **training_args_dict,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=cfg.logging.report_to,
        save_safetensors=True,
    )

    # 2. Initialize WandB (only on the main process)
    if accelerator.is_main_process:
        wandb.init(
            project=cfg.wabdb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            id=cfg.wandb_run_id,
            resume="allow"
        )

    # 3. Set Random Seed
    torch.manual_seed(cfg.system.seed)

    # 4. Create Model and Tokenizer
    model, tokenizer, model_config = create_dit_model_and_tokenizer(cfg)

    # 5. Create Diffusion Scheduler
    scheduler = create_diffusion_scheduler(cfg)

    # 6. Prepare Datasets
    train_dataset, eval_dataset = prepare_dataset(cfg, tokenizer)

    # 7. Create Data Collator
    data_collator = DITDataCollator(
        tokenizer,
        scheduler,
        model_config.mask_token_id,
    )

    # 8. Define Evaluation Metric
    perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)

    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_preds
        device = training_args.device
        logits_tensor = torch.from_numpy(logits).to(device, non_blocking=True)
        labels_tensor = torch.from_numpy(labels).to(device, non_blocking=True)
        shift_logits = logits_tensor[..., :-1, :].contiguous()
        shift_labels = labels_tensor[..., 1:].contiguous()
        ppl = perplexity.to(device)(shift_logits, shift_labels)
        return {"perplexity": ppl.item()}

    # 9. Create the EMA-enabled Trainer
    trainer = DITTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        pad_token_id=tokenizer.pad_token_id,
        ema_decay=0.9999,
    )

    # 10. Start Training
    print("Starting training with Accelerate + FSDP...")
    resume_from_checkpoint: Optional[str] = cfg.training.get("ckpt")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 11. Save Final Model (only on the main process)
    if accelerator.is_main_process:
        print("Training finished! Saving final model...")
        trainer.save_model()
        OmegaConf.save(cfg, os.path.join(training_args.output_dir, "config.yaml"))
        print(f"Model saved to {training_args.output_dir}")
        wandb.finish()


if __name__ == "__main__":
    # Use argparse and manual Hydra initialization for dynamic config loading
    parser = argparse.ArgumentParser(description="D2PLM Training with FSDP and Accelerate")
    parser.add_argument(
        "--config_name", 
        type=str, 
        default="train_config",
        help="Name of the hydra config file to use (without .yaml extension)."
    )
    args, hydra_overrides = parser.parse_known_args()

    hydra.initialize(config_path="train_config", version_base=None)
    cfg: DictConfig = hydra.compose(config_name=args.config_name, overrides=hydra_overrides)
    
    main(cfg)


Step 5: Understanding and Activating the Data Pipeline Optimization
Agent Action: No new code implementation is required for this step. The goal is to ensure the correct parameters (dataloader_pin_memory=True and non_blocking=True in compute_metrics) are set as specified in the plan. The underlying libraries (PyTorch, Hugging Face Trainer) will handle the dual-stream execution automatically.

The concept of overlapping data transfer and computation via dual CUDA streams is a powerful optimization that is achieved implicitly through correct configuration, rather than explicit code.

dataloader_pin_memory=True: This setting in TrainingArguments is the foundational requirement. It instructs the DataLoader to place fetched data batches into "pinned memory." Pinned memory is a special region of CPU RAM that the OS guarantees will not be paged to disk, providing a stable physical address for the GPU's Direct Memory Access (DMA) engine.

non_blocking=True: This argument, used in .to(device, non_blocking=True), is the trigger for asynchronous transfer. This call can only be truly non-blocking if the source tensor resides in pinned memory. When called, it initiates the data transfer via the DMA engine and immediately returns control to the CPU, which can then proceed with other tasks, such as preparing the next batch.

Hugging Face Trainer Internals: The Trainer is designed to leverage this mechanism. When it detects dataloader_pin_memory=True, its internal logic for moving data to the GPU will automatically use the non_blocking=True flag, thus enabling the asynchronous pipeline.

The resulting pipeline operates as follows:

| Timeslice | GPU Compute (Default Stream) | HtoD Copy (Copy Stream) | CPU Worker Threads |
| T0 | Begins computing Batch 1 | Initiates async copy of Batch 2 | Preparing Batch 3 |
| T1 | Finishes Batch 1, begins Batch 2 | Finishes Batch 2, initiates async copy of Batch 3 | Preparing Batch 4 |
| T2 | Finishes Batch 2, begins Batch 3 | Finishes Batch 3, initiates async copy of Batch 4 | Preparing Batch 5 |

This elegant orchestration ensures that the data for the next step is already arriving on the GPU while the current step is still being processed. The data transfer latency is effectively hidden behind the computation time, leading to a significant increase in training throughput by keeping the GPU constantly fed and active.

