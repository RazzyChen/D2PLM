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
from model.backbone.flow_matching_scheduler import DiscreteAbsorbingFlowMatchingScheduler
from model.backbone.dit_config import DITConfig
from model.backbone.dit_model import DITModel
from model.dataloader.DataPipe import load_and_preprocess_data
# Import the newly modularized Trainer and DataCollator
from model.trainer.DITTrainer import DITDataCollator, DITTrainer
from omegaconf import DictConfig, OmegaConf
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset


def create_dit_model_and_tokenizer(cfg: DictConfig):
    """Create DIT model and tokenizer"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create model configuration
    config = DITConfig(
        vocab_size=cfg.model.vocab_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        time_embedding_dim=cfg.model.time_embedding_dim,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        layer_norm_eps=cfg.model.layer_norm_eps,
        initializer_range=cfg.model.initializer_range,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id
        if hasattr(tokenizer, "mask_token_id")
        else 1,
        cls_token_id=tokenizer.cls_token_id
        if hasattr(tokenizer, "cls_token_id")
        else 2,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Create model
    model = DITModel(config)

    # Count parameters
    param_info = model.count_parameters()
    print(
        f"Total model parameters: {param_info['total_parameters']:,} ({param_info['total_parameters_m']:.1f}M)"
    )

    return model, tokenizer, config


def create_scheduler(cfg: DictConfig, model_config):
    """Create scheduler based on configuration - supports both diffusion and flow matching"""
    # Check if scheduler config exists and has flow matching setting
    use_flow_matching = False
    if hasattr(cfg, 'scheduler') and cfg.scheduler:
        use_flow_matching = cfg.scheduler.get("use_flow_matching_scheduler", False)
    
    if use_flow_matching:
        print("Using Flow Matching Scheduler...")
        scheduler = DiscreteAbsorbingFlowMatchingScheduler(
            vocab_size=model_config.vocab_size,
            absorbing_token_id=model_config.mask_token_id,
            num_flow_steps=cfg.flow_matching.num_flow_steps,
            flow_schedule=cfg.flow_matching.flow_schedule,
            min_flow_time=cfg.flow_matching.min_flow_time,
            max_flow_time=cfg.flow_matching.max_flow_time,
        )
    else:
        print("Using Traditional Diffusion Scheduler...")
        scheduler = DITDiffusionScheduler(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            steps_offset=cfg.diffusion.steps_offset,
        )
    
    return scheduler, use_flow_matching


def prepare_dataset(cfg: DictConfig, tokenizer):
    """Prepare datasets"""
    print("--- Loading Training Dataset ---")
    train_dataset = load_and_preprocess_data(
        lmdb_path=cfg.data.train_lmdb_path,
        tokenizer=tokenizer,
        cache_dir=cfg.data.train_cache_dir,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
    )
    print(f"Training set size: {len(train_dataset)}")

    print("\n--- Loading Validation Dataset ---")
    eval_dataset = load_and_preprocess_data(
        lmdb_path=cfg.data.val_lmdb_path,
        tokenizer=tokenizer,
        cache_dir=cfg.data.val_cache_dir,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
    )
    print(f"Validation set size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


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

    # 5. Create Scheduler (Diffusion or Flow Matching)
    scheduler, use_flow_matching = create_scheduler(cfg, model_config)

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

    # 9. Create the EMA-enabled Trainer with scheduler support
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
        use_flow_matching=use_flow_matching,  # Pass the scheduler type
    )

    # 10. Start Training
    scheduler_type = "Flow Matching" if use_flow_matching else "Diffusion"
    print(f"Starting training with Accelerate + FSDP + {scheduler_type}...")
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