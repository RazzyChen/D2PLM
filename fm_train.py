#!/usr/bin/env python3
# Flow Matching Training Script - Completely Independent from Diffusion
# Based on Discrete Absorbing Flow Matching (arXiv:2407.15595v2)

import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer, TrainingArguments

from model.backbone.dit_config import DITConfig
from model.backbone.dit_model import DITModel
from model.backbone.flow_matching_scheduler import (
    DiscreteAbsorbingFlowMatchingScheduler,
)
from model.dataloader.DataPipe import load_and_preprocess_data
from model.trainer.FMTrainer import FMDataCollator, FMTrainer


def create_dit_model_and_tokenizer(cfg: DictConfig):
    """Create DIT model and tokenizer for flow matching training"""

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


def create_flow_matching_scheduler(cfg: DictConfig, model_config):
    """Create flow matching scheduler"""
    print("Creating Flow Matching Scheduler...")
    scheduler = DiscreteAbsorbingFlowMatchingScheduler(
        vocab_size=model_config.vocab_size,
        absorbing_token_id=model_config.mask_token_id,
        num_flow_steps=cfg.flow_matching.num_flow_steps,
        flow_schedule=cfg.flow_matching.flow_schedule,
        min_flow_time=cfg.flow_matching.min_flow_time,
        max_flow_time=cfg.flow_matching.max_flow_time,
    )
    return scheduler


def prepare_dataset(cfg: DictConfig, tokenizer):
    """Prepare datasets for flow matching training"""
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
    Main Flow Matching training function.

    This function handles the complete flow matching training pipeline:
    - Dynamic run name and WandB ID generation for flow matching runs
    - Accelerator initialization for distributed training
    - TrainingArguments configuration
    - WandB initialization (main process only)
    - Seeding for reproducibility
    - Flow matching model, tokenizer, and scheduler creation
    - Dataset preparation
    - Flow matching trainer instantiation
    - Training execution
    - Final model saving

    Args:
        cfg (DictConfig): Flow matching configuration from Hydra.
    """

    # -- Dynamic Run Name Generation for Flow Matching --
    timestamp = (datetime.now() + timedelta(hours=8)).strftime("%H%M%m%d%Y")
    run_name_prefix = cfg.wabdb.get("run_name_prefix", "flow_matching_run")
    run_name = f"{run_name_prefix}_{timestamp}"
    print(f"Generated Flow Matching Run Name: {run_name}")

    # -- WandB Run ID Management --
    OmegaConf.set_struct(cfg, False)
    if "wandb_run_id" not in cfg or cfg.wandb_run_id is None:
        cfg.wandb_run_id = run_name
    OmegaConf.set_struct(cfg, True)
    print(f"WandB Run ID: {cfg.wandb_run_id}")

    # Initialize Accelerator
    accelerator = Accelerator()

    # 1. Configure TrainingArguments for Flow Matching
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_args = TrainingArguments(
        output_dir=cfg.weight.output_dir,
        per_device_train_batch_size=cfg.data.batch_size,
        per_device_eval_batch_size=cfg.data.batch_size,
        fp16=(cfg.system.mixed_precision == "fp16"),
        dataloader_num_workers=cfg.system.dataloader_num_workers,
        dataloader_pin_memory=True,  # Enable async data transfer
        **training_args_dict,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=cfg.logging.report_to,
        save_safetensors=True,
    )

    # 2. Initialize WandB for Flow Matching (main process only)
    if accelerator.is_main_process:
        wandb.init(
            project=cfg.wabdb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            id=cfg.wandb_run_id,
            resume="allow",
        )

    # 3. Set Random Seed
    torch.manual_seed(cfg.system.seed)

    # 4. Create Model and Tokenizer
    model, tokenizer, model_config = create_dit_model_and_tokenizer(cfg)

    # 5. Create Flow Matching Scheduler
    scheduler = create_flow_matching_scheduler(cfg, model_config)

    # 6. Prepare Datasets
    train_dataset, eval_dataset = prepare_dataset(cfg, tokenizer)

    # 7. Create Flow Matching Data Collator
    data_collator = FMDataCollator(
        tokenizer,
        scheduler,
        model_config.mask_token_id,
    )

    # 8. Define Evaluation Metric (adapted for flow matching)
    perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)

    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute evaluation metrics for flow matching.

        Note: For flow matching, we use direct token prediction without shifting,
        so we adjust the metric computation accordingly.
        """
        logits, labels = eval_preds
        device = training_args.device
        logits_tensor = torch.from_numpy(logits).to(device, non_blocking=True)
        labels_tensor = torch.from_numpy(labels).to(device, non_blocking=True)

        # For flow matching, use direct prediction (no shifting like in diffusion)
        ppl = perplexity.to(device)(logits_tensor, labels_tensor)
        return {"perplexity": ppl.item()}

    # 9. Create the Flow Matching Trainer
    trainer = FMTrainer(
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

    # 10. Start Flow Matching Training
    print("Starting Flow Matching training with Accelerate + FSDP...")
    resume_from_checkpoint: Optional[str] = cfg.training.get("ckpt")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 11. Save Final Flow Matching Model (main process only)
    if accelerator.is_main_process:
        print("Flow Matching training finished! Saving final model...")
        trainer.save_model()
        OmegaConf.save(cfg, os.path.join(training_args.output_dir, "fm_config.yaml"))
        print(f"Flow Matching model saved to {training_args.output_dir}")
        wandb.finish()


if __name__ == "__main__":
    # Flow Matching specific argument parsing
    parser = argparse.ArgumentParser(
        description="D2PLM Flow Matching Training with FSDP and Accelerate"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="FM_train_config",
        help="Name of the flow matching config file to use (without .yaml extension).",
    )
    args, hydra_overrides = parser.parse_known_args()

    hydra.initialize(config_path="train_config", version_base=None)
    cfg: DictConfig = hydra.compose(
        config_name=args.config_name, overrides=hydra_overrides
    )

    main(cfg)
