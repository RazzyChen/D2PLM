import os
import time
import psutil
import torch

import hydra
import ray
import ray.train.huggingface.transformers
import yaml
from omegaconf import DictConfig, OmegaConf
from ray.train import ScalingConfig
from transformers import (
    AutoTokenizer,
    EsmConfig,
    Trainer,
    TrainingArguments,
)

import wandb
from model.backbone.esm2ae import ESM2AE
from model.dataloader.DataPipe import load_and_preprocess_data
from model.utils.ModelSave import save_model
from model.utils.MyLRCallback import LogLearningRateCallback


def get_memory_usage():
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }


def train_func(config: dict):
    """Training function to be executed by each Ray worker."""
    cfg = OmegaConf.create(config)
    
    # Performance monitoring setup
    start_time = time.time()
    initial_memory = get_memory_usage()

    # Load DeepSpeed config
    with open(cfg.deepspeed.config_path, "r") as f:
        ds_config = yaml.safe_load(f)

    # Initialize wandb on the main worker
    if ray.train.get_context().get_world_rank() == 0:
        wandb.init(
            project=cfg.wandb.project,
            resume=cfg.wandb.resume,
            mode="offline",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"Initial memory usage: {initial_memory}")

    # Load configuration and tokenizer with optimizations
    model_config = EsmConfig.from_pretrained(
        cfg.model.pretrained_model_name,
        position_embedding_type=cfg.model.position_embedding_type,
        num_labels=cfg.model.num_labels,
        problem_type=cfg.model.problem_type,
        attn_implementation=cfg.model.attn_implementation,
        torch_dtype=getattr(torch, cfg.model.get('torch_dtype', 'float16')),
        
        # Add optimization configurations
        freeze_backbone=cfg.model.get('freeze_backbone', False),
        use_feature_cache=cfg.model.get('use_feature_cache', True),
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)

    # Initialize model with optimized config
    model = ESM2AE(model_config).cuda()
    
    # Print model statistics
    if ray.train.get_context().get_world_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

    # Load and preprocess data with optimizations
    train_mdb_path = cfg.data.train_path
    if not os.path.exists(train_mdb_path):
        raise FileNotFoundError(
            f"Training dataset path {train_mdb_path} does not exist."
        )
    
    data_loading_start = time.time()
    tokenized_train_dataset = load_and_preprocess_data(
        train_mdb_path, 
        tokenizer, 
        cfg.data.cache_dir,
        max_length=cfg.data.get('max_length', 512),
        batch_size=cfg.data.get('batch_size', 1000),
        preprocessing_num_proc=cfg.data.get('preprocessing_num_proc', None)
    )
    data_loading_time = time.time() - data_loading_start
    
    if ray.train.get_context().get_world_rank() == 0:
        print(f"Data loading completed in {data_loading_time:.2f} seconds")
        print(f"Dataset size: {len(tokenized_train_dataset):,} samples")

    # Define TrainingArguments with optimizations
    training_args = TrainingArguments(
        # Basic settings
        output_dir=cfg.trainer.output_dir,
        save_strategy=cfg.trainer.get('save_strategy', 'steps'),
        save_steps=cfg.trainer.get('save_steps', 500),
        learning_rate=cfg.trainer.learning_rate,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        num_train_epochs=cfg.trainer.num_train_epochs,
        max_steps=cfg.trainer.get('max_steps', -1),
        
        # Reproducibility
        seed=cfg.trainer.seed,
        data_seed=cfg.trainer.data_seed,
        
        # Data loading optimizations
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        dataloader_prefetch_factor=cfg.trainer.get('dataloader_prefetch_factor', 4),
        dataloader_pin_memory=cfg.trainer.get('dataloader_pin_memory', True),
        dataloader_drop_last=cfg.trainer.get('dataloader_drop_last', True),
        dataloader_persistent_workers=cfg.trainer.get('dataloader_persistent_workers', True),
        
        # Performance optimizations
        group_by_length=cfg.trainer.get('group_by_length', True),
        length_column_name=cfg.trainer.get('length_column_name', 'length'),
        remove_unused_columns=cfg.trainer.get('remove_unused_columns', False),
        prediction_loss_only=cfg.trainer.get('prediction_loss_only', True),
        skip_memory_metrics=cfg.trainer.get('skip_memory_metrics', True),
        
        # Mixed precision and computation
        tf32=cfg.trainer.get('tf32', True),
        bf16=cfg.trainer.get('bf16', True),
        fp16=cfg.trainer.get('fp16', False),
        gradient_checkpointing=cfg.trainer.get('gradient_checkpointing', True),
        
        # Optimization settings
        optim=cfg.trainer.optim,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        max_grad_norm=cfg.trainer.get('max_grad_norm', 1.0),
        
        # Learning rate and scheduler
        warmup_ratio=cfg.trainer.warmup_ratio,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        
        # Regularization
        weight_decay=cfg.trainer.weight_decay,
        adam_beta1=cfg.trainer.get('adam_beta1', 0.9),
        adam_beta2=cfg.trainer.adam_beta2,
        adam_epsilon=cfg.trainer.get('adam_epsilon', 1e-8),
        
        # Logging and evaluation
        logging_dir=cfg.trainer.logging_dir,
        logging_steps=cfg.trainer.logging_steps,
        eval_steps=cfg.trainer.get('eval_steps', 1000),
        evaluation_strategy=cfg.trainer.get('evaluation_strategy', 'no'),
        metric_for_best_model=cfg.trainer.metric_for_best_model,
        
        # Saving and checkpointing
        save_safetensors=cfg.trainer.save_safetensors,
        save_total_limit=cfg.trainer.save_total_limit,
        greater_is_better=cfg.trainer.greater_is_better,
        load_best_model_at_end=cfg.trainer.load_best_model_at_end,
        
        # Reporting
        push_to_hub=cfg.trainer.push_to_hub,
        report_to=cfg.trainer.report_to,
        
        # DeepSpeed configuration
        deepspeed=ds_config,
    )

    # Define callbacks with performance monitoring
    callbacks = [
        LogLearningRateCallback(),
        ray.train.huggingface.transformers.RayTrainReportCallback(),
    ]

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Prepare trainer for Ray
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    
    # Performance monitoring before training
    if ray.train.get_context().get_world_rank() == 0:
        setup_time = time.time() - start_time
        print(f"Setup completed in {setup_time:.2f} seconds")
        print(f"Starting training with {len(tokenized_train_dataset):,} samples")
        current_memory = get_memory_usage()
        print(f"Pre-training memory usage: {current_memory}")

    # Start training
    training_start = time.time()
    trainer.train()
    training_time = time.time() - training_start

    # Performance monitoring after training
    if ray.train.get_context().get_world_rank() == 0:
        final_memory = get_memory_usage()
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final memory usage: {final_memory}")
        
        # Log performance metrics to wandb
        wandb.log({
            "performance/setup_time_seconds": setup_time,
            "performance/training_time_seconds": training_time,
            "performance/data_loading_time_seconds": data_loading_time,
            "performance/final_memory_mb": final_memory["memory_mb"],
            "performance/final_gpu_memory_mb": final_memory["gpu_memory_mb"],
            "performance/samples_per_second": len(tokenized_train_dataset) * cfg.trainer.num_train_epochs / training_time,
        })

    # Save model on the main worker
    if ray.train.get_context().get_world_rank() == 0:
        save_dir = "./saved_model"
        save_model(trainer, save_dir)
        
        # Clear model cache to free memory
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
            cache_info = model.get_cache_info() if hasattr(model, 'get_cache_info') else {}
            print(f"Model cache info: {cache_info}")
        
        wandb.finish()


@hydra.main(
    version_base=None, config_path="./train_config", config_name="trainer_config"
)
def main(cfg: DictConfig):
    # Initialize Ray with optimizations
    ray.init(
        object_store_memory=int(2e9),  # 2GB object store
        dashboard_host="0.0.0.0",
        include_dashboard=False,  # Disable dashboard for better performance
    )

    # Print configuration summary
    print("=" * 60)
    print("OPTIMIZED ESM2AE TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {cfg.model.pretrained_model_name}")
    print(f"Attention: {cfg.model.attn_implementation}")
    print(f"Max sequence length: {cfg.data.get('max_length', 512)}")
    print(f"Batch size per device: {cfg.trainer.per_device_train_batch_size}")
    print(f"Gradient accumulation: {cfg.trainer.gradient_accumulation_steps}")
    print(f"Learning rate: {cfg.trainer.learning_rate}")
    print(f"Mixed precision: bf16={cfg.trainer.get('bf16', True)}")
    print(f"Workers: {cfg.ray.num_workers}")
    print("=" * 60)

    # Use Ray TorchTrainer with optimized settings
    from ray.train.torch import TorchTrainer

    ray_trainer = TorchTrainer(
        train_func,
        train_loop_config=OmegaConf.to_container(cfg, resolve=True),
        scaling_config=ScalingConfig(
            num_workers=cfg.ray.num_workers,
            resources_per_worker=cfg.ray.get('resources_per_worker', {
                "CPU": 11,
                "GPU": 1,
            }),
            use_gpu=cfg.ray.use_gpu,
            placement_strategy=cfg.ray.get('scaling_config', {}).get('placement_strategy', 'PACK'),
        ),
        run_config=ray.train.RunConfig(
            name="esm2ae_optimized",
            # Add checkpointing configuration if needed
        ),
    )

    # Start training with performance monitoring
    print("Starting distributed training...")
    start_time = time.time()
    result = ray_trainer.fit()
    total_time = time.time() - start_time

    print(f"\nTraining completed successfully in {total_time:.2f} seconds!")
    print(f"Result: {result}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
