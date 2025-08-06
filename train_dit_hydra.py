#!/usr/bin/env python3
"""
基于Hydra的DIT模型训练脚本
"""

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from model.backbone.diffusion_scheduler import DITDiffusionScheduler
from model.backbone.dit_config import DITConfig
from model.backbone.dit_model import DITModel
from model.dataloader.DataPipe import load_and_preprocess_data


class DITDataCollator:
    """DIT模型的数据整理器"""

    def __init__(self, tokenizer, scheduler, mask_token_id):
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.mask_token_id = mask_token_id

    def __call__(self, features):
        # 提取输入ID和注意力掩码
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        attention_mask = torch.stack(
            [torch.tensor(f["attention_mask"]) for f in features]
        )

        batch_size = input_ids.shape[0]

        # 随机采样时间步
        timesteps = self.scheduler.get_timesteps(batch_size, input_ids.device)

        # 添加噪声
        noisy_input_ids = self.scheduler.add_noise(
            input_ids, timesteps, self.mask_token_id
        )

        return {
            "input_ids": noisy_input_ids,
            "attention_mask": attention_mask,
            "timesteps": timesteps,
            "labels": input_ids,  # 原始序列作为标签
        }


class DITTrainer(Trainer):
    """自定义DIT训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失"""
        # 获取输入
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        timesteps = inputs["timesteps"]
        labels = inputs["labels"]

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )

        # 获取logits
        logits = outputs.last_hidden_state

        # 计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def create_dit_model_and_tokenizer(cfg: DictConfig):
    """创建DIT模型和tokenizer"""

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer词汇表大小: {len(tokenizer)}")

    # 创建模型配置
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
    )

    # 创建模型
    model = DITModel(config)

    # 统计参数
    param_info = model.count_parameters()
    print(
        f"模型总参数量: {param_info['total_parameters']:,} ({param_info['total_parameters_m']:.1f}M)"
    )

    return model, tokenizer, config


def create_diffusion_scheduler(cfg: DictConfig):
    """创建扩散调度器"""
    scheduler = DITDiffusionScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        steps_offset=cfg.diffusion.steps_offset,
    )
    return scheduler


def prepare_dataset(cfg: DictConfig, tokenizer):
    """准备数据集"""
    print("加载和预处理数据...")

    # 加载数据集
    dataset = load_and_preprocess_data(
        cfg.data.lmdb_path,
        tokenizer,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        cache_dir=cfg.data.cache_dir,
    )

    print(f"数据集大小: {len(dataset)}")
    return dataset


@hydra.main(version_base=None, config_path="train_config", config_name="train_config")
def main(cfg: DictConfig):
    """主训练函数"""

    # 设置随机种子
    torch.manual_seed(cfg.system.seed)

    # 设置设备
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和tokenizer
    model, tokenizer, model_config = create_dit_model_and_tokenizer(cfg)

    # 创建扩散调度器
    scheduler = create_diffusion_scheduler(cfg)

    # 准备数据集
    dataset = prepare_dataset(cfg, tokenizer)

    # 创建数据整理器
    data_collator = DITDataCollator(tokenizer, scheduler, tokenizer.mask_token_id)

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=cfg.output.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.data.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        save_total_limit=cfg.training.save_total_limit,
        dataloader_num_workers=cfg.system.dataloader_num_workers,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=cfg.logging.report_to,
        # 优化设置
        optim="adamw_torch",
        fp16=cfg.system.mixed_precision == "fp16",
        gradient_checkpointing=True,
        max_grad_norm=cfg.training.max_grad_norm,
        # 评估设置
        evaluation_strategy=cfg.evaluation.eval_strategy,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        # 保存设置
        save_strategy=cfg.checkpointing.save_strategy,
        save_safetensors=True,
    )

    # 创建训练器
    trainer = DITTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    final_save_dir = f"{cfg.output.output_dir}/final_model"
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)

    # 保存调度器
    scheduler.save_pretrained(final_save_dir)

    # 保存配置
    OmegaConf.save(cfg, f"{final_save_dir}/config.yaml")

    print(f"训练完成！模型已保存到 {final_save_dir}")

    return trainer, model, tokenizer, scheduler


if __name__ == "__main__":
    main()
