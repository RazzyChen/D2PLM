#!/usr/bin/env python3
"""
基于Hydra的DIT模型生成脚本
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from diffusers import DiffusionPipeline
import hydra
from omegaconf import DictConfig
import argparse
import os

from model.backbone.dit_config import DITConfig
from model.backbone.dit_model_hf import DITModel
from model.backbone.diffusion_scheduler_hf import DITDiffusionScheduler


class DITDiffusionPipeline(DiffusionPipeline):
    """
    基于Hugging Face Diffusers的DIT扩散管道
    """
    
    def __init__(self, model, scheduler, tokenizer):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = model.device
    
    def __call__(
        self,
        prompt_length: int = 100,
        num_inference_steps: int = 50,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        guidance_scale: float = 1.0,
        guidance_sequence: str = None,
        **kwargs
    ):
        """
        生成蛋白质序列
        """
        # 设置推理时间步
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 初始化完全被腐蚀的序列
        x_t = torch.full(
            (batch_size, prompt_length),
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # 如果提供了引导序列，编码它
        guidance_tokens = None
        if guidance_sequence:
            guidance_tokens = self.tokenizer.encode(
                guidance_sequence, 
                add_special_tokens=False
            )
        
        # 迭代去噪
        for i, t in enumerate(self.scheduler.timesteps):
            # 创建时间步张量
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            # 获取注意力掩码
            attention_mask = (x_t != self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            
            # 模型预测
            with torch.no_grad():
                outputs = self.model(
                    input_ids=x_t,
                    attention_mask=attention_mask,
                    timesteps=t_tensor,
                )
                
                # 获取logits
                logits = outputs.last_hidden_state
                
                # 应用温度
                logits = logits / temperature
                
                # Top-k采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(
                        logits, 
                        min(top_k, logits.size(-1)), 
                        dim=-1
                    )
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # 应用引导
                if guidance_tokens and guidance_scale > 1.0:
                    # 在引导位置强制使用引导序列的token
                    for j in range(min(len(guidance_tokens), prompt_length)):
                        logits[:, j, :] = float('-inf')
                        logits[:, j, guidance_tokens[j]] = 0
                
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个token
                next_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 
                    1
                ).view(batch_size, prompt_length)
                
                # 更新序列
                x_t = next_tokens
        
        # 解码生成的序列
        generated_sequences = []
        for i in range(batch_size):
            sequence = x_t[i].tolist()
            # 移除填充标记
            sequence = [token for token in sequence if token != self.tokenizer.pad_token_id]
            # 解码为氨基酸序列
            decoded_sequence = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_sequences.append(decoded_sequence)
        
        return generated_sequences


class ProteinGeneratorHF:
    """基于Hugging Face的蛋白质序列生成器"""
    
    def __init__(self, model_path, tokenizer_path, scheduler_path=None, device=None):
        """
        初始化生成器
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型配置
        self.config = DITConfig.from_pretrained(model_path)
        
        # 创建模型
        self.model = DITModel(self.config).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(
            torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
        )
        self.model.eval()
        
        # 创建或加载调度器
        if scheduler_path and os.path.exists(scheduler_path):
            self.scheduler = DITDiffusionScheduler.from_pretrained(scheduler_path)
        else:
            self.scheduler = DITDiffusionScheduler(
                num_train_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
            )
        
        # 创建扩散管道
        self.pipeline = DITDiffusionPipeline(
            self.model, 
            self.scheduler, 
            self.tokenizer
        )
        
        print(f"模型加载成功: {model_path}")
        print(f"词汇表大小: {len(self.tokenizer)}")
        print(f"最大序列长度: {self.config.max_position_embeddings}")
    
    def generate(
        self,
        seq_length: int = 100,
        num_sequences: int = 1,
        num_inference_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        guidance_sequence: str = None,
        guidance_scale: float = 1.0,
    ):
        """
        生成蛋白质序列
        """
        print(f"生成 {num_sequences} 个长度为 {seq_length} 的序列...")
        
        sequences = self.pipeline(
            prompt_length=seq_length,
            num_inference_steps=num_inference_steps,
            batch_size=num_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_sequence=guidance_sequence,
            guidance_scale=guidance_scale,
        )
        
        return sequences


@hydra.main(version_base=None, config_path="train_config", config_name="dit_hf_config")
def main(cfg: DictConfig):
    """主生成函数"""
    
    parser = argparse.ArgumentParser(description="使用Hydra配置生成蛋白质序列")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer路径")
    parser.add_argument("--scheduler_path", type=str, default=None, help="调度器路径")
    parser.add_argument("--output_file", type=str, default="generated_sequences_hydra.fasta", help="输出文件路径")
    parser.add_argument("--guidance_sequence", type=str, default=None, help="引导序列")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 {args.model_path} 不存在")
        return
    
    if not os.path.exists(args.tokenizer_path):
        print(f"错误: Tokenizer路径 {args.tokenizer_path} 不存在")
        return
    
    # 创建生成器
    generator = ProteinGeneratorHF(
        args.model_path, 
        args.tokenizer_path, 
        args.scheduler_path
    )
    
    # 生成序列
    sequences = generator.generate(
        seq_length=cfg.generation.seq_length,
        num_sequences=cfg.generation.num_sequences,
        num_inference_steps=cfg.generation.num_inference_steps,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        guidance_sequence=args.guidance_sequence,
        guidance_scale=cfg.generation.guidance_scale,
    )
    
    # 保存生成的序列
    with open(args.output_file, 'w') as f:
        for i, sequence in enumerate(sequences):
            f.write(f">generated_sequence_hydra_{i+1}\n")
            f.write(f"{sequence}\n")
    
    print(f"生成了 {len(sequences)} 个序列，已保存到 {args.output_file}")
    
    # 打印前几个序列作为示例
    print("\n生成的序列示例:")
    for i, sequence in enumerate(sequences[:3]):
        print(f"序列 {i+1}: {sequence}")


if __name__ == "__main__":
    main() 