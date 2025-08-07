import math
import torch
from diffusers import SchedulerMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from typing import Optional, Tuple, Union


class DiffusionSchedulerOutput(BaseOutput):
    """
    扩散调度器的输出
    """
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class DITDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    基于Hugging Face Diffusers的吸收扩散调度器
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        steps_offset: int = 0,
    ):
        super().__init__()
        
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.steps_offset = steps_offset
        
        # 计算噪声调度
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 预计算一些值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 计算q(x_t | x_0)的方差
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """余弦噪声调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor, mask_token_id: int) -> torch.Tensor:
        """
        添加噪声到序列
        
        Args:
            original_samples: 原始序列 [batch_size, seq_length]
            timesteps: 时间步 [batch_size]
            mask_token_id: 吸收态标记ID
            
        Returns:
            noisy_samples: 被腐蚀的序列 [batch_size, seq_length]
        """
        batch_size = original_samples.shape[0]
        
        # 获取当前时间步的alpha值
        alpha_t = self.alphas_cumprod[timesteps].view(batch_size, 1)
        
        # 计算腐蚀概率
        corruption_prob = 1.0 - alpha_t
        
        # 生成随机数
        random_values = torch.rand_like(original_samples.float())
        
        # 根据概率决定是否腐蚀为吸收态
        mask_condition = random_values < corruption_prob
        
        # 创建被腐蚀的序列
        noisy_samples = original_samples.clone()
        noisy_samples[mask_condition] = mask_token_id
        
        return noisy_samples, mask_condition
    
    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """随机采样时间步"""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
    
    def get_alpha_t(self, timesteps: torch.Tensor) -> torch.Tensor:
        """获取时间步t的alpha值"""
        return self.alphas_cumprod[timesteps]
    
    def get_alpha_t_prev(self, timesteps: torch.Tensor) -> torch.Tensor:
        """获取时间步t-1的alpha值"""
        return self.alphas_cumprod_prev[timesteps]
    
    def get_beta_t(self, timesteps: torch.Tensor) -> torch.Tensor:
        """获取时间步t的beta值"""
        return self.betas[timesteps]
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DiffusionSchedulerOutput, Tuple]:
        """
        单步去噪
        
        Args:
            model_output: 模型输出
            timestep: 当前时间步
            sample: 当前样本
            return_dict: 是否返回字典格式
            
        Returns:
            DiffusionSchedulerOutput或tuple
        """
        # 对于吸收扩散，我们直接使用模型输出来预测原始序列
        # 然后根据预测计算下一步的样本
        
        # 获取当前时间步的参数
        alpha_t = self.alphas_cumprod[timestep]
        alpha_t_prev = self.alphas_cumprod_prev[timestep]
        beta_t = self.betas[timestep]
        
        # 从模型输出中获取预测的原始序列
        pred_original_sample = model_output.argmax(dim=-1)
        
        # 计算下一步的样本
        # 这里简化处理，直接使用预测的原始序列
        prev_sample = pred_original_sample
        
        if not return_dict:
            return (prev_sample, pred_original_sample)
        
        return DiffusionSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )
    
    def add_noise_to_sample(
        self,
        original_samples: torch.Tensor,
        timesteps: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """
        为样本添加噪声（别名方法，保持兼容性）
        """
        return self.add_noise(original_samples, timesteps, mask_token_id)
    
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        缩放模型输入（对于吸收扩散，直接返回原样本）
        """
        return sample
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置推理时间步
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps, dtype=torch.long
        )
        if device is not None:
            self.timesteps = self.timesteps.to(device) 