import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from ..utils.ActivationFunction import SwiGLU
from ..utils.RoPE import RotaryPositionalEmbedding
from .dit_config import DITConfig


class DITModel(PreTrainedModel):
    """
    吸收扩散蛋白质语言模型 (DIT - Discrete Diffusion Transformer)
    基于Transformer架构的离散扩散模型，用于蛋白质序列生成
    集成RoPE位置编码
    """

    config_class = DITConfig
    base_model_prefix = "dit"
    supports_gradient_checkpointing = True

    def __init__(self, config: DITConfig):
        super().__init__(config)
        self.config = config

        # 词嵌入层
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # RoPE位置嵌入层
        self.rope = RotaryPositionalEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
        )

        # 时间嵌入层 (用于扩散步骤)
        self.time_embedding = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_size),
            SwiGLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Transformer编码器层
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation="gelu",  # 使用标准gelu，SwiGLU在SwiGLU类中已优化
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.num_hidden_layers,
        )

        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # 语言模型头 (预测x_0的logits)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化模型权重"""
        # 词嵌入初始化
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

        # 时间嵌入初始化
        for module in self.time_embedding:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 语言模型头初始化
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # 将词嵌入权重绑定到语言模型头
        self.lm_head.weight = self.embeddings.weight

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _get_sinusoidal_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        生成正弦时间嵌入
        Args:
            timesteps: 时间步张量 [batch_size]
        Returns:
            时间嵌入张量 [batch_size, time_embedding_dim]
        """
        half_dim = self.config.time_embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def _apply_rope_to_attention(
        self, hidden_states: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        将RoPE应用到注意力层的查询和键
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            seq_len: 序列长度
        Returns:
            应用了RoPE的隐藏状态
        """
        batch_size, _, hidden_size = hidden_states.shape
        head_dim = hidden_size // self.config.num_attention_heads

        # 重塑为多头注意力格式
        hidden_states = hidden_states.view(
            batch_size, seq_len, self.config.num_attention_heads, head_dim
        )

        # 应用RoPE
        hidden_states = self.rope(hidden_states, seq_len)

        # 重塑回原始格式
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        前向传播
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            timesteps: 扩散时间步 [batch_size]
            position_ids: 位置ID [batch_size, seq_len] (RoPE中不使用)
            head_mask: 注意力头掩码
            inputs_embeds: 预计算的嵌入
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式
        Returns:
            模型输出
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        # 获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # 应用RoPE位置嵌入
        inputs_embeds = self._apply_rope_to_attention(inputs_embeds, seq_length)

        # 时间嵌入
        if timesteps is None:
            raise ValueError("timesteps must be provided for diffusion models")

        time_emb = self._get_sinusoidal_time_embedding(timesteps)
        time_emb = self.time_embedding(time_emb)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_length, -1)

        # 组合嵌入 (RoPE已应用到词嵌入中)
        hidden_states = inputs_embeds + time_emb

        # 应用dropout
        hidden_states = F.dropout(
            hidden_states, p=self.config.hidden_dropout_prob, training=self.training
        )

        # Transformer编码器
        encoder_outputs = self.encoder(
            hidden_states,
            src_key_padding_mask=attention_mask == 0,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
        else:
            sequence_output = encoder_outputs[0]

        # 最终层归一化
        sequence_output = self.final_layer_norm(sequence_output)

        # 语言模型头
        logits = self.lm_head(sequence_output)

        if not return_dict:
            return (logits,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从隐藏状态获取logits
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
        Returns:
            logits [batch_size, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)

    def count_parameters(self):
        """
        统计模型参数
        Returns:
            包含参数统计信息的字典
        """
        total_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "total_parameters_m": total_parameters / 1_000_000,
            "trainable_parameters_m": trainable_parameters / 1_000_000,
        }
