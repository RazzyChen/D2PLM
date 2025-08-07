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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DITAttention(nn.Module):
    def __init__(self, config: DITConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim, max_seq_len=config.max_position_embeddings
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        query_states = self.rope(query_states)
        key_states = self.rope(key_states)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class DITEncoderLayer(nn.Module):
    def __init__(self, config: DITConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DITAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            SwiGLU(),
            nn.Linear(config.intermediate_size, self.hidden_size),
        )
        self.norm1 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.adaLN_modulation = nn.Sequential(
            SwiGLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

    def forward(
        self, hidden_states: torch.Tensor, t_emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Attention block
        residual = hidden_states
        hidden_states_norm = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.self_attn(hidden_states_norm, attention_mask=attention_mask)
        hidden_states = residual + gate_msa.unsqueeze(1) * attn_output

        # MLP block
        residual = hidden_states
        hidden_states_norm = modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + gate_mlp.unsqueeze(1) * mlp_output

        return hidden_states


class DITModel(PreTrainedModel):
    config_class = DITConfig
    base_model_prefix = "dit"
    supports_gradient_checkpointing = True

    def __init__(self, config: DITConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embeddings = nn.Embedding(
            config.vocab_size, self.hidden_size, padding_idx=config.pad_token_id
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.config.time_embedding_dim, self.hidden_size),
            SwiGLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        self.layers = nn.ModuleList(
            [DITEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.final_layer_adaLN_modulation = nn.Sequential(
            SwiGLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)

        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_init_weights)

        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_adaLN_modulation[-1].bias, 0)

        self.lm_head.weight = self.embeddings.weight

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _get_sinusoidal_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.config.time_embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=self.embeddings.weight.dtype) * -emb)
        emb = timesteps[:, None].to(self.embeddings.weight.dtype) * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.config.time_embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if timesteps is None:
            raise ValueError("timesteps must be provided for diffusion models")

        t_emb = self._get_sinusoidal_time_embedding(timesteps)
        t_emb = self.time_embedding(t_emb)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, t_emb, attention_mask=attention_mask)

        shift, scale = self.final_layer_adaLN_modulation(t_emb).chunk(2, dim=1)
        sequence_output = modulate(self.final_layer_norm(hidden_states), shift, scale)
        
        logits = self.lm_head(sequence_output)

        if not return_dict:
            output = (logits,) + ((all_hidden_states,) if output_hidden_states else ())
            return tuple(v for v in output if v is not None)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_parameters_m": total_params / 1_000_000,
            "trainable_parameters_m": trainable_params / 1_000_000,
        }