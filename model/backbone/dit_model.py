import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from ..utils.ActivationFunction import SwiGLU
from ..utils.RoPE import RotaryPositionalEmbedding
from .dit_config import DITConfig


def modulate(x, shift, scale):
    """Apply modulation to tensor x using shift and scale parameters.
    
    Args:
        x: Input tensor of shape [batch, seq_len, hidden_size]
        shift: Shift parameter of shape [batch, hidden_size]
        scale: Scale parameter of shape [batch, hidden_size]
    """
    # Use einops for clearer dimension expansion
    shift_expanded = rearrange(shift, 'batch hidden -> batch 1 hidden')
    scale_expanded = rearrange(scale, 'batch hidden -> batch 1 hidden')
    return x * (1 + scale_expanded) + shift_expanded


class DITAttention(nn.Module):
    """
    Multi-head attention module for DIT model with RoPE positional encoding.
    
    This attention module implements:
    - Multi-head self-attention mechanism
    - Rotary Position Embedding (RoPE) for positional awareness
    - Flash attention via scaled_dot_product_attention
    
    Args:
        config (DITConfig): Model configuration containing architecture parameters
    """
    
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

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention with RoPE positional encoding.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                           where 1 = attend, 0 = mask
                           
        Returns:
            torch.Tensor: Attention output of shape [batch_size, seq_len, hidden_size]
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for RoPE: [batch, seq_len, hidden] -> [batch, seq_len, num_heads, head_dim]
        query_states = rearrange(
            query_states, 
            'batch seq_len (num_heads head_dim) -> batch seq_len num_heads head_dim',
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim
        )
        key_states = rearrange(
            key_states,
            'batch seq_len (num_heads head_dim) -> batch seq_len num_heads head_dim',
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim
        )
        value_states = rearrange(
            value_states,
            'batch seq_len (num_heads head_dim) -> batch seq_len num_heads head_dim',
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim
        )

        # Apply RoPE positional encoding
        query_states = self.rope(query_states)
        key_states = self.rope(key_states)

        # Rearrange for attention: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = rearrange(query_states, 'batch seq_len num_heads head_dim -> batch num_heads seq_len head_dim')
        key_states = rearrange(key_states, 'batch seq_len num_heads head_dim -> batch num_heads seq_len head_dim')
        value_states = rearrange(value_states, 'batch seq_len num_heads head_dim -> batch num_heads seq_len head_dim')

        # Process attention_mask with einops for clarity
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand [batch, seq_len] -> [batch, 1, 1, seq_len] for scaled_dot_product_attention
                attention_mask = rearrange(attention_mask, 'batch seq_len -> batch 1 1 seq_len')
                # Convert to attention format (0 = attend, -inf = mask)
                attention_mask = attention_mask.to(dtype=query_states.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min

        # Execute attention computation
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )

        # Rearrange output: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = rearrange(
            attn_output,
            'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)'
        )
        attn_output = self.o_proj(attn_output)
        return attn_output


class DITEncoderLayer(nn.Module):
    """
    DIT Encoder Layer with Adaptive Layer Normalization (AdaLN).
    
    This layer implements:
    - Self-attention with RoPE positional encoding
    - MLP with SwiGLU activation functions
    - Adaptive Layer Normalization conditioned on time embeddings
    - Residual connections with learnable gating
    
    Args:
        config (DITConfig): Model configuration containing architecture parameters
    """
    
    def __init__(self, config: DITConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DITAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size * 2),
            SwiGLU(),
            nn.Linear(config.intermediate_size, config.intermediate_size * 2),
            SwiGLU(),
            nn.Linear(config.intermediate_size, self.hidden_size),
        )
        self.norm1 = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps
        )
        self.norm2 = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps
        )
        self.adaLN_modulation = nn.Sequential(
            SwiGLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply encoder layer with adaptive layer normalization.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            t_emb: Time embedding tensor of shape [batch_size, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Layer output of shape [batch_size, seq_len, hidden_size]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t_emb).chunk(6, dim=1)
        )

        # Attention block
        residual = hidden_states
        hidden_states_norm = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.self_attn(hidden_states_norm, attention_mask=attention_mask)
        # Apply gating with einops for clarity
        gate_msa_expanded = rearrange(gate_msa, 'batch hidden -> batch 1 hidden')
        hidden_states = residual + gate_msa_expanded * attn_output

        # MLP block
        residual = hidden_states
        hidden_states_norm = modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        mlp_output = self.mlp(hidden_states_norm)
        # Apply gating with einops for clarity
        gate_mlp_expanded = rearrange(gate_mlp, 'batch hidden -> batch 1 hidden')
        hidden_states = residual + gate_mlp_expanded * mlp_output

        return hidden_states


class DITModel(PreTrainedModel):
    """
    DIT (Diffusion Transformer) Model for protein sequence generation.
    
    This model implements a diffusion transformer architecture for protein language modeling:
    - Token embeddings for amino acid sequences
    - Sinusoidal time embeddings for diffusion timesteps
    - Stack of transformer encoder layers with adaptive layer normalization
    - Language modeling head for next token prediction
    
    The model is designed to work with discrete diffusion processes where tokens
    are gradually corrupted and then denoised during generation.
    
    Args:
        config (DITConfig): Model configuration containing all architecture parameters
    """
    
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
            nn.Linear(self.config.time_embedding_dim, self.hidden_size * 2),
            SwiGLU(),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
        )

        self.layers = nn.ModuleList(
            [DITEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps
        )
        self.final_layer_adaLN_modulation = nn.Sequential(
            SwiGLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        """
        Initialize model weights using optimal strategies for SwiGLU + DiT architecture.
        
        Based on DiT paper (Meta) and SwiGLU best practices:
        - Embedding weights: Normal distribution (std=0.02)
        - Linear layers: Xavier uniform (optimal for SwiGLU networks)
        - AdaLN modulation layers: Zero initialization (DiT key insight)
        - Attention/MLP output projections: Scaled initialization for residual paths
        - LM head: Weight sharing with embeddings
        """
        # Embedding initialization
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

        # Standard layer initialization optimized for SwiGLU
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                # Xavier uniform is optimal for SwiGLU networks
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init_weights)

        # DiT-specific initialization: Zero initialization for AdaLN modulation layers
        # This is crucial for stable training as shown in DiT paper
        for layer in self.layers:
            # Zero-init AdaLN modulation (critical for DiT performance)
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
            
            # Scale down output projections for better residual learning
            # Following DiT best practices for residual connections
            with torch.no_grad():
                layer.self_attn.o_proj.weight *= 0.1
                layer.mlp[-1].weight *= 0.1

        # Zero-init final AdaLN modulation
        nn.init.constant_(self.final_layer_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_adaLN_modulation[-1].bias, 0)

        # Weight sharing between embeddings and LM head
        self.lm_head.weight = self.embeddings.weight

    def get_input_embeddings(self):
        """Get the input embedding layer."""
        return self.embeddings

    def set_input_embeddings(self, value):
        """Set the input embedding layer."""
        self.embeddings = value

    def _get_sinusoidal_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal time embeddings.
        
        Args:
            timesteps: Tensor of shape [batch_size]
            
        Returns:
            Time embeddings of shape [batch_size, time_embedding_dim]
        """
        # Ensure timesteps are in valid range [0, 1]
        timesteps = torch.clamp(timesteps, min=0.0, max=1.0)
        
        half_dim = self.config.time_embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(
                half_dim, device=timesteps.device, dtype=self.embeddings.weight.dtype
            )
            * -emb
        )
        # Use einops for clearer broadcasting
        timesteps_expanded = rearrange(timesteps.to(self.embeddings.weight.dtype), 'batch -> batch 1')
        emb_expanded = rearrange(emb, 'half_dim -> 1 half_dim')
        emb = timesteps_expanded * emb_expanded
        
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
        """
        Forward pass of the DIT model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len] where 1=attend, 0=mask
            timesteps: Diffusion timesteps of shape [batch_size]
            inputs_embeds: Pre-computed input embeddings of shape [batch_size, seq_len, hidden_size]
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return BaseModelOutput or tuple
            
        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer output [batch_size, seq_len, hidden_size]
                - hidden_states: All layer outputs if output_hidden_states=True
                - attentions: None (not computed)
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            # Validate input token IDs are within vocabulary range
            if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
                invalid_ids = input_ids[(input_ids < 0) | (input_ids >= self.config.vocab_size)]
                raise ValueError(f"Input contains invalid token IDs: {invalid_ids.unique()}. "
                               f"Valid range is [0, {self.config.vocab_size-1}]")
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
        """
        Count the total and trainable parameters in the model.
        
        Returns:
            dict: Dictionary containing parameter counts and their values in millions
                - total_parameters: Total number of parameters
                - trainable_parameters: Number of trainable parameters
                - total_parameters_m: Total parameters in millions
                - trainable_parameters_m: Trainable parameters in millions
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_parameters_m": total_params / 1_000_000,
            "trainable_parameters_m": trainable_params / 1_000_000,
        }