# model/trainer/cpu_ema.py
# CPU-based Exponential Moving Average for efficient memory usage

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from collections import OrderedDict
import copy


class CPUEMAModel:
    """
    CPU-based Exponential Moving Average model for efficient GPU memory usage.
    
    This implementation stores EMA weights on CPU to save GPU memory while
    providing functionality to temporarily load weights to GPU for evaluation.
    
    Args:
        model: The PyTorch model to track
        decay: EMA decay factor (default: 0.9999)
        device: Device to store EMA weights ('cpu' recommended)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = 'cpu'):
        self.decay = decay
        self.device = device
        self.step_count = 0
        
        # Store EMA parameters on CPU
        self.ema_params = OrderedDict()
        self._initialize_ema_params(model)
        
    def _initialize_ema_params(self, model: nn.Module):
        """Initialize EMA parameters by copying model parameters to CPU."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Copy parameter to CPU and store
                    self.ema_params[name] = param.detach().cpu().clone()
    
    def step(self, model: nn.Module):
        """
        Update EMA parameters with current model parameters.
        
        Args:
            model: Current model with updated parameters
        """
        self.step_count += 1
        
        # Compute bias correction for early training steps
        bias_correction = 1.0 - self.decay ** self.step_count
        corrected_decay = 1.0 - (1.0 - self.decay) / bias_correction
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    # Move current parameter to CPU for computation
                    current_param_cpu = param.detach().cpu()
                    
                    # Update EMA: ema = decay * ema + (1 - decay) * current
                    self.ema_params[name].mul_(corrected_decay).add_(
                        current_param_cpu, alpha=1.0 - corrected_decay
                    )
    
    def copy_to_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Copy EMA parameters to model (temporarily loads to GPU).
        
        Args:
            model: Model to copy EMA parameters to
            
        Returns:
            Dictionary of original parameters for restoration
        """
        original_params = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    # Store original parameter for restoration
                    original_params[name] = param.detach().clone()
                    
                    # Copy EMA parameter to model's device
                    ema_param = self.ema_params[name].to(param.device)
                    param.copy_(ema_param)
        
        return original_params
    
    def restore_model(self, model: nn.Module, original_params: Dict[str, torch.Tensor]):
        """
        Restore original parameters to model.
        
        Args:
            model: Model to restore
            original_params: Dictionary of original parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in original_params:
                    param.copy_(original_params[name])
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for checkpointing."""
        return {
            'ema_params': self.ema_params,
            'decay': self.decay,
            'step_count': self.step_count,
            'device': self.device
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        self.ema_params = state_dict['ema_params']
        self.decay = state_dict['decay']
        self.step_count = state_dict['step_count']
        self.device = state_dict.get('device', 'cpu')
    
    def to(self, device: str):
        """Move EMA parameters to specified device."""
        for name in self.ema_params:
            self.ema_params[name] = self.ema_params[name].to(device)
        self.device = device
    
    def __repr__(self):
        return f"CPUEMAModel(decay={self.decay}, step_count={self.step_count}, device={self.device})"


class EMAContextManager:
    """
    Context manager for temporarily using EMA model during evaluation.
    
    Usage:
        with EMAContextManager(model, ema_model):
            # Model now uses EMA weights
            outputs = model(inputs)
        # Model restored to original weights
    """
    
    def __init__(self, model: nn.Module, ema_model: CPUEMAModel):
        self.model = model
        self.ema_model = ema_model
        self.original_params = None
    
    def __enter__(self):
        """Enter context: apply EMA weights to model."""
        self.original_params = self.ema_model.copy_to_model(self.model)
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore original weights."""
        if self.original_params is not None:
            self.ema_model.restore_model(self.model, self.original_params)
            self.original_params = None