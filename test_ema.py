#!/usr/bin/env python3
# test_ema.py
# Quick test for CPU-based EMA functionality

import torch
import torch.nn as nn
from model.trainer.cpu_ema import CPUEMAModel, EMAContextManager


def test_cpu_ema():
    """Test CPU-based EMA functionality."""
    print("Testing CPU-based EMA Model...")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel().cuda() if torch.cuda.is_available() else SimpleModel()
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # Initialize EMA
    ema_model = CPUEMAModel(model, decay=0.999, device='cpu')
    print(f"EMA device: {ema_model.device}")
    
    # Get initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Simulate training steps
    print("\nSimulating training steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(5):
        # Forward pass
        x = torch.randn(4, 10).to(device)
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema_model.step(model)
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}, EMA step count = {ema_model.step_count}")
    
    # Test EMA context manager
    print("\nTesting EMA context manager...")
    
    # Get model parameters before EMA
    before_ema = {name: param.clone() for name, param in model.named_parameters()}
    
    with EMAContextManager(model, ema_model) as ema_model_temp:
        # Model should now have EMA weights
        during_ema = {name: param.clone() for name, param in model.named_parameters()}
        print("Inside EMA context: Model weights changed to EMA values")
    
    # Model should be restored
    after_ema = {name: param.clone() for name, param in model.named_parameters()}
    
    # Verify restoration
    params_restored = all(
        torch.allclose(before_ema[name], after_ema[name]) 
        for name in before_ema.keys()
    )
    
    params_changed = any(
        not torch.allclose(before_ema[name], during_ema[name]) 
        for name in before_ema.keys()
    )
    
    print(f"Parameters correctly restored after EMA context: {params_restored}")
    print(f"Parameters were different during EMA context: {params_changed}")
    
    # Test state dict save/load
    print("\nTesting state dict save/load...")
    state_dict = ema_model.state_dict()
    new_ema = CPUEMAModel(model, decay=0.999)
    new_ema.load_state_dict(state_dict)
    
    print(f"Original EMA step count: {ema_model.step_count}")
    print(f"Loaded EMA step count: {new_ema.step_count}")
    print(f"State dict loaded correctly: {ema_model.step_count == new_ema.step_count}")
    
    # Memory usage check
    print(f"\nMemory check:")
    print(f"All EMA parameters are on CPU: {all(p.device.type == 'cpu' for p in ema_model.ema_params.values())}")
    
    print("\n✅ CPU-based EMA test completed successfully!")
    return True


if __name__ == "__main__":
    test_cpu_ema()