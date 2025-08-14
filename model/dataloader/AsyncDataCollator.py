# model/dataloader/AsyncDataCollator.py
# Asynchronous Data Collator with Dual CUDA Stream Pipeline

import torch
import nvtx
from typing import Dict, Any, List, Union
from transformers import PreTrainedTokenizer


class AsyncDataCollator:
    """
    Asynchronous data collator that implements dual CUDA stream pipeline
    for overlapping data transfer and computation.
    
    Features:
    1. Dedicated copy stream for async data transfer
    2. Non-blocking GPU memory transfers
    3. Automatic tensor device management
    4. Compatible with existing data collators
    """
    
    def __init__(self, base_collator, device: Union[str, torch.device], stream_priority: int = 0):
        """
        Args:
            base_collator: The original data collator (DITDataCollator or FMDataCollator)
            device: Target CUDA device
            stream_priority: CUDA stream priority (higher = more priority)
        """
        self.base_collator = base_collator
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Create dedicated copy stream for async data transfer
        self.copy_stream = torch.cuda.Stream(priority=stream_priority)
        
        # Keep track of current batch for pipeline
        self.current_batch = None
        self.next_batch_ready = False
        
    @nvtx.annotate("AsyncDataCollator.__call__", color="cyan")
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process batch with asynchronous data transfer.
        
        Pipeline:
        1. Base collator processes features on CPU
        2. Async transfer to GPU using dedicated stream
        3. Synchronize before returning
        """
        # Step 1: CPU processing using base collator
        with nvtx.annotate("base_collator_cpu", color="green"):
            cpu_batch = self.base_collator(features)
        
        # Step 2: Async GPU transfer
        with nvtx.annotate("async_gpu_transfer", color="orange"):
            gpu_batch = self._async_transfer_to_device(cpu_batch)
        
        return gpu_batch
    
    def _async_transfer_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Transfer batch to GPU asynchronously using dedicated stream."""
        gpu_batch = {}
        
        with torch.cuda.stream(self.copy_stream):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    # Ensure source tensor is pinned for efficient transfer
                    if not value.is_pinned():
                        value = value.pin_memory()
                    
                    # Async transfer to GPU
                    gpu_batch[key] = value.to(self.device, non_blocking=True)
                else:
                    # Non-tensor values (scalars, etc.)
                    gpu_batch[key] = value
        
        # Synchronize copy stream to ensure transfer completion
        self.copy_stream.synchronize()
        return gpu_batch
    
    def prefetch_next_batch(self, features: List[Dict[str, Any]]) -> None:
        """
        Prefetch next batch asynchronously without blocking current computation.
        This enables true pipeline parallelism.
        """
        with torch.cuda.stream(self.copy_stream):
            cpu_batch = self.base_collator(features)
            self.current_batch = self._async_transfer_to_device(cpu_batch)
            self.next_batch_ready = True
    
    def get_prefetched_batch(self) -> Dict[str, torch.Tensor]:
        """Get previously prefetched batch."""
        if not self.next_batch_ready:
            raise RuntimeError("No prefetched batch available")
        
        # Ensure copy stream operations are complete
        self.copy_stream.synchronize()
        
        batch = self.current_batch
        self.current_batch = None
        self.next_batch_ready = False
        return batch
    
    def __getattr__(self, name):
        """Delegate missing attributes to base collator."""
        return getattr(self.base_collator, name)


class PipelinedDataLoader:
    """
    Dataloader wrapper that implements dual-stream pipeline for maximum throughput.
    
    Architecture:
    - Stream 0 (default): GPU computation
    - Stream 1 (copy): Data transfer H2D
    
    Pipeline stages:
    1. While GPU computes batch N, CPU prepares batch N+1
    2. While GPU computes batch N, copy stream transfers batch N+1 to GPU
    3. GPU computation and data transfer overlap for maximum efficiency
    """
    
    def __init__(self, dataloader, async_collator: AsyncDataCollator):
        self.dataloader = dataloader
        self.async_collator = async_collator
        self.compute_stream = torch.cuda.default_stream()
        
    def __iter__(self):
        data_iter = iter(self.dataloader)
        
        try:
            # Preload first batch
            first_features = next(data_iter)
            current_batch = self.async_collator(first_features)
            
            # Pipeline loop
            for next_features in data_iter:
                # Start async prefetch of next batch while current batch is being processed
                with torch.cuda.stream(self.async_collator.copy_stream):
                    self.async_collator.prefetch_next_batch(next_features)
                
                # Yield current batch for computation
                yield current_batch
                
                # Get prefetched batch (this will sync copy stream)
                current_batch = self.async_collator.get_prefetched_batch()
            
            # Yield last batch
            yield current_batch
            
        except StopIteration:
            return
    
    def __len__(self):
        return len(self.dataloader)