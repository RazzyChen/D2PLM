import wandb
from transformers import TrainerCallback

class LogLearningRateCallback(TrainerCallback):
    """
    Custom callback to log learning rate to wandb.
    """

    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer and len(optimizer.param_groups) > 0:
            lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
            # Check if wandb has been initialized
            if wandb.run:
                wandb.log({"learning_rate": lr})  # Log learning rate to wandb