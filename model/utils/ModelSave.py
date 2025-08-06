from transformers import Trainer


# ============================ Model Saving ============================
def save_model(trainer: Trainer, save_dir: str, mode: str = "all"):
    """
    Save the model and tokenizer in safetensor format.

    Args:
        trainer: The trainer instance
        save_dir: Directory path to save the model
        mode: 'esm_encoder' to save only ESM and encoder weights, 'all' to save the full model
    """
    import os

    from safetensors.torch import save_file

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    if mode == "esm_encoder":
        # Save only ESM and encoder weights in safetensors format
        esm_encoder_state_dict = {}
        for name, param in trainer.model.state_dict().items():
            if name.startswith("esm.") or name.startswith("encoder."):
                esm_encoder_state_dict[name] = param.detach().cpu()
        save_file(esm_encoder_state_dict, f"{save_dir}/esm_encoder.safetensors")
        print(
            f"\nOnly ESM and encoder weights saved in safetensors format at: {save_dir}/esm_encoder.safetensors"
        )
    else:
        # Save full model in safetensors format
        trainer.model.save_pretrained(
            save_dir,
            safe_serialization=True,  # This ensures safetensor format
        )
        print(f"\nFull model saved in safetensors format at: {save_dir}")

    # Save tokenizer and config in both modes
    trainer.tokenizer.save_pretrained(save_dir)
    trainer.model.config.save_pretrained(save_dir)
