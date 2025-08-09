# D2PLM: A Protein Language Model

This project provides a protein language model based on the Denoising Diffusion Probabilistic Model (D3PM) framework, implemented entirely using Hugging Face's Transformers and Diffusers libraries. The model uses an Absorbing Diffusion mechanism to generate novel protein sequences that conform to biological rules.

## Project Overview

### Core Technologies
- **Hugging Face Integration**: Built entirely on the Transformers and Diffusers frameworks.
- **Hydra Configuration**: Utilizes Hydra for flexible and powerful configuration management.
- **SwiGLU Activation**: Employs the optimized SwiGLU activation function.
- **RoPE Positional Embedding**: Uses Rotary Position Embedding for better positional awareness.
- **Absorbing Diffusion**: Leverages an absorbing state discrete diffusion model for sequence generation.
- **Transformer Architecture**: A 10-layer Transformer encoder with a 1024-dimensional hidden state.
- **ESM-2 Tokenizer**: Reuses the tokenizer from ESM-2 for protein sequences.
- **UniRef50 Dataset**: Trained on the standard, redundancy-reduced UniRef50 protein sequence dataset.

### Model Specifications
- **Total Parameters**: ~450M (24 layers)
- **Layers**: 20 Transformer encoder layers
- **Hidden Dimension**: 1024
- **Attention Heads**: 16
- **FFN Dimension**: 4096 (4 × 1024)
- **Vocabulary Size**: 33 (20 amino acids + special tokens)
- **Activation Function**: SwiGLU
- **Positional Embedding**: RoPE (Rotary Position Embedding)

## Project Structure

```
D2PLM/
├── model/
│   ├── backbone/
│   │   ├── dit_config.py         # DIT model configuration
│   │   ├── dit_model.py          # DIT model implementation (with RoPE)
│   │   ├── diffusion_scheduler.py # Diffusion scheduler
│   │   └── __init__.py
│   ├── dataloader/
│   │   └── DataPipe.py           # Data loading and tokenization
│   └── utils/
│       ├── ActivationFunction.py # SwiGLU activation function
│       ├── RoPE.py              # RoPE implementation
│       ├── ModelSave.py          # Model saving utility
│       └── MyLRCallback.py       # Learning rate monitoring callback
├── tools/
│   └── prepare_dataset.py      # New script for data preprocessing
├── train_config/
│   ├── train_config.yaml         # Main training configuration
│   └── ZERO2.yaml               # DeepSpeed Zero2 configuration
├── train.py                      # Main training script
└── README.md                     # This file
```

## Installation and Setup

### 1. Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- An NVIDIA GPU with at least 8GB of memory is recommended.
- Docker

### 2. Environment Setup

The recommended and only supported method for setting up the environment is by using the provided Dockerfile (`env.dockerfile`). This file contains all necessary dependencies and configurations, ensuring a consistent and reproducible setup.

Build and run the Docker container to get started.

## Workflow: From Data to Training

The project follows a clear, two-stage workflow: data preparation followed by model training.

### Stage 1: Data Preparation

This is a crucial, one-time step to prepare your dataset for efficient training. We provide a powerful script to handle this.

**What the script does:**
The `tools/prepare_dataset.py` script automates the following:
1.  **Loads Raw Data**: Reads sequences from your large, original LMDB dataset.
2.  **Creates Train/Validation Split**: Randomly splits the data into training and validation sets (defaulting to a 0.5% validation split as in the ESM-2 paper).
3.  **Homology Reduction**: Uses `mmseqs2` to remove sequences from the training set that are highly similar to the validation set. This is crucial for ensuring the validation metrics are reliable and not inflated due to data leakage.
4.  **Generates Final LMDBs**: Saves the processed training and validation sets into two separate, final LMDB datasets, ready for training.

**How to run:**

Before you can start training, you must execute this script. Please ensure `mmseqs2` is installed and available in your environment.

```bash
python tools/prepare_dataset.py --raw_lmdb_path /path/to/your/uniref50.lmdb --output_dir ./prepared_data
```

**Arguments:**
*   `--raw_lmdb_path`: Path to your original, unprocessed UniRef50 LMDB dataset.
*   `--output_dir`: A directory where all processed files, including the final `train_lmdb` and `validation_lmdb` folders, will be saved.

After the script finishes, you will find `train_lmdb` and `validation_lmdb` folders inside your specified output directory.

### Stage 2: Model Training

#### 1. Configure Paths

After preparing your data, you must update the `train_config/train_config.yaml` file to point the training script to the newly created datasets. Modify the `data` section as shown below:

```yaml
# train_config/train_config.yaml

data:
  train_lmdb_path: "./prepared_data/train_lmdb"    # Path to your processed training set
  val_lmdb_path: "./prepared_data/validation_lmdb" # Path to your processed validation set
  train_cache_dir: "./data_cache/train"
  val_cache_dir: "./data_cache/validation"
  max_length: 1024
  batch_size: 8
```

#### 2. Run Training

With your datasets prepared and your configuration pointing to them, you can start the training process. The training is managed by Hydra, allowing for flexible configuration.

**Start a default training run:**
```bash
python train.py
```

**Override configuration on the command line:**
For example, to change the learning rate and number of epochs:
```bash
python train.py training.learning_rate=2e-4 training.num_epochs=5
```

**Multi-GPU Training:**
The project uses Ray for distributed training. The number of workers (GPUs) is set in `train_config.yaml` under the `ray` section. The script will automatically handle the distributed setup.



## Configuration Details

- **`train_config.yaml`**: Contains all major configurations for the model architecture, diffusion process, training loop (learning rate, batch size, optimizer), and system settings.
- **`ZERO2.yaml`**: Configuration for DeepSpeed ZeRO Stage 2 optimization, including FP16 and memory optimizations.

## Contributing

Issues and Pull Requests are welcome to improve the project.

## License

This project is licensed under the MIT License.