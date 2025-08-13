# D2PLM: A Protein Language Model

This project provides a protein language model with **dual training paradigms**: traditional **Diffusion** and modern **Flow Matching** approaches. Both systems are implemented using Hugging Face's Transformers and Diffusers libraries and use absorbing state mechanisms to generate novel protein sequences that conform to biological rules.

## 🔥 New: Dual Training Paradigms

D2PLM now supports **two completely independent training systems**:

- **🌊 Flow Matching**: Based on Discrete Absorbing Flow Matching (arXiv:2407.15595v2) - A modern, efficient alternative to diffusion
- **🎯 Diffusion**: Traditional discrete diffusion with absorbing states

Both systems are **completely separated** with no shared dependencies, allowing you to choose the approach that best fits your research needs.

## Project Overview

### Core Technologies
- **🤗 Hugging Face Integration**: Built entirely on the Transformers and Diffusers frameworks
- **⚙️ Hydra Configuration**: Flexible and powerful configuration management for both paradigms
- **🚀 SwiGLU Activation**: Optimized activation function with improved initialization
- **🔄 RoPE Positional Embedding**: Rotary Position Embedding for better positional awareness
- **🌊 Flow Matching**: Discrete Absorbing Flow Matching (arXiv:2407.15595v2) implementation
- **🎯 Absorbing Diffusion**: Traditional discrete diffusion with absorbing states
- **🏗️ Transformer Architecture**: DIT (Diffusion Transformer) with optimized design
- **🧬 ESM-2 Tokenizer**: Protein-specific tokenization from ESM-2
- **📊 UniRef50 Dataset**: Standard redundancy-reduced protein sequence dataset
- **⚡ FSDP + Accelerate**: Modern distributed training with PyTorch native parallelism
- **📈 EMA Training**: Exponential Moving Average for enhanced stability and performance
- **🔧 Modular Architecture**: Completely separated systems for maximum flexibility

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
│   │   ├── dit_config.py               # DIT model configuration
│   │   ├── dit_model.py                # DIT model implementation (with RoPE + einops)
│   │   ├── diffusion_scheduler.py      # Diffusion scheduler
│   │   ├── flow_matching_scheduler.py  # 🌊 Flow Matching scheduler
│   │   └── __init__.py
│   ├── trainer/
│   │   ├── DITTrainer.py              # 🎯 Pure Diffusion trainer with EMA
│   │   ├── FMTrainer.py               # 🌊 Pure Flow Matching trainer with EMA
│   │   └── __init__.py
│   ├── dataloader/
│   │   └── DataPipe.py                # Accelerate-compatible data loading
│   └── utils/
│       ├── ActivationFunction.py      # SwiGLU activation function
│       ├── RoPE.py                    # RoPE implementation
│       ├── ModelSave.py               # Model saving utility
│       └── MyLRCallback.py            # Learning rate monitoring callback
├── tools/
│   └── prepare_dataset.py           # Data preprocessing script
├── train_config/
│   ├── train_config.yaml             # 🎯 Diffusion training configuration
│   └── FM_train_config.yaml          # 🌊 Flow Matching training configuration
├── train.py                          # 🎯 Diffusion training script
├── fm_train.py                       # 🌊 Flow Matching training script
├── Claude.md                         # Refactoring documentation
└── README.md                         # This file
```

### 🔄 Training System Architecture

The project now features **two completely independent training systems**:

#### 🎯 Diffusion System
- **Files**: `train.py` + `train_config.yaml` + `DITTrainer.py`
- **Scheduler**: `diffusion_scheduler.py`
- **Approach**: Traditional discrete diffusion with absorbing states

#### 🌊 Flow Matching System
- **Files**: `fm_train.py` + `FM_train_config.yaml` + `FMTrainer.py`  
- **Scheduler**: `flow_matching_scheduler.py`
- **Approach**: Discrete Absorbing Flow Matching (arXiv:2407.15595v2)

## Installation and Setup

### 1. Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- An NVIDIA GPU with at least 8GB of memory is recommended
- Hugging Face Accelerate for distributed training
- Docker (recommended for environment consistency)

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

With your datasets prepared and your configuration pointing to them, you can choose between **two training paradigms**. Both systems are managed by Hydra for flexible configuration.

### 🎯 Diffusion Training

**For Single GPU Training:**
```bash
python train.py
```

**For Multi-GPU Training with FSDP + Accelerate:**

First, configure Accelerate for your hardware setup (one-time setup):
```bash
accelerate config
```

Then launch distributed training:
```bash
accelerate launch train.py
```

**Override configuration on the command line:**
```bash
accelerate launch train.py training.learning_rate=2e-4 training.max_steps=100000
```

### 🌊 Flow Matching Training

**For Single GPU Training:**
```bash
python fm_train.py
```

**For Multi-GPU Training with FSDP + Accelerate:**
```bash
accelerate launch fm_train.py
```

**Override Flow Matching configuration:**
```bash
accelerate launch fm_train.py training.learning_rate=4e-4 flow_matching.num_flow_steps=50
```

**Use custom config file for either system:**
```bash
# For Diffusion
accelerate launch train.py --config_name train_config

# For Flow Matching  
accelerate launch fm_train.py --config_name FM_train_config
```

### ⚡ Quick Comparison

| Feature | 🎯 Diffusion | 🌊 Flow Matching |
|---------|-------------|------------------|
| **Training Script** | `train.py` | `fm_train.py` |
| **Config File** | `train_config.yaml` | `FM_train_config.yaml` |
| **Scheduler** | Traditional discrete diffusion | Discrete absorbing flow matching |
| **Loss Function** | Shifted cross-entropy on corrupted tokens | Direct prediction on corrupted tokens |
| **WandB Project** | `D2PLM_DiT_FSDP` | `D2PLM_FlowMatching_Pure` |
| **Output Directory** | `/workspace/d2plm/weight` | `/workspace/d2plm/weight_flow_matching` |

### 🚀 Enhanced Training Features

Both training systems include several performance enhancements:

- **⚡ FSDP Integration**: Automatic memory optimization with PyTorch's native FullyShardedDataParallel
- **📈 EMA Training**: Exponential Moving Average for enhanced model stability and performance  
- **🔄 Async Data Pipeline**: Overlapped data loading and computation for maximum GPU utilization
- **🔧 Modular Design**: Clean separation between trainer logic and main orchestration
- **🎯 Independent Systems**: Zero shared dependencies between diffusion and flow matching
- **📊 Separate Tracking**: Independent WandB projects and output directories
- **⚙️ Flexible Configuration**: Hydra-based config management with runtime overrides



## Configuration Details

### 🎯 Diffusion Configuration
- **`train_config.yaml`**: Pure diffusion training configuration including model architecture, diffusion scheduler parameters, training hyperparameters, and system settings
- **Features**: Traditional discrete diffusion with absorbing states, shifted cross-entropy loss

### 🌊 Flow Matching Configuration  
- **`FM_train_config.yaml`**: Pure flow matching configuration with specialized flow parameters, independent WandB project settings, and optimized hyperparameters
- **Features**: Discrete absorbing flow matching, direct token prediction loss, cosine flow schedule

### ⚙️ System Configuration
- **Accelerate Config**: Hardware-specific distributed training configuration managed by `accelerate config` command
- **Shared Components**: Both systems use the same model architecture (DIT) and data pipeline, but with completely separate training logic

## 🔄 Recent Updates & Architecture

### v2.0: Dual Training Paradigms (Latest)

**Major Additions:**
- **🌊 Flow Matching System**: Complete implementation of Discrete Absorbing Flow Matching based on arXiv:2407.15595v2
- **🎯 Pure Separation**: Zero shared dependencies between diffusion and flow matching systems  
- **📁 Dual Architecture**: Independent trainers, configs, and scripts for each paradigm
- **📊 Enhanced Tracking**: Separate WandB projects and output directories

**Technical Improvements:**
- **🔧 Modular Design**: 
  - `DITTrainer.py`: Pure diffusion trainer with EMA and custom loss
  - `FMTrainer.py`: Pure flow matching trainer with independent implementation
- **⚡ Performance Optimizations**:
  - Einops integration for cleaner tensor operations
  - Optimized initialization strategy based on DiT paper recommendations
  - Enhanced EMA implementation for both systems
- **🚀 FSDP + Accelerate**: Native PyTorch FSDP replaces Ray + DeepSpeed

### Migration Notes

**From v1.x (Ray + DeepSpeed):**
1. **Dependencies**: Ray removed, Accelerate required for multi-GPU training
2. **Launch Commands**: 
   - Diffusion: `accelerate launch train.py`
   - Flow Matching: `accelerate launch fm_train.py`
3. **Configuration**: Two independent config files with specialized parameters
4. **Architecture**: Complete separation allows choosing training paradigm independently

**System Requirements:**
- PyTorch 2.0+ for native FSDP support
- Hugging Face Accelerate for distributed training
- Diffusers library for EMA and training utilities

## Contributing

Issues and Pull Requests are welcome to improve the project.

## License

This project is licensed under the MIT License.