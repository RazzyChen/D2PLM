# D2PLM 蛋白质语言模型

这是一个基于吸收扩散概率模型 (D3PM) 的蛋白质语言模型项目，完全基于Hugging Face Transformers和Diffusers框架实现。该模型采用吸收扩散 (Absorbing Diffusion) 机制，能够从头生成符合生物学规则的新蛋白质序列。

## 项目概述

### 核心技术
- **Hugging Face集成**: 完全基于Transformers和Diffusers框架
- **Hydra配置管理**: 使用Hydra进行灵活的配置管理
- **SwiGLU激活函数**: 优化的SwiGLU激活函数
- **RoPE位置编码**: 旋转位置编码 (Rotary Position Embedding)
- **吸收扩散**: 使用吸收态离散扩散概率模型进行序列生成
- **Transformer架构**: 10层Transformer编码器，1024维隐藏状态
- **ESM2 Tokenizer**: 复用ESM2的蛋白质序列分词器
- **UniRef50数据集**: 使用经过聚类去冗余的标准蛋白质序列数据集

### 模型规格
- **总参数量**: ~390M (10层)
- **层数**: 20层Transformer编码器
- **隐藏维度**: 1024
- **注意力头数**: 16
- **FFN维度**: 4096 (4 × 1024)
- **词汇表大小**: ~25 (20种氨基酸 + 特殊标记)
- **激活函数**: SwiGLU
- **位置编码**: RoPE (Rotary Position Embedding)

## 项目结构

```
D3PLM/
├── model/
│   ├── backbone/
│   │   ├── dit_config.py         # DIT模型配置
│   │   ├── dit_model.py          # DIT模型实现 (集成RoPE)
│   │   ├── diffusion_scheduler.py # 扩散调度器
│   │   └── __init__.py
│   ├── dataloader/
│   │   └── DataPipe.py           # 数据加载和预处理
│   └── utils/
│       ├── ActivationFunction.py # SwiGLU激活函数
│       ├── RoPE.py              # RoPE位置编码实现
│       ├── ModelSave.py          # 模型保存工具
│       └── MyLRCallback.py       # 学习率监控回调
├── train_config/
│   ├── train_config.yaml         # 主要训练配置
│   └── ZERO2.yaml               # DeepSpeed Zero2配置
├── train_dit_hydra.py            # Hydra训练脚本
├── generate_hydra.py             # Hydra生成脚本
├── test_integration.py           # 集成测试
├── test_rope_simple.py          # RoPE测试脚本
├── requirements.txt              # 项目依赖
└── README.md                     # 项目说明
```

## 安装和设置

### 1. 环境要求
- Python 3.8+
- CUDA 11.0+ (用于GPU训练)
- 至少8GB GPU内存 (RTX 2080 Super优化)

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 数据准备
将UniRef50数据集放置在 `./UniRef50/` 目录下，支持LMDB格式。

## 使用方法

### 训练模型

使用Hydra配置进行训练：
```bash
python train_dit_hydra.py
```

#### 自定义配置训练
```bash
python train_dit_hydra.py training.learning_rate=2e-4 training.num_train_epochs=5
```

#### 多GPU训练
```bash
python train_dit_hydra.py training.per_device_train_batch_size=4 system.device=cuda
```

### 生成蛋白质序列

```bash
python generate_hydra.py \
    --model_path ./checkpoints/final_model \
    --tokenizer_path ./checkpoints/final_model \
    --output_file generated_sequences.fasta
```

#### 使用引导序列生成
```bash
python generate_hydra.py \
    --model_path ./checkpoints/final_model \
    --tokenizer_path ./checkpoints/final_model \
    --guidance_sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" \
    --output_file guided_sequences.fasta
```

## 配置说明

### 主要配置 (train_config.yaml)
- **模型配置**: 包含所有模型架构参数
- **扩散配置**: 扩散过程的超参数
- **训练配置**: 学习率、批次大小、优化器等
- **系统配置**: 设备、精度、编译等设置

### DeepSpeed配置 (ZERO2.yaml)
- **Zero2优化**: Stage 2 ZeRO优化
- **FP16支持**: 针对2080 Super的FP16优化
- **内存优化**: 减少显存使用

## 性能优化

### 针对RTX 2080 Super的优化
- **FP16混合精度**: 充分利用Tensor Core
- **批次大小优化**: batch_size=8, gradient_accumulation_steps=4
- **RoPE位置编码**: 更好的位置表示能力
- **SwiGLU激活**: 更高效的激活函数

### 训练建议
- 使用FP16训练以获得最佳性能
- 根据显存情况调整batch_size
- 启用gradient_accumulation增加有效batch size

## 测试

### 运行RoPE测试
```bash
python test_rope_simple.py
```

### 运行集成测试
```bash
python test_integration.py
```

## 实验结果

### 训练性能 (RTX 2080 Super)
- **训练速度**: ~1000 sequences/second
- **显存使用**: ~6-8GB
- **收敛时间**: 预计10-15小时 (10 epochs)

### 生成质量
- **序列长度**: 50-500氨基酸
- **生物学合理性**: 通过结构预测验证
- **多样性**: 高序列多样性

## 技术细节

### RoPE (Rotary Position Embedding)
- **实现**: 基于einops的高效实现
- **优势**: 更好的相对位置编码
- **应用**: 在注意力层中应用旋转位置编码

### 扩散过程
- **前向过程**: 逐步将序列腐蚀为MASK
- **反向过程**: 学习从噪声恢复原始序列
- **调度器**: 线性β调度

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。 
