FROM nvcr.io/nvidia/pytorch:25.06-py3

# Install Python dependencies
RUN pip install --no-cache --break-system-packages -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    accelerate \
    biopython \
    datasets \
    deepspeed \
    hydra-core \
    lmdb \
    numpy \
    nvitop \
    pandas \
    peft \
    diffusers \
    psutil \
    scipy \
    einops \
    ray \
    torchmetrics \
    transformers \
    wandb 

# Install MMseqs2
RUN cd /opt && \
    wget -q https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz && \
    tar xzf mmseqs-linux-gpu.tar.gz && \
    rm mmseqs-linux-gpu.tar.gz && \
    chmod +x mmseqs/bin/mmseqs

# Set environment variables
ENV PATH="/opt/mmseqs/bin:${PATH}"
ENV HF_ENDPOINT=https://hf-mirror.com

# Set working directory
WORKDIR /workspace/

CMD ["/bin/bash"]
