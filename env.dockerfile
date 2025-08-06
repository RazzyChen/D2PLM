FROM nvcr.io/nvidia/pytorch:25.06-py3

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

# 设置工作目录
WORKDIR /workspace/

CMD ["/bin/bash"]
