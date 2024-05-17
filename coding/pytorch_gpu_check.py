# filename: pytorch_gpu_check.py
import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("检测到CUDA设备:", torch.cuda.get_device_name(0))
else:
    print("未检测到CUDA设备，请检查PyTorch安装和GPU驱动。")
    exit()