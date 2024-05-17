# filename: pytorch_stress_test.py
import torch
import time

# 设置输入和卷积核大小
input_size = 25000
kernel_size = 3

# 创建随机输入和卷积核，使用 float16 数据类型
input_tensor = torch.randn(1, 1, input_size, input_size, dtype=torch.float16)
conv_kernel = torch.randn(1, 1, kernel_size, kernel_size, dtype=torch.float16)

# 创建卷积层
conv_layer = torch.nn.Conv2d(1, 1, kernel_size)

# ---- CPU 卷积运算 ----
device = torch.device("cpu")
conv_layer = conv_layer.to(device)

# 将卷积层偏置项和权重设置为 float16 数据类型
conv_layer.bias.data = conv_layer.bias.data.half()
conv_layer.weight.data = conv_layer.weight.data.half()

input_tensor_cpu = input_tensor.to(device)

print(f"输入张量形状：{input_tensor_cpu.shape}")

# 记录开始时间 (CPU)
start_time_cpu = time.time()

# 执行卷积运算 (CPU)
output_tensor_cpu = conv_layer(input_tensor_cpu)

# 记录结束时间 (CPU)
end_time_cpu = time.time()

print(f"输出张量形状 (CPU)：{output_tensor_cpu.shape}")

# ---- GPU 卷积运算 ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conv_layer = conv_layer.to(device)

# 将卷积层偏置项和权重设置为 float16 数据类型
conv_layer.bias.data = conv_layer.bias.data.half()
conv_layer.weight.data = conv_layer.weight.data.half()

input_tensor_gpu = input_tensor.to(device)

print(f"输入张量形状：{input_tensor_gpu.shape}")

# 记录开始时间 (GPU)
start_time_gpu = time.time()

# 执行卷积运算 (GPU)
output_tensor_gpu = conv_layer(input_tensor_gpu)

# 记录结束时间 (GPU)
end_time_gpu = time.time()

print(f"输出张量形状 (GPU)：{output_tensor_gpu.shape}")

# ---- 计算运算时间 ----
elapsed_time_cpu = end_time_cpu - start_time_cpu
elapsed_time_gpu = end_time_gpu - start_time_gpu

print(f"卷积运算时间 (CPU): {elapsed_time_cpu:.2f} 秒")
print(f"卷积运算时间 (GPU): {elapsed_time_gpu:.2f} 秒")

# ---- 保存结果 ----
with open("gpu_stress_test_result.txt", "w") as f:
    f.write(f"卷积运算时间 (CPU): {elapsed_time_cpu:.2f} 秒\n")
    f.write(f"卷积运算时间 (GPU): {elapsed_time_gpu:.2f} 秒")

print("测试结果已保存到 gpu_stress_test_result.txt 文件中")