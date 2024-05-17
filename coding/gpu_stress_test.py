# filename: gpu_stress_test.py
import tensorflow as tf
import time

# 设置测试参数
matrix_size = 10000  # 矩阵大小
iterations = 100     # 迭代次数

# 创建随机矩阵
matrix1 = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)
matrix2 = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)

# 使用GPU进行矩阵乘法
with tf.device('/GPU:0'):
  start_time = time.time()
  for _ in range(iterations):
    result = tf.matmul(matrix1, matrix2)
  end_time = time.time()

# 计算平均运算时间
average_time = (end_time - start_time) / iterations

# 将结果保存到文件
with open("gpu_stress_test_results.txt", "w") as f:
  f.write(f"Matrix size: {matrix_size}x{matrix_size}\n")
  f.write(f"Iterations: {iterations}\n")
  f.write(f"Average time per iteration: {average_time:.6f} seconds\n")

print(f"测试结果已保存到 gpu_stress_test_results.txt 文件中。")