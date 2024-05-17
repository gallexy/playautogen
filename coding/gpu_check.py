# filename: gpu_check.py
import tensorflow as tf

# 获取GPU设备列表
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("检测到GPU设备:", gpus)
else:
    print("未检测到GPU设备，请检查TensorFlow安装和GPU驱动。")
    exit()