# filename: gpu_stress_test_tf.py
import tensorflow as tf
import time
import numpy as np

# Set the number of iterations and the size of the array
num_iterations = 1000
array_size = 1024*1024

# Create a TensorFlow graph
graph = tf.Graph()

# Create a TensorFlow session
with graph.as_default():
    # Create a large tensor
    x = tf.random.normal((array_size,))

    # Create an operation that performs a simple computation on the tensor
    y = tf.multiply(x, 2.0)

    # Create a simple operation to add to the graph
    z = tf.add(y, 1.0)

# Create a TensorFlow session
sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Start the timer
start_time = time.time()

# Run the operation multiple times to stress the GPU
for i in range(num_iterations):
    sess.run(z)

# Stop the timer
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

# Print the results
print(f"GPU stress test complete. Execution time: {execution_time:.2f} seconds")

# Save the results to a file
with open('gpu_stress_test_results.txt', 'w') as f:
    f.write(f"GPU stress test results:\n")
    f.write(f"Execution time: {execution_time:.2f} seconds\n")
    f.write(f"Array size: {array_size}\n")
    f.write(f"Number of iterations: {num_iterations}\n")

print("Results saved to gpu_stress_test_results.txt")