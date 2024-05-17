# filename: numpy_usage_example.py
import numpy as np

# Create a 1D numpy array
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(array_1d)

# Create a 2D numpy array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:")
print(array_2d)

# Perform some operations
print("\nArray Operations:")
print("Sum of elements in array_1d:", np.sum(array_1d))
print("Mean of elements in array_2d:", np.mean(array_2d))

# Reshape the array
reshaped_array = array_1d.reshape(5, 1)
print("\nReshaped Array:")
print(reshaped_array)