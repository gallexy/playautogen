# filename: random_plot.py
import matplotlib.pyplot as plt
import numpy as np

# Generate 100 random numbers between 0 and 1
data = np.random.rand(100)

# Create a scatter plot
plt.scatter(range(len(data)), data)
plt.xlabel("Index")
plt.ylabel("Random Value")
plt.title("Scatter Plot of Random Numbers")

# Save the plot as a PNG image
plt.savefig("random_plot.png")
print("Plot saved as 'random_plot.png'")