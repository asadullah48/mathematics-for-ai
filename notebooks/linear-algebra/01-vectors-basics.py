"""
Introduction to Vectors for AI
==============================

This notebook covers the basics of vectors and their applications in AI.
"""

import numpy as np
import matplotlib.pyplot as plt

# Create a simple vector
vector = np.array([1, 2, 3])
print(f"Vector: {vector}")
print(f"Vector shape: {vector.shape}")
print(f"Vector magnitude: {np.linalg.norm(vector)}")

# Vector operations
v1 = np.array([1, 2])
v2 = np.array([3, 4])

print(f"Vector addition: {v1 + v2}")
print(f"Dot product: {np.dot(v1, v2)}")

# Visualization
plt.figure(figsize=(8, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
plt.quiver(0, 0, v1[0]+v2[0], v1[1]+v2[1], angles='xy', scale_units='xy', scale=1, color='g', label='v1+v2')
plt.xlim(-1, 5)
plt.ylim(-1, 6)
plt.grid(True)
plt.legend()
plt.title('Vector Addition')
plt.show()
