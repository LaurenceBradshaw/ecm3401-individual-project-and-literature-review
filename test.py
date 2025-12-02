import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

a_vals = np.linspace(0.1, 90, 300)
b_vals = np.linspace(0, 60, 300)

a_grid, b_grid = np.meshgrid(a_vals, b_vals)

w = 1.0 / ((1.0 / np.sin(np.radians(a_grid)))**2 + (4.0 * 10.0**(-b_grid / 30.0))**2)
# w = np.sin(np.radians(a_grid))**2 * 10**(b_grid / 10)
# w = 1 / (1/(np.sin(np.radians(a_grid))**2)+(1/(10**(b_grid/10))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(a_grid, b_grid, w)

ax.set_xlabel("a (degrees)")
ax.set_ylabel("b (dB-Hz)")
ax.set_zlabel("w")

plt.show()
