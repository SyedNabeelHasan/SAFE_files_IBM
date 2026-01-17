import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. Load VTI file
# ============================================

file_path = r"D:/numerical computation/Results/Lid driven cavity/Time_stack_p 101x101/time_data_t9999.vti"        # <<< CHANGE THIS PATH
grid = pv.read(file_path)

print("\nGrid info:")
print(grid)
print("\nAvailable data arrays:")
print(grid.array_names)

# ============================================
# 2. Extract pressure array
# ============================================

# Change this name if needed after printing array_names
pressure_name = grid.array_names[0]   # or explicitly: "pressure"
p_raw = grid[pressure_name]

# ============================================
# 3. Grid dimensions and reshape
# ============================================

nx, ny, nz = grid.dimensions        # nz should be 1 for 2D
print("\nGrid dimensions:", nx, ny, nz)

# VTK stores data in Fortran order
P = p_raw.reshape((ny, nx), order="F")

# ============================================
# 4. Create coordinate mesh
# ============================================

xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y)

# ============================================
# 5. Plot pressure contours
# ============================================

fig, ax = plt.subplots(figsize=(8, 6))

# Filled contours (smooth field)
cf = ax.contourf(X, Y, P, 200, cmap="viridis")

# Contour lines (isobars)
cs = ax.contour(X, Y, P, 200, colors="black", linewidths=0.6)

# Optional labels on contour lines
ax.clabel(cs, fontsize=8, inline=True)

# Colorbar
plt.colorbar(cf, ax=ax, label="Pressure")

# Labels and title
ax.set_title("Pressure Contours from VTI File")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.show()
