import numpy as np
import matplotlib.pyplot as plt

# =====================================
# USER: Put file path here manually
# =====================================

file_path_u = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_u/Time_stack_u_t000045.npz"
file_path_v = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_v/Time_stack_v_t000045.npz"
# Example:
# file_path = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_p/Time_stack_p_t000200.npz"

# =====================================
# Load NPZ file
# =====================================

data_u = np.load(file_path_u)
data_v = np.load(file_path_v)

print("Available arrays:", data_u.files)
print("Available arrays:", data_v.files)

key_u = data_u.files[0]       # only one array expected
key_v = data_v.files[0]       # only one array expected
U = data_u[key_u]
V = data_v[key_v]

# print("Loaded:", key)
# print("Raw shape:", F.shape)
# print("Dtype:", F.dtype)

# =====================================
# Handle stacked or 2D data automatically
# =====================================

if U.ndim == 3:
    print("Detected stacked data → using first snapshot.")
    F = U[0]              # shape becomes (ny, nx)

elif U.ndim != 2:
    raise ValueError(f"Unsupported array shape: {U.shape}")

ny, nx = U.shape
print("Final field shape:", U.shape)

# =====================================
# Grid (edit physical domain if needed)
# =====================================

# Grid index coordinates
x = np.arange(nx)
y = np.arange(ny)

# If you want physical coordinates instead, uncomment and edit:
# xmin, xmax = 0.0, 36.0
# ymin, ymax = 0.0, 21.0
# x = np.linspace(xmin, xmax, nx)
# y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y)

# =====================================
# Plot
# =====================================

fig, ax = plt.subplots(figsize=(8, 6))

# -------- Filled contours --------
cf = ax.contourf(X, Y, U, 50, cmap="coolwarm")
plt.colorbar(cf, ax=ax, label=key_u)
ax.streamplot(X, Y, U, V, color= 'k', density = 3, linewidth=1)

# -------- Isobars (contour lines) --------
# cs = ax.contour(X, Y, F, 25, colors="black", linewidths=0.6)
# ax.clabel(cs, fontsize=8)

# -------- Formatting --------

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect("equal")
ax.grid(False)

plt.tight_layout()
plt.show()
