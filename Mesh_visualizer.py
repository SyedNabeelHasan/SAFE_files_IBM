import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === Load saved geometry ===
path1 = r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz"
path2 = r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz"

RAX = np.load(path1, allow_pickle=True)
GAX = np.load(path2, allow_pickle=True)

filtered_interior_x = RAX["array1"]
sorted_ghost_nodes = GAX["array1"]
sorted_first_interface = GAX["array2"]
unique_rounded_points = GAX["array5"]
rounded_horizontal = GAX["array6"]
interpolation_points = GAX.get("array3", None)
mirror_array = GAX.get("array4", None)

print("✅ Geometry data loaded successfully!")
print(f"Ghost sets available: {len(sorted_ghost_nodes)}")

# === Grid spacing control ===
Δh = 2  # 👈 user-defined uniform grid spacing

# === Initial setup ===
set_idx = 0
point_idx = 0

# Extract data for plotting
fx, fy = zip(*filtered_interior_x)
efx, efy = zip(*unique_rounded_points)
hx,hy = zip(*rounded_horizontal)

fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.3, right=0.8)

# === Plot range ===
x_min, x_max = 0, 10
y_min, y_max = 0, 10

ax.set_xlim(x_min - Δh, x_max + Δh)
ax.set_ylim(y_min - Δh, y_max + Δh)
ax.set_aspect("equal", adjustable="datalim")

# === Gridlines ===
ax.xaxis.set_major_locator(plt.MultipleLocator(Δh))
ax.yaxis.set_major_locator(plt.MultipleLocator(Δh))
ax.grid(True, which="major", color="#cccccc", linewidth=0.6)

# Keep gridlines consistent during zoom/pan
def update_grid(event):
    ax.xaxis.set_major_locator(plt.MultipleLocator(Δh))
    ax.yaxis.set_major_locator(plt.MultipleLocator(Δh))
    ax.grid(True, which="major", color="#cccccc", linewidth=0.6)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("draw_event", update_grid)

# === Base background ===
ax.scatter(fx, fy, color="gray", s=6, alpha=0.4, label="Interior")
ax.scatter(efx, efy, color="red", s=1, label="Actual Edge")
ax.scatter(hx, hy, color="red", s=1)

# === Initialize scatters ===
ghost_set_sc = ax.scatter([], [], color="green", s=8, alpha=0.6, label="Ghost Set")
ghost_sc = ax.scatter([], [], color="blue", s=10, label="Selected Ghost Point")
mirror_sc = ax.scatter([], [], color="black", s=10, label="Mirror Point")
interp_sc = ax.scatter([], [], color="#d9027f", s=11, label="Interpolation Points")

# === Legend moved outside ===
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    frameon=True,
    fontsize=8,
    title="Legend",
    title_fontsize=9
)

ax.set_title(f"Set {set_idx}, Point {point_idx}")

# === Slider setup ===
ax_slider_set = plt.axes([0.2, 0.15, 0.55, 0.03])
slider_set = Slider(ax_slider_set, "Set Index", 0, len(sorted_ghost_nodes) - 1, valinit=set_idx, valstep=1)

ax_slider_point = plt.axes([0.2, 0.08, 0.55, 0.03])
slider_point = Slider(ax_slider_point, "Point Index", 0, len(sorted_ghost_nodes[set_idx]) - 1, valinit=point_idx, valstep=1)

# === Update function ===
def update(val):
    set_i = int(slider_set.val)
    point_i = int(slider_point.val)

    # --- Update all ghost points (green) ---
    if set_i < len(sorted_ghost_nodes):
        gx_all, gy_all = zip(*sorted_ghost_nodes[set_i])
        ghost_set_sc.set_offsets(np.column_stack((gx_all, gy_all)))
    else:
        ghost_set_sc.set_offsets([])

    # --- Selected ghost (blue) ---
    if set_i < len(sorted_ghost_nodes) and point_i < len(sorted_ghost_nodes[set_i]):
        gx, gy = sorted_ghost_nodes[set_i][point_i]
        ghost_sc.set_offsets([[gx, gy]])
    else:
        ghost_sc.set_offsets([])

    # --- Mirror (black) ---
    if mirror_array is not None and set_i < len(mirror_array) and point_i < len(mirror_array[set_i]):
        mx, my = mirror_array[set_i][point_i]
        mirror_sc.set_offsets([[mx, my]])
    else:
        mirror_sc.set_offsets([])

    # --- Interpolation points (pink) ---
    if interpolation_points is not None and set_i < len(interpolation_points) and point_i < len(interpolation_points[set_i]):
        interp_group = interpolation_points[set_i][point_i]
        if len(interp_group) > 0:
            i_x, i_y = zip(*interp_group)
            interp_sc.set_offsets(np.column_stack((i_x, i_y)))
        else:
            interp_sc.set_offsets([])
    else:
        interp_sc.set_offsets([])

    ax.set_title(f"Set {set_i}, Point {point_i}")
    fig.canvas.draw_idle()

# === Adjust point-slider range dynamically ===
def update_point_slider(val):
    set_i = int(slider_set.val)
    if set_i < len(sorted_ghost_nodes):
        slider_point.valmax = len(sorted_ghost_nodes[set_i]) - 1
        slider_point.ax.set_xlim(slider_point.valmin, slider_point.valmax)
    update(val)

slider_set.on_changed(update_point_slider)
slider_point.on_changed(update)

plt.show()
