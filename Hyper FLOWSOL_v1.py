import numpy as np
import matplotlib.pyplot as plt
import time
from colorama import Fore, Style, init
from matplotlib.animation import FuncAnimation
import psutil
from scipy.sparse.linalg import cg
from tqdm import tqdm
import torch
import scipy
import os
import cupyx.scipy.sparse as gpu_sp
import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg
import scipy.sparse as sp
import cupy as cp
# import cupyx.scipy.sparse as sp
import gc
import cupyx.scipy.sparse as cusparse


output_dir = r"D:/numerical computation/Results/FAH4001"

os.makedirs(output_dir, exist_ok=True)
#=======================================================================================================================================#
#                                                                VERSION CHECK BLOCK
#=======================================================================================================================================#
print("="*60)
print("🔎 Environment & Library Versions")
print("="*60)

import sys

print("Python version        :", sys.version.replace("\n", " "))

print("NumPy version         :", np.__version__)
print("Matplotlib version    :", plt.matplotlib.__version__)
print("SciPy version         :", scipy.__version__)
print("CuPy version          :", cp.__version__)
print("PyTorch version       :", torch.__version__)
print("psutil version        :", psutil.__version__)

# CUDA info (PyTorch)
print("\n 🖥️ CUDA / GPU Info (PyTorch)")
print("CUDA available        :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version          :", torch.version.cuda)
    print("GPU name              :", torch.cuda.get_device_name(0))
    print("GPU capability        :", torch.cuda.get_device_capability(0))

# CUDA info (CuPy)
print("\n 🖥️ CUDA / GPU Info (CuPy)")
try:
    print("CuPy CUDA runtime     :", cp.cuda.runtime.runtimeGetVersion())
    print("CuPy GPU name         :", cp.cuda.Device(0).name)
except Exception as e:
    print("CuPy CUDA info error  :", e)

print("="*60)

# Total and available memory (in GB)
total = psutil.virtual_memory().total / (1024**3)
available = psutil.virtual_memory().available / (1024**3)
print(Fore.GREEN + f"total space {total} GB" + Style.RESET_ALL)
print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                                   IMPORTING DATA
#=======================================================================================================================================#

mesh_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz")
gax_file = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# sorted_first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# second_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
inside_pt = mesh_data["array1"]
ghost_nodes = gax_file["array1"]
sorted_first_interface = gax_file["array7"]
sfi = gax_file["array8"]
ffi = gax_file["array9"]
ghost_nodes_list = ghost_nodes_list = [list(map(tuple, block)) for block in ghost_nodes]
first_interface = set(tuple(point) for point in ffi)
first_interface = np.array(
    sorted(first_interface),
    dtype=float)

second_interface = np.array(list(sfi.item()), dtype=float)
del_h = float(mesh_data["del_h"]) 

# print(">/\<",sfi[0])
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                                   REBUILDING DOMAIN
#=======================================================================================================================================#

conversion_factor = 1/del_h     # mesh size
# conversion_factor = 
print("grid size ",inside_pt[0][0],inside_pt[0][1])
print(conversion_factor)


def cord_transfer_logic(a):
    x_coord=inside_pt[a][0]
    y_coord=inside_pt[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

def cord_transfer_logic_l1(a):
    x_coord=first_interface[a][0]
    y_coord=first_interface[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

def cord_transfer_logic_l2(a):
    x_coord=second_interface[a][0]
    y_coord=second_interface[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

cn = nx = int(mesh_data["nx"])  #201
rn = ny = int(mesh_data["ny"]) 
print(del_h,nx,ny)


# numeric 2D mesh

u_mat = np.full((rn, cn), np.nan)   # u_velocity

v_mat = np.full((rn, cn), np.nan)   # v_velocity

p_mat = np.full((rn, cn), np.nan)   # pressure


variable_array = []                                 # variable marker mesh (in use for pressure)
for i in range (0,len(inside_pt),1):
    r,c = cord_transfer_logic(i)
    x = f'x{r}|{c}'       # beware of row and column to x-y coordinate system
    variable_array.append(x)
# print("🐒: ",len(variable_array),variable_array)


variable_array_copy = variable_array.copy()         # all the pressure BC editing is done on it. Make ghost_node value == first_interface

# Appending all the initial conditions to respective mesh nodes u, v and p (for fluid nodes)
for i in range(0,len(inside_pt),1):
    r,c = cord_transfer_logic(i)
    u_mat[r][c] = 0                 # uniform initial velocity (u) condition through out the geometry
    v_mat[r][c] = 0                 # uniform initial velocity (v) condition through out the geometry
    p_mat[r][c] = 0                 # uniform initial pressure condition
    


drich_u = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             # x direction velocity drichilit boundary condition
drich_v = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]            # y direction velocity drichilit boundary condition
drich_p = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]            # pressure drichilit boundary condition


for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        u_mat[r][c] = drich_u[i]
        v_mat[r][c] = drich_v[i]
        p_mat[r][c] = drich_p[i]
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                       SOME EXAMPLES OF BOUNDARY CONDITIONS
#=======================================================================================================================================#
# BCs for some example problems
# ======================================Backward Facing Step=================================================
drich_u = [0,0,0,"NDN",0,1]                                 # u velocity drichilit boundary condition
drich_v = [0,0,0,"NDN",0,0]                                 # v velocity drichilit boundary condition
drich_p = ["NDN","NDN","NDN",0,"NDN","NDN"]                 # p pressure drichilit boundary condition

   
Ne_BC_u = ["NCN","NCN","NCN",0,"NCN","NCN"]                 # u velocity drichilit boundary condition
Ne_BC_v = ["NCN","NCN","NCN",0,"NCN","NCN"]                 # v velocity drichilit boundary condition
Ne_BC_p = [0,0,0,"NCN",0,0]                                 # p velocity drichilit boundary condition
# # ==============================================Channel Flow================================================
# drich_u = [0,"NDN",0,1]                         # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                             # v velocity drichilit boundary condition
# drich_p = ["NDN",0,"NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN",0,"NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]             # v velocity drichilit boundary condition
# Ne_BC_p = [0,"NCN",0,0]                         # p velocity drichilit boundary condition
# # #==========================================Lid driven Cavity=======================================
# drich_u = [0,0,1,0]                                 # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                                 # v velocity drichilit boundary condition
# drich_p = ["NDN","NDN","NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN","NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,0,0,0]                                 # p velocity drichilit boundary condition
#=========================================================================================================================================#
#                                                        SETTING UP BOUNDARY CONDITIONS
#=========================================================================================================================================#
# drich_u = ["NDN","NDN","NDN",1,0,0,0,0]                             # u velocity drichilit boundary condition
# drich_v = [0,"NDN",0,0,0,0,0,0]                                # v velocity drichilit boundary condition
# drich_p = ["NDN",0,"NDN","NDN","NDN","NDN","NDN","NDN"]                # p pressure drichilit boundary condition

   
# Ne_BC_u = [0,0,0,"NCN","NCN","NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN",0,"NCN","NCN","NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,"NCN",0,0,0,0,0,0]                                 # p velocity drichilit boundary condition

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        # print(r,c,"mmmm")
        if (drich_u[i] != "NDN"):
            u_mat[r][c] = drich_u[i]
        if (drich_v[i] != "NDN"):
            v_mat[r][c] = drich_v[i]
        if (drich_p[i] != "NDN"):    
            p_mat[r][c] = drich_p[i]


#=======================================================================================================================================#
#                                                               END
#=======================================================================================================================================#
#=============Linearizing data structure for Neumenn BCs============
# first interface
rf_list = []
cf_list = []
# ghost nodes
rgn_list = []
cgn_list = []
# Neumann BC list
Ne_BC_u_list = []
Ne_BC_v_list = []
Ne_BC_p_list = []

for i in range(len(sorted_first_interface)):
    for j in range(len(sorted_first_interface[i])):
        xf, yf = sorted_first_interface[i][j]
        rf_list.append(int(round(yf * conversion_factor))) 
        cf_list.append(int(round(xf * conversion_factor)))

        if (Ne_BC_u[i] != "NCN"):
            Ne_BC_u_list.append(Ne_BC_u[i])
        else:
            Ne_BC_u_list.append(np.nan)

        if (Ne_BC_v[i] != "NCN"):
            Ne_BC_v_list.append(Ne_BC_v[i])
        else:
            Ne_BC_v_list.append(np.nan)

        if (Ne_BC_p[i] != "NCN"):
            Ne_BC_p_list.append(Ne_BC_p[i])
        else:
            Ne_BC_p_list.append(np.nan)

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        xgn, ygn = ghost_nodes[i][j]
        rgn_list.append(int(round(ygn * conversion_factor))) 
        cgn_list.append(int(round(xgn * conversion_factor)))


# moving linearized list to GPU (common to Drichilet and Neumann)
rf = cp.asarray(rf_list)
cf = cp.asarray(cf_list)

rgn = cp.asarray(rgn_list)
cgn = cp.asarray(cgn_list)
# moving to GPU memory
Ne_BC_u_vector = cp.asarray(Ne_BC_u_list)
Ne_BC_v_vector = cp.asarray(Ne_BC_v_list)
Ne_BC_p_vector = cp.asarray(Ne_BC_p_list)
# masking
Ne_mask_u = ~cp.isnan(Ne_BC_u_vector)
Ne_mask_v = ~cp.isnan(Ne_BC_v_vector)
Ne_mask_p = ~cp.isnan(Ne_BC_p_vector)

#=========Linearizing data structure for Drichilet BCs=========
 
drich_bc_u_list = []
drich_bc_v_list = []
drich_bc_p_list = []

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):

        if (drich_u[i] != "NDN"):
            drich_bc_u_list.append(drich_u[i])
        else:
            drich_bc_u_list.append(np.nan)
        if(drich_v[i] != "NDN"):
            drich_bc_v_list.append(drich_v[i])
        else:
            drich_bc_v_list.append(np.nan)
        if(drich_p[i] != "NDN"):
            drich_bc_p_list.append(drich_p[i])
        else:
            drich_bc_p_list.append(np.nan)
# moving to GPU memory
drich_bc_u_vector = cp.asarray(drich_bc_u_list)
drich_bc_v_vector = cp.asarray(drich_bc_v_list)
drich_bc_p_vector = cp.asarray(drich_bc_p_list)
# creating mask
drich_mask_u = ~cp.isnan(drich_bc_u_vector)
drich_mask_v = ~cp.isnan(drich_bc_v_vector)
drich_mask_p = ~cp.isnan(drich_bc_p_vector) 

print(len(rf_list),len(rgn_list),len(cf_list),len(cgn_list),len(drich_bc_u_list),len(drich_bc_v_list),len(drich_bc_p_list))
# time.sleep(7000)
print(drich_bc_u_vector)
print(drich_mask_u,len(drich_mask_u))
# time.sleep(900)

#=========================================================== Geometry check ============================================================#
lowerlimit = 0
upperlimit = 1
x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
Z = u_mat  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
plt.title("Quick Geometry Check...")
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#=======================================================================================================================================#
# target = (0.0787,9.9949)
# import numpy as np
# target = (7.5,2.0)
# target = np.array(target)

# matches = np.where((inside_pt == target).all(axis=1))[0]

# if len(matches) == 0:
#     print("Target not found:", target)
# else:
#     idx = matches[0]
#     print("Target index:", idx)

# target = (0.0787,9.9162)
# import numpy as np

# target = np.array(target)

# matches = np.where((inside_pt == target).all(axis=1))[0]

# if len(matches) == 0:
#     print("Target not found:", target)
# else:
#     idx = matches[0]
#     print("Target index:", idx)

# target = (0.1574,9.9949)
# import numpy as np

# target = np.array(target)

# matches = np.where((inside_pt == target).all(axis=1))[0]

# if len(matches) == 0:
#     print("Target not found:", target)
# else:
#     idx = matches[0]
#     print("Target index:", idx)

# time.sleep(900)

total_time_steps = 5000000
del_t = 0.0001

del_h = 1.0/(nx-1)
print("△h = ",del_h)

Re = 500        # Set Reynolds number here
etr = 1000        # Set etr here (every time iteration)
RESTART = 0   # 1 (restart) 0 (no restart)

# print(p_mat[80][79])
# time.sleep(800)
u_old = cp.asarray(u_mat)
v_old = cp.asarray(v_mat)
p_old = cp.asarray(p_mat)
# print(">>",p_mat[80][79])

sst_main = time.time()
#--------------------------------------------------------------------------------------------------------#
B_vector_sequence = []
#---------------------------------------start of time loop-----------------------------------------------#

u_p_cpu = u_mat.copy()           # u_velocity copy mesh
v_p_cpu = v_mat.copy()           # v_velocity copy mesh

u_p = cp.asarray(u_p_cpu)       # u_p moved to GPU MEMORY
v_p = cp.asarray(v_p_cpu)       # v_p moved to GPU MEMORY
    
u_copy = u_mat.copy()
v_copy = v_mat.copy() 

u_copy = cp.asarray(u_copy)     # u_copy/u_new moved to GPU MEMORY
v_copy = cp.asarray(v_copy)     # v_copy/v_new movd to GPU MEMORY

p_prime = p_mat.copy()
p_prime = cp.asarray(p_prime)   # p_prime grid moved to GPU MEMORY

if (RESTART == 1):

    start = 800001
    # =====================================
    # USER: Put file path here manually
    # =====================================

    file_path_u = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_u/Time_stack_u_t0800000.npz"
    file_path_v = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_v/Time_stack_v_t0800000.npz"
    file_path_p = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_p/Time_stack_p_t0800000.npz"

    # =====================================
    # Load NPZ file
    # =====================================

    data_u = np.load(file_path_u)
    data_v = np.load(file_path_v)
    data_p = np.load(file_path_p)

    print("Available arrays:", data_u.files)
    print("Available arrays:", data_v.files)
    print("Available arrays:", data_p.files)

    key_u = data_u.files[0]       # only one array expected
    key_v = data_v.files[0]       # only one array expected
    key_p = data_p.files[0]       # only one array expected

    U = data_u[key_u]
    V = data_v[key_v]
    P = data_p[key_p]

    u_old = U
    v_old = V
    p_old = P
    # moving to GPU memory 
    u_old = cp.asarray(u_mat)
    v_old = cp.asarray(v_mat)
    p_old = cp.asarray(p_mat)

else:
    start = 1
    pass
    

#=======================================================================================================================================#
#                                                           TIME LOOP BEGINS
#=======================================================================================================================================#
for t in range(start, total_time_steps, 1):
    
    # print("=================================================================================================")
    # print("itn = ",t,"/",total_time_steps)
    st = time.time() 
    # --- first & second interface indices generation ---
    if(t == start):
        coords_f = cp.asarray(first_interface, dtype=cp.float64)
        coords_s = cp.asarray(second_interface, dtype=cp.float64)
        coords_i = cp.asarray(inside_pt, dtype=cp.float64)
        io = cp.rint(coords_i[:,1] * conversion_factor).astype(cp.int32)
        jo = cp.rint(coords_i[:,0] * conversion_factor).astype(cp.int32)

        ipf = cp.rint(coords_f[:,1] * conversion_factor).astype(cp.int32)
        jpf = cp.rint(coords_f[:,0] * conversion_factor).astype(cp.int32)

        ips = cp.rint(coords_s[:,1] * conversion_factor).astype(cp.int32)
        jps = cp.rint(coords_s[:,0] * conversion_factor).astype(cp.int32)

    # --- velocities ---
    u = u_old[ipf, jpf]
    v = v_old[ipf, jpf]
    # print(u)

    # convection (u)
    du_dx_back = (u_old[ipf, jpf] - u_old[ipf, jpf-1]) / del_h
    du_dx_forw = (u_old[ipf, jpf+1] - u_old[ipf, jpf]) / del_h
    u_du_dx = u * cp.where(u >= 0, du_dx_back, du_dx_forw)

    du_dy_back = (u_old[ipf, jpf] - u_old[ipf-1, jpf]) / del_h
    du_dy_forw = (u_old[ipf+1, jpf] - u_old[ipf, jpf]) / del_h
    v_du_dy = v * cp.where(v >= 0, du_dy_back, du_dy_forw)

    Hu_conv_f = u_du_dx + v_du_dy

    # convection (v)
    dv_dx_back = (v_old[ipf, jpf] - v_old[ipf, jpf-1]) / del_h
    dv_dx_forw = (v_old[ipf, jpf+1] - v_old[ipf, jpf]) / del_h
    u_dv_dx = u * cp.where(u >= 0, dv_dx_back, dv_dx_forw)

    dv_dy_back = (v_old[ipf, jpf] - v_old[ipf-1, jpf]) / del_h
    dv_dy_forw = (v_old[ipf+1, jpf] - v_old[ipf, jpf]) / del_h
    v_dv_dy = v * cp.where(v >= 0, dv_dy_back, dv_dy_forw)

    Hv_conv_f = u_dv_dx + v_dv_dy

    # diffusion
    Hu_diffusion_f = (1/Re)*(u_old[ipf, jpf+1] + u_old[ipf, jpf-1] + u_old[ipf+1, jpf] + u_old[ipf-1, jpf] - 4*u_old[ipf, jpf])/(del_h**2)

    Hv_diffusion_f = (1/Re)*(v_old[ipf, jpf+1] + v_old[ipf, jpf-1] + v_old[ipf+1, jpf] + v_old[ipf-1, jpf] - 4*v_old[ipf, jpf])/(del_h**2)

    # update
    Up = u_old[ipf, jpf] + del_t*(-Hu_conv_f + Hu_diffusion_f)
    Vp = v_old[ipf, jpf] + del_t*(-Hv_conv_f + Hv_diffusion_f)

    u_p[ipf, jpf] = Up
    v_p[ipf, jpf] = Vp

    #---second interface---
    u = u_old[ips, jps]
    v = v_old[ips, jps]

    du_dx_back = (3*u_old[ips, jps] - 4*u_old[ips, jps-1] + u_old[ips, jps-2])/(2*del_h)
    du_dx_forw = (-3*u_old[ips, jps] + 4*u_old[ips, jps+1] - u_old[ips, jps+2])/(2*del_h)
    u_du_dx = u * cp.where(u >= 0, du_dx_back, du_dx_forw)

    du_dy_back = (3*u_old[ips, jps] - 4*u_old[ips-1, jps] + u_old[ips-2, jps])/(2*del_h)
    du_dy_forw = (-3*u_old[ips, jps] + 4*u_old[ips+1, jps] - u_old[ips+2, jps])/(2*del_h)
    v_du_dy = v * cp.where(v >= 0, du_dy_back, du_dy_forw)

    Hu_conv_s = u_du_dx + v_du_dy

    dv_dx_back = (3*v_old[ips, jps] - 4*v_old[ips, jps-1] + v_old[ips, jps-2])/(2*del_h)
    dv_dx_forw = (-3*v_old[ips, jps] + 4*v_old[ips, jps+1] - v_old[ips, jps+2])/(2*del_h)
    u_dv_dx = u * cp.where(u >= 0, dv_dx_back, dv_dx_forw)

    dv_dy_back = (3*v_old[ips, jps] - 4*v_old[ips-1, jps] + v_old[ips-2, jps])/(2*del_h)
    dv_dy_forw = (-3*v_old[ips, jps] + 4*v_old[ips+1, jps] - v_old[ips+2, jps])/(2*del_h)
    v_dv_dy = v * cp.where(v >= 0, dv_dy_back, dv_dy_forw)

    Hv_conv_s = u_dv_dx + v_dv_dy

    Hu_diffusion_s = (1/Re)*(u_old[ips,jps+1] + u_old[ips,jps-1] + u_old[ips+1,jps] + u_old[ips-1,jps] - 4*u_old[ips,jps])/(del_h**2)
    Hv_diffusion_s = (1/Re)*(v_old[ips,jps+1] + v_old[ips,jps-1] + v_old[ips+1,jps] + v_old[ips-1,jps] - 4*v_old[ips,jps])/(del_h**2)

    Up = u_old[ips,jps] + del_t*(-Hu_conv_s + Hu_diffusion_s)
    Vp = v_old[ips,jps] + del_t*(-Hv_conv_s + Hv_diffusion_s)

    u_p[ips,jps] = Up
    v_p[ips,jps] = Vp

    et = time.time()
    # print("time: ",et-st)
    
    # # Applying neumann BC
    u_p[rgn[Ne_mask_u], cgn[Ne_mask_u]] = u_p[rf[Ne_mask_u], cf[Ne_mask_u]]
    v_p[rgn[Ne_mask_v], cgn[Ne_mask_v]] = v_p[rf[Ne_mask_v], cf[Ne_mask_v]]
    
    # # Applying drichilet BC
    u_p[rgn[drich_mask_u],cgn[drich_mask_u]] = drich_bc_u_vector[drich_mask_u]
    v_p[rgn[drich_mask_v],cgn[drich_mask_v]] = drich_bc_v_vector[drich_mask_v]
    # print(rgn)
    

#========================================================================================================================================#
#                                                               Building [A][p'] = [B]
#========================================================================================================================================#
    # for moving and deforming bodies
    # inside_pt = time_based_inside_pt[t]     # here inside points which are going to be function of time are stored in time_based_inside_pt
    # ghost_node = time_based_ghost_node[t]   # here ghost points which are going to be function of time are stored in time_based_ghost_node
    if(t == start):
        u_p_cpu = cp.asnumpy(u_p)
        v_p_cpu = cp.asnumpy(v_p)
        p_old_cpu = cp.asnumpy(p_old) 

        sat = time.time()
        # Building co-efficient matrix A here
        A = np.zeros((len(variable_array),len(variable_array)),dtype=np.float16)
        B = []
        for i in range(0,len(variable_array),1):
            x_coord=inside_pt[i][0]
            y_coord=inside_pt[i][1]
            # At the point (x_coord, y_coord)
            # print("<M>",x_coord,y_coord)
            row = int(round((y_coord * conversion_factor),0))
            col = int(round((x_coord * conversion_factor),0)) 
            # print(">M<",row,col)
            io_cpu=row
            jo_cpu=col 
            # Find the indices of the neighboring points
            east = col+1
            west = col-1
            south = row-1
            north = row+1  
                    
            # Neighbor handling with safe check
            key_east = f'x{row}|{east}'
            key_west = f'x{row}|{west}'
            key_south = f'x{south}|{col}'
            key_north = f'x{north}|{col}'
            # print(key_east)
            # print(key_north)
            # print(key_south)
            # print(key_west)

            a = [-4]
            b_e = []
            b_vector_data = []
            if key_east in variable_array:
                east_m = variable_array.index(key_east)
                A[i][east_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t =  round((east/conversion_factor),4)
                    y_t = round((row/conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord==2):
                                print("HI-1!!!!",Ne_BC_p[ne_pos])
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
               
            if key_west in variable_array:
                west_m = variable_array.index(key_west)
                A[i][west_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((west / conversion_factor),4)
                    y_t = round((row/ conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord == 2):
                                print("HI-2!!!!",Ne_BC_p[ne_pos])
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
                      
            if key_south in variable_array:
                south_m = variable_array.index(key_south)
                A[i][south_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((col / conversion_factor),4)
                    y_t = round((south/ conversion_factor),4)
                    target = (x_t,y_t)
                    # need to convert target back into x-y coordinate
                    # print("down",target,key_south,key_west)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        # print(";;;;")
                        ne_pos = ghost_nodes_list.index(current_sub_gn)      # this line tells which edge we are dealng with
                        if (Ne_BC_p[ne_pos] != "NCN"):        # this implies a neuman condition exist 
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            a.append(1)                         
                            if (x_coord == 2 and y_coord==2):
                                print("HI-3!!!!",Ne_BC_p[ne_pos])
                                print("NNNN: ",ne_pos)
                        else:                                  # this implies a drichilet condition exist (p' = p[n+1]-p[n] = 0)
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
        
            if key_north in variable_array:
                north_m = variable_array.index(key_north)
                A[i][north_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((col / conversion_factor),4)
                    y_t = round((north/ conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            if (x_coord == 2 and y_coord== 2):
                                print("HI-4!!!!",Ne_BC_p[ne_pos])
                            a.append(1)                     # appending 1 in "a"
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass

            diag_a = np.sum(a)
            A[i][i] = diag_a
            Avg_b1 = (u_p_cpu[io_cpu,jo_cpu+1] - u_p_cpu[io_cpu,jo_cpu-1])/(2)
            Avg_b2 = (v_p_cpu[io_cpu+1,jo_cpu] - v_p_cpu[io_cpu-1,jo_cpu])/(2) 
            zeta = del_t/del_h
            const = (del_h**2/(del_t*del_h)) * (  (Avg_b1  +  2*zeta*p_old_cpu[io_cpu,jo_cpu] - zeta*p_old_cpu[io_cpu,jo_cpu-1] - zeta*p_old_cpu[io_cpu,jo_cpu+1])
                                                + (Avg_b2 + 2*zeta*p_old_cpu[io_cpu,jo_cpu] - zeta*p_old_cpu[io_cpu-1,jo_cpu] - zeta*p_old_cpu[io_cpu+1,jo_cpu]) )
            # print(p_old_cpu[io][jo-1],p_old_cpu[io][jo+1],p_old_cpu[io+1][jo],p_old_cpu[io-1][jo],p_old_cpu[io][jo],const," i = ",i,io,jo)
            # print(zeta,Avg_b1,Avg_b2,const," i = ",i,io_cpu,jo_cpu)      
            
            b_e.append(const)
            b_final = np.sum(b_e)
            B.append(b_final)
            B_vector_sequence.append(b_vector_data)
    
        B_vector_sequence_gpu = cp.asarray(B_vector_sequence)
        B_np = np.array(B, dtype=np.float64)
        A_np = np.array(A, dtype=np.float32)
        eat = time.time()
        print("A_time: ",eat-sat)
        print("--------pop-------")
        print("A matrix")
        print(A_np)

    if(t > start):
        sbt = time.time()
        Avg_b1 = (u_p[io,jo+1] - u_p[io,jo-1])/(2)  # (Up[j+1] - Up[j-1])/2
        Avg_b2 = (v_p[io+1,jo] - v_p[io-1,jo])/(2)  # (Vp[i-1] - Vp[i-1])/2
        zeta = del_t/del_h
        const = (del_h**2/(del_t*del_h)) * ( (Avg_b1  +  2*zeta*p_old[io,jo] - zeta*p_old[io,jo-1] - zeta*p_old[io,jo+1])
                                                + (Avg_b2 + 2*zeta*p_old[io,jo] - zeta*p_old[io-1,jo] - zeta*p_old[io+1,jo]) )
        b = B_vector_sequence_gpu
        b_sum = const + cp.sum(b)
        B_gpu_m = cp.asarray(b_sum, dtype=cp.float64)
        
    # A_np[6162,:] = 0
    # A_np[6162][6162] = 1
    # A_np[6162][6163] = 0
    # A_np[6083][6162]= 0
    # B_np[6162] = 0
    # if(t > start):
    #     B_gpu_m[6162] = 0

    #---------------------------------------------------------------------------------------------------------------#
    #                                                    GMRES Method
    #---------------------------------------------------------------------------------------------------------------#

    #============================================================
    # MOVE MATRIX TO GPU ONLY ONCE
    # ============================================================
    
    if t == start:
        # The matrix is converted into CSR form and moved ont GPU only once
        A_csr = sp.csr_matrix(A_np)
        A_gpu = cusparse.csr_matrix(A_csr)
        B_gpu = cp.asarray(B_np)
    if t > start:
        B_gpu = cp.asarray(B_gpu_m,dtype=cp.float64)

    sgmres = time.time()
    solution_vector, info = splinalg.gmres(A_gpu,B_gpu, tol = 1e-3, restart=20, maxiter=10)     

    egmres = time.time()

    #--------------------------------------------------------------------------------------------------------------------------------#
    #                                                               END
    #--------------------------------------------------------------------------------------------------------------------------------#
    # if(t%etr < 1e-3):
    print("itn = ",t,"/",total_time_steps)
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    print(solution_vector)
    print(info)

    p_prime[io,jo] = solution_vector
    # Neumann BCs
    p_prime[rgn[Ne_mask_p], cgn[Ne_mask_p]] = p_prime[rf[Ne_mask_p], cf[Ne_mask_p]]
    # Drichilet BCs
    p_prime[rgn[drich_mask_p],cgn[drich_mask_p]] = drich_bc_p_vector[drich_mask_p]

    p_new = p_prime + p_old        #corrected pressure copy mesh     # p(n+1) = p' + p(n)
    # print(p_new[79])
    # time.sleep(2)
    

# # --- velocities ---
#     u = u_old[ipf, jpf]
#     v = v_old[ipf, jpf]
#     # print(u)

#     # convection (u)
#     du_dx_back = (u_old[ipf, jpf] - u_old[ipf, jpf-1]) / del_h
#     du_dx_forw = (u_old[ipf, jpf+1] - u_old[ipf, jpf]) / del_h
#     u_du_dx = u * cp.where(u >= 0, du_dx_back, du_dx_forw)

#     du_dy_back = (u_old[ipf, jpf] - u_old[ipf-1, jpf]) / del_h
#     du_dy_forw = (u_old[ipf+1, jpf] - u_old[ipf, jpf]) / del_h
#     v_du_dy = v * cp.where(v >= 0, du_dy_back, du_dy_forw)

#     Hu_conv_f = u_du_dx + v_du_dy

#     # convection (v)
#     dv_dx_back = (v_old[ipf, jpf] - v_old[ipf, jpf-1]) / del_h
#     dv_dx_forw = (v_old[ipf, jpf+1] - v_old[ipf, jpf]) / del_h
#     u_dv_dx = u * cp.where(u >= 0, dv_dx_back, dv_dx_forw)

#     dv_dy_back = (v_old[ipf, jpf] - v_old[ipf-1, jpf]) / del_h
#     dv_dy_forw = (v_old[ipf+1, jpf] - v_old[ipf, jpf]) / del_h
#     v_dv_dy = v * cp.where(v >= 0, dv_dy_back, dv_dy_forw)

#     Hv_conv_f = u_dv_dx + v_dv_dy

#     # diffusion
#     Hu_diffusion_f = (1/Re)*(u_old[ipf, jpf+1] + u_old[ipf, jpf-1] + u_old[ipf+1, jpf] + u_old[ipf-1, jpf] - 4*u_old[ipf, jpf])/(del_h**2)

#     Hv_diffusion_f = (1/Re)*(v_old[ipf, jpf+1] + v_old[ipf, jpf-1] + v_old[ipf+1, jpf] + v_old[ipf-1, jpf] - 4*v_old[ipf, jpf])/(del_h**2)

    # update
    u_copy[ipf, jpf] = u_old[ipf, jpf] + del_t*(-Hu_conv_f + Hu_diffusion_f - ((p_new[ipf, jpf+1] - p_new[ipf, jpf-1])/(2*del_h)))
    v_copy[ipf, jpf] = v_old[ipf, jpf] + del_t*(-Hv_conv_f + Hv_diffusion_f - ((p_new[ipf+1, jpf] - p_new[ipf-1, jpf])/(2*del_h)))


    # #---second interface---
    # u = u_old[ips, jps]
    # v = v_old[ips, jps]

    # du_dx_back = (3*u_old[ips, jps] - 4*u_old[ips, jps-1] + u_old[ips, jps-2])/(2*del_h)
    # du_dx_forw = (-3*u_old[ips, jps] + 4*u_old[ips, jps+1] - u_old[ips, jps+2])/(2*del_h)
    # u_du_dx = u * cp.where(u >= 0, du_dx_back, du_dx_forw)

    # du_dy_back = (3*u_old[ips, jps] - 4*u_old[ips-1, jps] + u_old[ips-2, jps])/(2*del_h)
    # du_dy_forw = (-3*u_old[ips, jps] + 4*u_old[ips+1, jps] - u_old[ips+2, jps])/(2*del_h)
    # v_du_dy = v * cp.where(v >= 0, du_dy_back, du_dy_forw)

    # Hu_conv_s = u_du_dx + v_du_dy

    # dv_dx_back = (3*v_old[ips, jps] - 4*v_old[ips, jps-1] + v_old[ips, jps-2])/(2*del_h)
    # dv_dx_forw = (-3*v_old[ips, jps] + 4*v_old[ips, jps+1] - v_old[ips, jps+2])/(2*del_h)
    # u_dv_dx = u * cp.where(u >= 0, dv_dx_back, dv_dx_forw)

    # dv_dy_back = (3*v_old[ips, jps] - 4*v_old[ips-1, jps] + v_old[ips-2, jps])/(2*del_h)
    # dv_dy_forw = (-3*v_old[ips, jps] + 4*v_old[ips+1, jps] - v_old[ips+2, jps])/(2*del_h)
    # v_dv_dy = v * cp.where(v >= 0, dv_dy_back, dv_dy_forw)

    # Hv_conv_s = u_dv_dx + v_dv_dy

    # Hu_diffusion_s = (1/Re)*(u_old[ips,jps+1] + u_old[ips,jps-1] + u_old[ips+1,jps] + u_old[ips-1,jps] - 4*u_old[ips,jps])/(del_h**2)
    # Hv_diffusion_s = (1/Re)*(v_old[ips,jps+1] + v_old[ips,jps-1] + v_old[ips+1,jps] + v_old[ips-1,jps] - 4*v_old[ips,jps])/(del_h**2)

    u_copy[ips, jps] = u_old[ips,jps] + del_t*(-Hu_conv_s + Hu_diffusion_s - ((p_new[ips, jps+1] - p_new[ips, jps-1])/(2*del_h)))
    v_copy[ips, jps] = v_old[ips,jps] + del_t*(-Hv_conv_s + Hv_diffusion_s - ((p_new[ips+1, jps] - p_new[ips-1, jps])/(2*del_h)))


    et = time.time()
    # print("time: ",et-st)

        # # Applying neumann BC
    u_copy[rgn[Ne_mask_u], cgn[Ne_mask_u]] = u_copy[rf[Ne_mask_u], cf[Ne_mask_u]]
    v_copy[rgn[Ne_mask_v], cgn[Ne_mask_v]] = v_copy[rf[Ne_mask_v], cf[Ne_mask_v]]
    
    # # Applying drichilet BC
    u_copy[rgn[drich_mask_u],cgn[drich_mask_u]] = drich_bc_u_vector[drich_mask_u]
    v_copy[rgn[drich_mask_v],cgn[drich_mask_v]] = drich_bc_v_vector[drich_mask_v]
    
    # print(u_copy[79])
    # time.sleep(2)

    p_old = p_new
    u_old = u_copy
    v_old = v_copy

    #===========================SAVING DATA===============================#
    if t % etr < 1e-3:

        u_old_xt = cp.asnumpy(u_old)
        v_old_xt = cp.asnumpy(v_old)
        p_old_xt = cp.asnumpy(p_old)

        base_dir = r"D:/numerical computation/geometry meshing/Meshes"

        # ----------------------------
        # Create folders once (safe)
        # ----------------------------
        u_dir = os.path.join(base_dir, "Time_stack_u")
        v_dir = os.path.join(base_dir, "Time_stack_v")
        p_dir = os.path.join(base_dir, "Time_stack_p")

        os.makedirs(u_dir, exist_ok=True)
        os.makedirs(v_dir, exist_ok=True)
        os.makedirs(p_dir, exist_ok=True)

        print("@",t)
        tag = f"{t:07d}"   # zero padded timestep
        print("@",tag)

        # ----------------------------
        # Force numeric arrays (important)
        # ----------------------------
        u_clean = np.asarray(u_old_xt, dtype=np.float32)
        v_clean = np.asarray(v_old_xt, dtype=np.float32)
        p_clean = np.asarray(p_old_xt, dtype=np.float32)

        # ----------------------------
        # Save with UNIQUE filenames
        # ----------------------------
        u_path = os.path.join(u_dir, f"Time_stack_u_t{tag}.npz")
        v_path = os.path.join(v_dir, f"Time_stack_v_t{tag}.npz")
        p_path = os.path.join(p_dir, f"Time_stack_p_t{tag}.npz")

        np.savez(u_path, u=u_clean)
        np.savez(v_path, v=v_clean)
        np.savez(p_path, p=p_clean)

        print(f"✅ Saved timestep {t}")



#=======================================================================================================================================#
#                                                               POST-PROCESSING
#=======================================================================================================================================#
u_old_xt = cp.asnumpy(u_old,dtype=cp.float32)
v_old_xt = cp.asnumpy(v_old,dtype=cp.float32)
p_old_xt = cp.asnumpy(p_old,dtype=cp.float32)

lowerlimit = 0
upperlimit = 1

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = u_old_xt  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
ax.streamplot(X, Y, u_old, v_old, color= 'k', density = 1.5, linewidth=1)
plt.title("Lid Driven Cavity: Velocity Streamlines")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = v_old_xt  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
# plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = p_old_xt  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
# plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.title("Lid Driven Cavity: Pressure plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
    

    




