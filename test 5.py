import numpy as np
from pyevtk.hl import imageToVTK
import os
import matplotlib.pyplot as plt

# Output directory (make sure it exists)
outdir = r"D:/numerical computation/geometry meshing/Meshes/Time_stack"     # location where time-step .vti files are saved
timestepss = np.load(r"D:/numerical computation/geometry meshing/Meshes/Time_stack.npz", allow_pickle=True) # file containing all time-steps
timesteps = timestepss["array1"] # from the above file extract the array of time-steps (genrally the 1st array)

nx, ny = timesteps[0].shape[1], timesteps[0].shape[0] # dimensions of the 2D data

# Grid spacing
dx, dy = 1.0, 1.0

for t, data in enumerate(timesteps):
    # Expand to 3D (ny, nx, 1)
    data3d = data.reshape(ny, nx, 1).astype(np.float32)

    # Base filename: "time_data_t0", "time_data_t1", ...
    filepath = os.path.join(outdir, f"time_data_t{t}")

    # Write VTI file
    imageToVTK(filepath,
               origin=(0.0, 0.0, 0.0),
               spacing=(dx, dy, 1.0),
               pointData={"field": data3d})
