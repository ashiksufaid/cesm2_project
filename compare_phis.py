import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# File paths
# -----------------------------
restart_file = "/scratch/ashiksufaid/mod_f2000/run/control_f2000.cam.r.0011-01-01-00000.nc"
topo_file    = "my_topo_capped_himalaya.nc"

# -----------------------------
# Load datasets
# -----------------------------
ds_restart = xr.open_dataset(restart_file)
ds_topo    = xr.open_dataset(topo_file)

# -----------------------------
# Extract PHIS and convert to meters
# -----------------------------
g = 9.81

phis_restart = ds_restart["PHIS"] / g
phis_topo    = ds_topo["PHIS"] / g

# -----------------------------
# Handle dimension differences
# (topo file may not have time dim)
# -----------------------------
if "time" in phis_restart.dims:
    phis_restart = phis_restart.isel(time=0)

# -----------------------------
# Ensure same coordinate names
# -----------------------------
# Sometimes topo files use 'lat'/'lon' or 'latitude'/'longitude'
print("Restart coords:", list(ds_restart.coords))
print("Topo coords   :", list(ds_topo.coords))

# Adjust here if needed
lat_name = "lat"
lon_name = "lon"

# -----------------------------
# Select South Asia region
# -----------------------------
lat_bounds = (5, 35)
lon_bounds = (60, 100)

phis_restart = phis_restart.sel({lat_name: slice(*lat_bounds),
                                lon_name: slice(*lon_bounds)})

phis_topo = phis_topo.sel({lat_name: slice(*lat_bounds),
                          lon_name: slice(*lon_bounds)})

# -----------------------------
# Interpolate topo to restart grid (important!)
# -----------------------------
phis_topo_interp = phis_topo.interp_like(phis_restart)

# -----------------------------
# Compute difference
# -----------------------------
phis_diff = phis_restart - phis_topo_interp

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Restart
im0 = axes[0].contourf(phis_restart[lon_name], phis_restart[lat_name],
                       phis_restart, levels=30)
axes[0].set_title("Restart PHIS (m)")
plt.colorbar(im0, ax=axes[0])

# Topo
im1 = axes[1].contourf(phis_topo_interp[lon_name], phis_topo_interp[lat_name],
                       phis_topo_interp, levels=30)
axes[1].set_title("Topo File PHIS (m)")
plt.colorbar(im1, ax=axes[1])

# Difference
im2 = axes[2].contourf(phis_restart[lon_name], phis_restart[lat_name],
                       phis_diff, levels=30, cmap="RdBu_r")
axes[2].set_title("Difference (Restart - Topo) (m)")
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.savefig("PHIS comparison")

# -----------------------------
# Print key diagnostics
# -----------------------------
print("Max difference (m):", float(phis_diff.max()))
print("Min difference (m):", float(phis_diff.min()))
