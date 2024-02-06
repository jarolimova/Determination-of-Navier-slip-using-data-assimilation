Running assimilation
====================

Main script: *assimilation.py* - given mesh and data find the solution which minimized the error functional assuming the NS equations hold

Run *python3 assimilation.py --h* to see all the possible settings

Preparing data
==============
- *generate_meshes.sh* - create synthetic surface meshes
- *simulate_artificial_data.py* - compute the flow on given 3D mesh with different theta values and average inflow velocity
- *make_data3D.py* - interpolate computed velocity to given (coarser and shorter) mesh and add noise
