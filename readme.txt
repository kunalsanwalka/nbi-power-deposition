The set of codes in this directory are used to generate an interpolation function for the deposited NBI energy.

In principle, the user only needs to interact with deposition_profile() in power_deposition_profile.py.
That function generates an interpolation function for P_NBI(x,y,z) for a given density interpolation function. The density interpolation function itself can be generated based on a pleaides output.
In the current code, the density interpolation function is generated from a pleiades output via load_density_profile() in power_deposition_profile.py

To load the cross sections for CX, this code uses the Aurora python package. This is not a default package in conda and may need to be installed seperately for this code to work.

TODO for Kunal: Make the interpolation function output from deposition_profile() be based on the gyrocenter of the particles rather than the birth points of the fast ions themselves.