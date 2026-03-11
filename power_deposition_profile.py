# -*- coding: utf-8 -*-
"""
@author: kunal

This script calculates the NBI power deposited per voxel in the plasma.

An external user should only need to interact with deposition_profile. This function takes in-
1. beamEnergy: The energy of the beam in eV. (Assuming 'D' NBI for now.)
2. beamPower: The total power of the beam in Watts.
3. interpFunc: An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
              This is normally defined using sc.interpolate.RegularGridInterpolator on the density profile loaded from Pleiades.
4. beamletDiv: The divergence of the beamlet in radians.

It returns powerInterpFunc: An interpolation function that takes in (x, y, z) coordinates (in m) and returns the power density at that point in MW/m^3.
This is defined using sc.interpolate.RegularGridInterpolator on the power deposition profile calculated on a 3D voxel grid.
"""

# Stuff to allow multiprocessing to happen without issues
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Let plots display without X11 crashing
import matplotlib
matplotlib.use('TkAgg')

import time
import h5py
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
plt.rcParams.update({'font.size': 20})

from particle_positions import three_d_particle_tracks
from cross_sections import cross_sections_array

def toy_density_profile(points):
    """
    This is a toy density profile for testing purposes.
    It takes in (r, z) coordinates and returns the density at that point.

    The density is 1e10 m^-3 everywhere except in a cylindrical region where it goes to 1e19 m^-3.
    The cylindrical region is defined by r < 0.2 and -0.5 < z < 0.5.

    Parameters
    ----------
    points : array-like
        An array of shape (n, 2) where the columns correspond to r and z coordinates in meters.

    Returns
    -------
    densityArr : array-like
        An array of shape (n,) containing the density at each point in m^-3.
    """

    r = points[:,0]
    z = points[:,1]

    densityArr = 1e10 * np.ones_like(r)
    densityArr[(r < 0.2) & (z > -0.5) & (z < 0.5)] = 1e19

    return densityArr

def load_density_profile(densityProfileFilePath, makeplot=False):
    """
    This function loads the density profile from a given .hdf5 file and creates an interpolation function.


    Parameters
    ----------
    densityProfileFilePath : str
        The file path of the .hdf5 file containing the density profile.
    makeplot : bool, optional
        Whether to plot the density profile. The default is False.

    Returns
    -------
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    """

    # Open .hdf5 file
    with h5py.File(densityProfileFilePath, 'r') as hdfFile:

        # Load the mesh
        mesh = hdfFile['Mesh']
        rArr = mesh['R'][:]
        zArr = mesh['Z'][:]

        # Load the density
        equilibrium = hdfFile['Equilibrium']
        densityArr = equilibrium['n_e'][:].T

    # 1D arrays of the mesh points
    r1dArr = rArr[0, :]
    z1dArr = zArr[:, 0]

    # Create an interpolation function for the density profile
    interpFunc = sc.interpolate.RegularGridInterpolator((r1dArr, z1dArr), densityArr, method='linear', bounds_error=False, fill_value=0)

    if makeplot == True:

        # Plot the density profile
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        rArr = np.linspace(0, 0.5, 100)
        zArr = np.linspace(-1.5, 1.5, 100)
        R, Z = np.meshgrid(rArr, zArr)

        points = np.array([R.flatten(), Z.flatten()]).T
        densityArr = interpFunc(points).reshape(R.shape)

        pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label('Density [m^-3]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')

        plt.show()

    return interpFunc

def rhs_power_deposition(posAlongTrack, n, crossSectionsArr, densityArr, distArr):
    """
    This function calculates the rate of change of the beam injected fast neutrals.

    Parameters
    ----------
    index : int
        The index of the particle in the singleParticlePositions array.
    n : float
        The number density of the fast neutrals. [arbitrary scaling]
    crossSectionsArr : array-like
        The array of cross sections for charge exchange.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    singleParticlePositions : array-like
        The array of particle positions in the lab frame. 
        The shape is (position index, 3) where the columns correspond to x, y, z coordinates in meters.

    Returns
    -------
    dFastNeutrals : float
        The rate of change of the number density of fast neutrals at the given particle position.
    """

    # Get the charge exchange cross section
    sigma1_0  = crossSectionsArr[0]

    # Get the density at the given particle position
    positionIdx = np.abs(distArr - posAlongTrack).argmin()
    density = densityArr[positionIdx]

    print(f'posAlongTrack: {posAlongTrack:.2f} m, density: {density:.2e} m^-3, sigma1_0: {sigma1_0:.2e} m^2, n: {n[0]:.2e} m^-3')

    # Rate of change of fast neutrals
    dFastNeutrals = - density * sigma1_0 * n[0]
    
    return dFastNeutrals

def single_particle_deposition_profile(beamEnergy, interpFunc, singleParticlePositions, makeplot=False):
    """
    This function calculates the power deposition profile for a single particle track.
    It does this by solving the rate of change of the beam injected fast neutrals along the particle track, and then differentiating the result to get the power deposition profile.

    Parameters
    ----------
    beamEnergy : float
        The energy of the beam in eV.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    singleParticlePositions : array-like
        The array of particle positions in the lab frame. 
        The shape is (position index, 3) where the columns correspond to x, y, z coordinates in meters.
    makeplot : bool, optional
        Whether to plot the power deposition profile and density along the particle track. The default is False

    Returns
    -------
    powerDepositionArr : array-like
        The power deposition profile along the particle track. The shape is (position index,).
    """

    # Get the cross sections for the given beam energy
    crossSectionsArr = cross_sections_array(beamEnergy)

    # Distance along the particle track
    initialPos = singleParticlePositions[0]
    distArr = np.linalg.norm(singleParticlePositions - initialPos, axis=1)

    # Density along the particle track
    z = singleParticlePositions[:,2]
    r = np.sqrt(singleParticlePositions[:,0]**2 + singleParticlePositions[:,1]**2)
    points = np.array([r, z]).T
    densityArr = interpFunc(points)
    densityArr = np.maximum(densityArr, 1e15) # Density floor to avoid numerical issues

    # Mutiply the density by the cross section to get the scaling factor for the rate of change of fast neutrals
    scalingArr = densityArr.copy() * crossSectionsArr[0]

    # Cummulative integral of the scaling factor along the particle track
    cumScalingArr = np.cumsum(scalingArr * np.gradient(distArr))

    # Analytical solution for the number density of fast neutrals along the particle track
    solnArr = np.exp(-cumScalingArr)

    # Power deposition rate
    dEdl = -np.gradient(solnArr, distArr)
    dEdl = np.maximum(dEdl, 0) # Power deposition cannot be negative, so we set any negative values to 0

    # Power deposition profile (energy deposited per unit length) along the particle track
    powerDepositionArr = dEdl * np.gradient(distArr)

    if makeplot == True:

        # Plot the power deposition profile and density along the particle track
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(distArr, solnArr, label='NBI Power [arb. u.]', linewidth=4)
        ax.plot(distArr, powerDepositionArr/np.max(powerDepositionArr), label='Power Deposition Profile', linewidth=4)
        ax.plot(distArr, densityArr/1e19, label=r'Density [m$^{-3}$]', linewidth=4)

        ax.set_xlabel('Distance along particle track [m]')
        ax.legend()

        plt.show(block=True)

    return powerDepositionArr

def compute_single_particle_deposition(i, beamEnergy, interpFunc, pointsArrLab):
    """
    This is a wrapper function for single_particle_deposition_profile.
    It allows the logic of single_particle_deposition_profile to be parallelized.
    """

    return i, single_particle_deposition_profile(beamEnergy, interpFunc, pointsArrLab[i], makeplot=False)

def all_particles_deposition_profile(beamEnergy, beamPower, interpFunc, pointsArrLab):
    """
    This function calculates the power deposition profile for all particle tracks in pointsArrLab, and scales it by the beam power and the number of particles.

    Parameters
    ----------
    beamEnergy : float
        The energy of the beam in eV.
    beamPower : float
        The total power of the beam in Watts.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    pointsArrLab : array-like
        The array of particle positions in the lab frame. 
        The shape is (particle index, position index, 3) where the last dimension corresponds to x, y, z coordinates in meters.

    Returns
    -------
    powerDepositionArr : array-like
        The power deposition profile for all particle tracks, scaled by the beam power and number of particles. 
        The shape is (particle index, position index).
    """

    # Array to hold the power deposition profile for each particle
    powerDepositionArr = np.zeros(shape=(pointsArrLab.shape[0], pointsArrLab.shape[1]))

    # Worker so that single_particle_deposition_profile can be called with i
    worker = partial(compute_single_particle_deposition, 
                     beamEnergy=beamEnergy, 
                     interpFunc=interpFunc, 
                     pointsArrLab=pointsArrLab)

    # Parallelize the computation of the power deposition profile for each particle track
    with Pool(cpu_count()) as pool:
        results = pool.map(worker, range(len(pointsArrLab)))

    for i, result in results:
        powerDepositionArr[i] = result

    # Scale the power deposition profile by the beam power and the number of particles
    powerDepositionArr *= beamPower/len(pointsArrLab)

    return powerDepositionArr

def deposition_profile(beamEnergy, beamPower, interpFunc, beamletDiv, makeplot=False):
    """
    This function calculates the NBI power deposition profile on a 3D voxel grid, and creates an interpolation function for it.
    
    NOTE: The power density is where the NBI fast ions are ionized, NOT their gyrocenters. 
          To go from the birth-point to the gyrocenter needs a gyroradius correction based on the local B field.
    
    Parameters
    ----------
    beamEnergy : float
        The energy of the beam in eV. ('D' NBI assumed.)
    beamPower : float
        The total power of the beam in Watts.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
        Normally defined using sc.interpolate.RegularGridInterpolator on the density profile loaded from Pleiades.
    beamletDiv : float
        The divergence of the beamlet in radians.
    makeplot : bool, optional
        Whether to plot the density profile and power deposition profile on the midplane. The default is False.

    Returns
    -------
    powerInterpFunc : function
        An interpolation function that takes in (x, y, z) coordinates (in m) and returns the power density at that point in MW/m^3.
        Defined using sc.interpolate.RegularGridInterpolator on the power deposition profile calculated on a 3D voxel grid.
        To get the power deposition profile for a set of points, call powerInterpFunc(points) where points is an array of shape (n, 3) where the columns correspond to x, y, z coordinates in meters.
    """

    # Load the NBI particle tracks in the lab frame
    _, pointsArrLab = three_d_particle_tracks(beamletDiv)

    # Calculate the power deposition profile for all particle tracks
    powerDepositionArr = all_particles_deposition_profile(beamEnergy, beamPower, interpFunc, pointsArrLab)

    #### Calculate the power deposition profile on a 3D voxel grid

    # Flatten the power deposition and positions for binning
    pointsFlat = pointsArrLab.reshape(-1, 3)
    powerDepositionFlat = powerDepositionArr.reshape(-1)

    # 3D grid for binning the power deposition profile
    nx, ny, nz = 50, 50, 50   # resolution
    # Limits are user defined based on the expected size of the plasma
    xmin, xmax = -0.3, 0.3
    ymin, ymax = -0.3, 0.3
    zmin, zmax = -1.0, 1.0

    summedPowerDeposition, edges = np.histogramdd(pointsFlat, 
                                                  bins=(nx, ny, nz), 
                                                  range=[[xmin, xmax], 
                                                         [ymin, ymax], 
                                                         [zmin, zmax]], 
                                                  weights=powerDepositionFlat)
    
    # Convert to power density by dividing by the volume of each voxel
    voxelVolume = (xmax - xmin)/nx * (ymax - ymin)/ny * (zmax - zmin)/nz
    powerDensity = summedPowerDeposition/voxelVolume

    # Print the total power deposited by integrating the power density over the volume
    totalPowerDeposited = np.sum(powerDensity) * voxelVolume
    print(f'Total power deposited: {totalPowerDeposited/1e6:.2f} MW')

    #### Make an interpolation function for the power deposition profile

    # Compute voxel centers
    x_centers = 0.5 * (edges[0][1:] + edges[0][:-1])
    y_centers = 0.5 * (edges[1][1:] + edges[1][:-1])
    z_centers = 0.5 * (edges[2][1:] + edges[2][:-1])

    powerInterpFunc = sc.interpolate.RegularGridInterpolator((x_centers, y_centers, z_centers),
                                                             powerDensity,
                                                             bounds_error=False,
                                                             fill_value=0.0)
    
    if makeplot == True:

        fig = plt.figure(figsize=(20, 5), tight_layout=True)
        ax = fig.add_subplot(111)

        Z, X = np.meshgrid(z_centers, x_centers)
        Y = np.zeros_like(Z)

        # Z, Y = np.meshgrid(z_centers, y_centers)
        # X = np.zeros_like(Z)

        R = np.sqrt(X**2 + Y**2)
        points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        pointsPolar = np.array([R.flatten(), Z.flatten()]).T

        #### Plasma density plot

        # Plot the plasma density profile on the midplane
        densityMidplane = interpFunc(pointsPolar).reshape(X.shape) / 1e19 # Convert to 10^19 m^-3
        densityPlt = ax.contourf(Z, X, densityMidplane, 
                                 levels=100, 
                                 cmap='viridis')
        
        # Make the colorbar ticklabels nicer
        maxDensity = np.max(densityMidplane)
        print(f'Max density on midplane: {maxDensity:.2f} x 10^19 m^-3')
        ticks = np.linspace(0, int(maxDensity), 5)
        cbar = fig.colorbar(densityPlt, ticks=ticks)
        cbar.set_label(r'Plasma Density [10$^{19}$ m$^{-3}$]')

        #### Power density plot

        # Plot the power deposition profile on the midplane
        powerDensityMidplane = powerInterpFunc(points).reshape(X.shape) / 1e6 # Convert to MW/m^3
        # Ignore points where the density is very low to make the plot clearer
        maskedPowerDensityMidplane = np.ma.masked_where(powerDensityMidplane < 1, powerDensityMidplane)
        powerDensPlt = ax.contourf(Z, X, maskedPowerDensityMidplane, 
                                   levels=100, 
                                   cmap='inferno', 
                                   alpha=0.8)

        # Make the colorbar ticklabels nicer
        maxPowerDensity = np.max(powerDensityMidplane)
        print(f'Max power density on midplane: {maxPowerDensity:.2f} MW/m^3')
        ticks = np.linspace(0, int(maxPowerDensity), 5)
        cbar = fig.colorbar(powerDensPlt, ticks=ticks)
        cbar.set_label(r'NBI Power Density [MW/m$^3$]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('X [m]')

        ax.set_aspect('equal')

        plt.show(block=True)

    return powerInterpFunc

if __name__ == "__main__":

    startTime = time.time()

    # NBI parameters
    beamletDiv = 0.04 # radians
    beamEnergy = 25e3 # eV
    beamPower = 1e6 # W

    # Load the plasma density profile from Pleiades
    pleiadesPath = '/home/sanwalka/nbi_power_deposition/pleiades_260105053.h5'
    interpFunc = load_density_profile(pleiadesPath)

    # # Change to the toy density profile for testing purposes
    # # interpFunc = toy_density_profile

    # Load the NBI particle tracks in the lab frame
    _, pointsArrLab = three_d_particle_tracks(beamletDiv)

    # (r,z) points to plot the density profile
    rArr = np.linspace(0, 0.5, 100)
    zArr = np.linspace(-1.5, 1.5, 100)
    R, Z = np.meshgrid(rArr, zArr)

    points = np.array([R.flatten(), Z.flatten()]).T
    densityArr = interpFunc(points).reshape(R.shape)

    # Convert the particle tracks to (r,z) coordinates
    zArrTracks = pointsArrLab[:,:,2]
    rArrTracks = np.sqrt(pointsArrLab[:,:,0]**2 + pointsArrLab[:,:,1]**2)

    # Plot the particle tracks in the lab frame on top of the density profile
    fig = plt.figure(figsize=(15, 3), tight_layout=True)
    ax = fig.add_subplot(111)

    pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')
    cbar = fig.colorbar(pltObj, ticks=[0, 5e18, 1e19])
    cbar.set_label(r'Density [m$^{-3}$]')

    cmap = plt.get_cmap('Blues', len(zArrTracks))
    for i in range(len(zArrTracks)):
        ax.plot(zArrTracks[i,:], rArrTracks[i,:], c='dodgerblue', linewidth=0.5, alpha=0.3)

    ax.set_ylim(0, np.max(rArr))
    ax.set_xlim(np.min(zArr), np.max(zArr))
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('R [m]')
    ax.set_aspect('equal')

    # plt.close()
    plt.show(block=True)

    # Plot the deposition profile for a single particle track
    _ = single_particle_deposition_profile(25e3, interpFunc, pointsArrLab[0], makeplot=True)

    # Calculate the power deposition profile interpolation functionP@th2Fusion
    powerInterpFunc = deposition_profile(beamEnergy, beamPower, interpFunc, beamletDiv, makeplot=True)

    stopTime = time.time()
    print(f'Time taken: {(stopTime - startTime):.2f} seconds')