# -*- coding: utf-8 -*-
"""
@author: kunal

This script generates the particle tracks of the beam injected fast neutrals. 
It makes them in beam-aligned coordinates, and then transforms them to the lab frame.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
from nbi_radial_dist import radial_weight

def position_tracker(x0, y0, angle, xLocArr):
    
    # Slope of the line
    slope = np.tan(angle)
    
    # y = m*(x - x_1) + y_1
    yLocArr = slope*(xLocArr-x0) + y0
    
    return xLocArr, yLocArr

def generate_particle_tracks(beamletDiv, startPos, stopPos, numXPos, makeplot=False):
    """
    This function generates a set of particle tracks given the NBI parameters.
    This is an axisymmetrized function. i.e. the beam is modelled as a 1D
    emitter of neutral particles and their positions are tracked in 2D.

    With the current settings, it makes 1034 particle tracks. 
    This is a good number to get a smooth power deposition profile without being too computationally expensive.

    Parameters
    ----------
    beamletDiv : float
        Divergence of the beamlet.
        Divergence is defined in the documentation for np.random.normal.
    startPos : float
        Starting x position of the particle tracks.
    stopPos : float
        Ending x position of the particle tracks.
    numXPos : int
        Number of positions along the beam path where we want to track the beam particle positions.
    makeplot : bool, optional
        Whether to make a plot of the particle tracks. The default is False.

    Returns
    -------
    trackingXPosArr : np.array
        1D array that contains all the x positions for the particles.
    trackingYPosArr : np.array
        2D array that contains all the y positions for the particles.
        1st index- particle number
        2nd index- y position
    """
    
    # Curvature of the NBI grid
    gridCurve = 3.4 # m

    # Radius of the NBI grid
    gridRad = 4/39.37 # m

    #### Computational parameters

    # JV: Number of discrete beamlets radii (must be odd the preserve rotational symmetry, r = 0 case)
    # JV: Includes +/- r so number of radii = numBeamlets / 2 (for even) or (numBeamlets - 1) / 2 (for odd) 
    numBeamlets = 20

    # JV: Square rooted number of beamlet points on a square beamlet grid
    numTotal = 12 #JV: NEW NUMBER OF BEAMLETS IN GRID, SQUARE TO FIND NUMBER OF BEAMLET LOCATIONS IN SQUARE GRID

    # Number of particles per beamlet
    particlesBeamlet = 11

    # x values at which we want the particle tracks
    trackingXPosArr = np.linspace(startPos, stopPos, numXPos)

    # Angle subtended by the outermost grid point
    gridAngle = np.arctan(gridRad/np.max(gridCurve)) #radians

    # Array of constantly spaced angles
    angleArr = np.linspace(-gridAngle, gridAngle, numBeamlets) #radians

    # Get the launch point positions in cartesian coordinates
    gridXValsArr = gridCurve*np.cos(angleArr)
    gridYValsArr = gridCurve*np.sin(angleArr)

    ############# JV: Weight launch positions by radii ###########################
    weights = radial_weight(numTotal, gridYValsArr).counter(cart=False, plots=0)
    gridXValsArr = np.repeat(gridXValsArr, weights)
    gridYValsArr = np.repeat(gridYValsArr, weights)
    angleArr = np.repeat(angleArr, weights)
    numBeamlets = len(gridYValsArr) # update number of beamlets

    # Flip the x values so the beam launches from left to right
    gridXValsArr = -gridXValsArr

    # Move the x values by the curvature radius so the middle of the grid is at (0,0)
    gridXValsArr += gridCurve

    # Angles at which 0 divergence particles launch from the beamlet
    launchAngleArr = (np.pi-angleArr) % np.pi

    #### Launch and track all the particles

    # Array to store the y values of the particle tracks. It is a 2D array
    # 1st index- particle number
    # 2nd index- x position
    trackingYPosArr = np.zeros(shape=(numBeamlets*particlesBeamlet, len(trackingXPosArr)))

    # Index of the current particle being tracked
    currParticleNumber = 0

    # Go over each beamlet
    for k in range(len(gridXValsArr)):
        
        # x and y positions
        xCurr = gridXValsArr[k]
        yCurr = gridYValsArr[k]
        
        # Get the slopes of the particles
        slopesArr = np.random.normal(loc=launchAngleArr[k], scale=beamletDiv, size=(particlesBeamlet))

        # Go over each slope and track the particle positions
        for j in range(len(slopesArr)):
            
            # Get the particle track
            _, yArr = position_tracker(xCurr, yCurr, slopesArr[j], trackingXPosArr)
            
            # Append yArr to trackingYPosArr
            trackingYPosArr[currParticleNumber] = yArr
            
            currParticleNumber+=1

    if makeplot:
    
        # Plot the particle tracks
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        for i in range(len(trackingYPosArr)):
            if i%1 == 0:
                ax.plot(trackingXPosArr, trackingYPosArr[i], color='blue', alpha=0.1)

        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.set_title(f'Beamlet Divergence = {np.round(beamletDiv*1e3, 2)} mrad')

        ax.set_aspect('equal')

        plt.show(block=True)

    return trackingXPosArr, trackingYPosArr

def three_d_particle_tracks(beamletDiv, makeplot=False):
    """
    This function generates the 3D particle tracks of the beam injected fast neutrals.
    It first generates the 2D tracks in beam-aligned coordinates, and then transforms them to the lab frame.

    Parameters
    ----------
    beamletDiv : float
        Divergence of the beamlet. [radians]
    makeplot : bool, optional
        Whether to make a plot of the particle tracks. The default is False.

    Returns
    -------
    pointsArrBeam : np.array
        3D array that contains all the particle positions in the beam frame.
        1st index- particle number
        2nd index- position along the track
        3rd index- x, y, or z position
    pointsArrLab : np.array
        3D array that contains all the particle positions in the lab frame.
        1st index- particle number
        2nd index- position along the track
        3rd index- x, y, or z position
    """

    # Number of positions along the particle track where we want to track the particle positions
    numPositions = 100

    # Get the 2D particle tracks in beam-aligned coordinates
    # z is along the beam path, r is the radial position from the beam center
    trackingZPosArr, trackingRPosArr = generate_particle_tracks(beamletDiv, startPos=3, stopPos=4, numXPos=numPositions, makeplot=False)

    # Make a random angle array to transform the 2D tracks to 3D
    angleArr = np.random.uniform(low=0, high=2*np.pi, size=trackingRPosArr.shape[0])

    # Transform the 2D tracks to 3D
    trackingXPosArr = trackingRPosArr * np.cos(angleArr[:, np.newaxis])
    trackingYPosArr = trackingRPosArr * np.sin(angleArr[:, np.newaxis])

    # Tile the z positions to match the shape of the x and y position arrays
    trackingZPosArr = np.tile(trackingZPosArr, (len(trackingXPosArr), 1))

    #### Convert to lab frame
    # Position of the beam center in the lab frame
    beamPos = np.array([2.475, 0, 2.475]) # m

    # Angle of the NBI
    beamAngle = 3*np.pi/4 # radians

    # Shift the origin to the lab frame
    machineXPosArr = beamPos[0] + np.cos(beamAngle)*trackingXPosArr - np.sin(beamAngle)*trackingZPosArr
    machineYPosArr = beamPos[1] + trackingYPosArr
    machineZPosArr = beamPos[2] + np.sin(beamAngle)*trackingXPosArr + np.cos(beamAngle)*trackingZPosArr

    # Make a points array in the lab and beam frames
    pointsArrLab = np.zeros(shape=(np.shape(trackingXPosArr)[0], np.shape(trackingXPosArr)[1], 3))
    pointsArrLab[:,:,0] = machineXPosArr
    pointsArrLab[:,:,1] = machineYPosArr
    pointsArrLab[:,:,2] = machineZPosArr

    pointsArrBeam = np.zeros(shape=(np.shape(trackingXPosArr)[0], np.shape(trackingXPosArr)[1], 3))
    pointsArrBeam[:,:,0] = trackingXPosArr
    pointsArrBeam[:,:,1] = trackingYPosArr
    pointsArrBeam[:,:,2] = trackingZPosArr

    if makeplot:

        startingXPosArr = pointsArrBeam[:,0,0]
        startingYPosArr = pointsArrBeam[:,0,1]
        startingRPosArr = np.sqrt(startingXPosArr**2 + startingYPosArr**2)

        fig = plt.figure(figsize=(20, 8), tight_layout=True)
        # Scatterpoint of the starting positions of the particle tracks in the beam frame
        ax1 = fig.add_subplot(121)
        # Histogram of the starting positions of the particle tracks in the beam frame
        ax2 = fig.add_subplot(122)

        ax1.scatter(startingXPosArr, startingYPosArr, color='blue', alpha=0.7)
        ax2.hist(startingRPosArr.flatten(), bins=50, color='blue', alpha=0.7)

        ax1.set_title('Beam Frame')
        ax1.set_xlabel('x position (m)')
        ax1.set_ylabel('y position (m)')
        ax1.set_aspect('equal')

        ax2.set_xlabel('r position (m)')
        ax2.set_ylabel('Number of particles')
        
        plt.suptitle(f'Beamlet Divergence = {np.round(beamletDiv*1e3, 2)} mrad')

        plt.show(block=True)

        #### Plot of the particle tracks in the lab frame

        fig = plt.figure(figsize=(15, 8), tight_layout=True)
        fig.suptitle('Lab Frame')

        # Scatter plot of the start of the particle tracks in the lab frame
        ax1 = fig.add_subplot(121)

        ax1.scatter(pointsArrLab[:,0,2], pointsArrLab[:,0,1], color='blue', alpha=0.7)
        ax1.set_xlabel('z position (m)')
        ax1.set_ylabel('y position (m)')
        ax1.set_aspect('equal')

        # Plot of the particle tracks in the lab frame
        ax2 = fig.add_subplot(122)

        ax2.scatter(pointsArrLab[:,0,0], pointsArrLab[:,0,1], color='blue', alpha=0.7)
        ax2.set_xlabel('x position (m)')
        ax2.set_ylabel('y position (m)')
        ax2.set_aspect('equal')

        plt.show(block=True)

    return pointsArrBeam, pointsArrLab

if __name__ == "__main__":

    # Divergence of the beamlet
    beamletDiv = 0.02 # radians

    pointsArrBeam, pointsArrLab = three_d_particle_tracks(beamletDiv, makeplot=False)

    print(f'Number of particles: {np.shape(pointsArrBeam)[0]}')
    print(f'Number of positions tracked per particle: {np.shape(pointsArrBeam)[1]}')