# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 12:20:15 2025

@author: kunal

This script contains all the relevant cross sections needed to calculate the NBI power deposition profile.
"""

import matplotlib
matplotlib.use('TkAgg')

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def f_metal_vapor(E1, a1, a2, a3, a4, a5, a6):
    """
    Sub function defined in the Tabata et. al. paper for H on sodium. 
    Same format used in the code to make comparison with the paper easier.
    """
    
    #Rydberg energy
    Er=25 # [keV]
    
    if a3=='omitted' or a4=='omitted':
        
        return a1 * ((E1/Er)**a2) / (1 + ((E1/a5)**(a2+a6)))
    
    return a1 * ((E1/Er)**a2) / (1 + ((E1/a3)**(a2+a4)) + ((E1/a5)**(a2+a6)))

def f_H1(E1, c1, c2):
    """
    Sub function defined in the Tabata et. al. paper for H on H. 
    Same format used in the code to make comparison with the paper easier.
    """
    
    #Rydberg energy
    Er=1.361e-2 # [keV]
    
    #Cross section normalization
    sig0=1e-16 # [cm^2]
    
    return sig0 * c1 * ((E1/Er)**c2)

def f_H2(E1, c1, c2, c3, c4):
    """
    Sub function defined in the Tabata et. al. paper for H on H. 
    Same format used in the code to make comparison with the paper easier.
    """
    
    return f_H1(E1,c1,c2) / (1 + (E1/c3)**(c2+c4))

def f_H3(E1, c1, c2, c3, c4, c5, c6):
    """
    Sub function defined in the Tabata et. al. paper for H on H. 
    Same format used in the code to make comparison with the paper easier.
    """
    
    return f_H1(E1,c1,c2) / (1 + (E1/c3)**(c2+c4) + (E1/c5)**(c2+c6))

def f_H4(E1, c1, c2, c3, c4, c5, c6, c7, c8):
    """
    Sub function defined in the Tabata et. al. paper for H on H. 
    Same format used in the code to make comparison with the paper easier.
    """
    
    return f_H1(E1,c1,c2) * (1 + (E1/c3)**(c4-c2))/ (1 + (E1/c5)**(c4+c6) + (E1/c7)**(c4+c8))

def f_C1(E, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    """
    Sub function defined in the Suno et. al. paper for H on C. 
    Same format used in the code to make comparison with the paper easier.
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
    """
    
    num = a1 * np.e**(-(a2/E)**a3)
    
    den1 = (E/a4)**a5
    den2 = (E/a6)**a7
    den3 = (E/a8)**a9
    den = 1 + den1 + den2 + den3
    
    return num/den

def f_C2(E, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14):
    """
    Sub function defined in the Suno et. al. paper for H on C. 
    Same format used in the code to make comparison with the paper easier.
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
    """
    
    num = a10 * np.e**(-(a11/E)**a12)
    den = 1 + (E/a13)**a14
    
    term2 = num/den
    term1 = f_C1(E, a1, a2, a3, a4, a5, a6, a7, a8, a9)
    
    return term1 + term2

def f_C3(E, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16):
    """
    Sub function defined in the Suno et. al. paper for H on C. 
    Same format used in the code to make comparison with the paper easier.
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
    """
    
    num = a10 * np.e**(-(a11/E)**a12)
    
    den1 = (E/a13)**a14
    den2 = (E/a15)**a16
    den = 1 + den1 + den2
    
    term2 = num/den
    term1 = f_C1(E, a1, a2, a3, a4, a5, a6, a7, a8, a9)
    
    return term1 + term2

def c6_cx(E):
    """
    This function defines the CX cross section of H on C^{6+}. The reaction being modelled is-
    C^{6+} + H --> C^{5+} + H^+
    
    The paper can be found here- https://doi.org/10.1016/j.adt.2006.01.001
    Paper title-
    Cross section database for carbon atoms and ions: Electron-impact ionization, excitation, and charge exchange in collisions with hydrogen atoms
    Authors-
    Hiroya Suno, Takako Kato
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
        
    Returns the cross section for the given CX reaction in m^2.
    """
    
    a1 = 2.301e2
    a2 = 5.983e3
    a3 = 3.117e-1
    a4 = 1.3e4
    a5 = 6.613e-1
    a6 = 7.299e4
    a7 = 4.895
    a8 = 2.1e4
    a9 = 1.557
    a10 = 3.355e-2
    a11 = 1.8e5
    a12 = -9.768e-2
    a13 = 4.5e3
    a14 = 3.584
    
    return 1e-20*f_C2(E, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)

def c5_cx(E):
    """
    This function defines the CX cross section of H on C^{5+}. The reaction being modelled is-
    C^{5+} + H --> C^{4+} + H^+
    
    The paper can be found here- https://doi.org/10.1016/j.adt.2006.01.001
    Paper title-
    Cross section database for carbon atoms and ions: Electron-impact ionization, excitation, and charge exchange in collisions with hydrogen atoms
    Authors-
    Hiroya Suno, Takako Kato
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
        
    Returns the cross section for the given CX reaction in m^2.
    """
    
    a1 = 1.397e2
    a2 = 1.897e4
    a3 = 2.697e-1
    a4 = 5.814e4
    a5 = 3.761
    a6 = 3.211e4
    a7 = 1.477
    a8 = 9.75e4
    a9 = 5.583
    a10 = 9.227e2
    a11 = 5.496e2
    a12 = -4.044e-1
    a13 = 7.59e-7
    a14 = 1.755e-1
    
    return 1e-20*f_C2(E, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)

def c4_cx(E):
    """
    This function defines the CX cross section of H on C^{4+}. The reaction being modelled is-
    C^{4+} + H --> C^{3+} + H^+
    
    The paper can be found here- https://doi.org/10.1016/j.adt.2006.01.001
    Paper title-
    Cross section database for carbon atoms and ions: Electron-impact ionization, excitation, and charge exchange in collisions with hydrogen atoms
    Authors-
    Hiroya Suno, Takako Kato
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
        
    Returns the cross section for the given CX reaction in m^2.
    """
    
    a1 = 9.6e1
    a2 = 3.088e5
    a3 = 1.218e-1
    a4 = 7.784e4
    a5 = 2.059
    a6 = 6.458e4
    a7 = 5.498
    a8 = 9.596e4
    a9 = 1.69
    a10 = 1.27e2
    a11 = 2.08e2
    a12 = -2.589e-1
    a13 = 2.83e2
    a14 = -3.554e-1
    a15 = 7.157e1
    a16 = -1.956
    
    return 1e-20*f_C3(E, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16)

def c3_cx(E):
    """
    This function defines the CX cross section of H on C^{3+}. The reaction being modelled is-
    C^{3+} + H --> C^{2+} + H^+
    
    The paper can be found here- https://doi.org/10.1016/j.adt.2006.01.001
    Paper title-
    Cross section database for carbon atoms and ions: Electron-impact ionization, excitation, and charge exchange in collisions with hydrogen atoms
    Authors-
    Hiroya Suno, Takako Kato
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
        
    Returns the cross section for the given CX reaction in m^2.
    """
    
    a1 = 6.239e1
    a2 = 1.873e4
    a3 = 2.589e-1
    a4 = 5.783e4
    a5 = 4.767
    a6 = 3.026e4
    a7 = 1.492
    a8 = 9.744e4
    a9 = 4.254
    a10 = 3.847e2
    a11 = 1.946e2
    a12 = -1.814e-1
    a13 = 4.743e-3
    a14 = 3.457e-1
    
    return 1e-20*f_C2(E, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)

def c2_cx(E):
    """
    This function defines the CX cross section of H on C^{2+}. The reaction being modelled is-
    C^{2+} + H --> C^{+} + H^+
    
    The paper can be found here- https://doi.org/10.1016/j.adt.2006.01.001
    Paper title-
    Cross section database for carbon atoms and ions: Electron-impact ionization, excitation, and charge exchange in collisions with hydrogen atoms
    Authors-
    Hiroya Suno, Takako Kato
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
        
    Returns the cross section for the given CX reaction in m^2.
    """
    
    a1 = 3.719e1
    a2 = 1.086e4
    a3 = 3.514e-1
    a4 = 3.785e4
    a5 = 9.584e-1
    a6 = 3.462e4
    a7 = 3.442
    a8 = 4.597e4
    a9 = 8.611e-1
    
    return 1e-20*f_C1(E, a1, a2, a3, a4, a5, a6, a7, a8, a9)

def c1_cx(E):
    """
    This function defines the CX cross section of H on C^{+}. The reaction being modelled is-
    C^{+} + H --> C + H^+
    
    The paper can be found here- https://doi.org/10.1016/j.adt.2006.01.001
    Paper title-
    Cross section database for carbon atoms and ions: Electron-impact ionization, excitation, and charge exchange in collisions with hydrogen atoms
    Authors-
    Hiroya Suno, Takako Kato
    
    E : float
        Center of mass energy of the colliding particles in eV/amu
        
    Returns the cross section for the given CX reaction in m^2.
    """
    
    a1 = 3.016e3
    a2 = 2.941e6
    a3 = 2.283e-1
    a4 = 9.976e3
    a5 = 2.911
    a6 = 2.497e3
    a7 = 1.604
    a8 = 4.969e4
    a9 = 4.98
    
    return 1e-20*f_C1(E, a1, a2, a3, a4, a5, a6, a7, a8, a9)

def ion_to_neutral_cross_section(energy,ionSpecies='D',neutralSpecies='D'):
    """
    This function calculates the CX cross section for a given fast ion species
    and background neutral.
    
    D+ to D0

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigma1_0 : float
        Charge exchange cross section [m^2].
    """
    
    #Define the cross section variable
    sigma1_0=0
    
    if ionSpecies=='D' and neutralSpecies=='D':
        
        # Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Janev-Smith cross sections are native to aurora
        sigma1_0=aurora.janev_smith_rates.js_sigma_cx_n1_q1(E)
        
    elif ionSpecies=='H' and neutralSpecies=='H':
        
        # Convert energy to keV/amu
        E=energy/(1e3)
        
        # Janev-Smith cross sections are native to aurora
        sigma1_0=aurora.janev_smith_rates.js_sigma_cx_n1_q1(E)
        
        # print(sigma1_0)
        
    elif ionSpecies=='H' and neutralSpecies=='H2':
        
        # Convert energy to keV/amu
        E=energy/1e3
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 6
        
        Eth=2.5e-3
        c1=2.12e2
        c2=1.721
        c3=6.7e-4
        c4=3.239e-1
        c5=4.34e-3
        c6=1.296
        c7=0.142
        c8=9.34
        c9=2.997
        
        E1=E-Eth
        
        sigma1_0=f_H2(E1, c1, c2, c3, c4) + f_H3(E1, c5, c2, c6, c7, c8, c9)
    
    elif ionSpecies=='D' and neutralSpecies=='D2':
        
        # Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 6
        
        Eth=2.5e-3
        c1=2.12e2
        c2=1.721
        c3=6.7e-4
        c4=3.239e-1
        c5=4.34e-3
        c6=1.296
        c7=0.142
        c8=9.34
        c9=2.997
        
        E1=E-Eth
        
        sigma1_0=f_H2(E1, c1, c2, c3, c4) + f_H3(E1, c5, c2, c6, c7, c8, c9)
        
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'Database for inelastic collisions of sodium 
        # atoms with electrons, protons, and multiply charged ions' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X08000405
        # Section 3.3
        # This formula is only valid for E>0.5keV
        
        a1=24.21
        a2=1
        a3=0
        a4=1.68e-3
        a5=5.02
        a6=-18.34
        a7=3.342
        a8=0.798
        a9=8.155e-4
        a10=3.25
        a11=1
        a12=0
        
        sigma1_0=1e-16*a1 * (np.exp(-a2*E)*(a12+np.log(a11+a3*E))/E + a4*np.exp(-a5*E)/(E**a6) + a7*np.exp(-a8/E)/(1+a9*E**a10))
        
    elif ionSpecies=='D' and neutralSpecies=='Li':
        
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=-8.21e-3
        a1=2.88e4
        a2=2
        a3=0.63
        a4=-0.71
        a5=3.72
        a6=2.96
        a7=2.8e-3
        a8=15.6
        
        E1=E-Et
        
        sigma1_0=1e-16 * (f_metal_vapor(E1,a1,a2,a3,a4,a5,a6) + a7*f_metal_vapor(E1/a8,a1,a2,a3,a4,a5,a6))
    
    elif ionSpecies=='H' and neutralSpecies=='C+':
        
        # At this stage, it should be in cm^2
        sigma1_0 = c1_cx(energy) * 1e4
        
    elif ionSpecies=='H' and neutralSpecies=='C2+':
        
        # At this stage, it should be in cm^2
        sigma1_0 = c2_cx(energy) * 1e4
        
    elif ionSpecies=='H' and neutralSpecies=='C3+':
        
        # At this stage, it should be in cm^2
        sigma1_0 = c3_cx(energy) * 1e4
        
    elif ionSpecies=='H' and neutralSpecies=='C4+':
        
        # At this stage, it should be in cm^2
        sigma1_0 = c4_cx(energy) * 1e4
        
    elif ionSpecies=='H' and neutralSpecies=='C5+':
        
        # At this stage, it should be in cm^2
        sigma1_0 = c5_cx(energy) * 1e4
        
    elif ionSpecies=='H' and neutralSpecies=='C6+':
        
        # At this stage, it should be in cm^2
        sigma1_0 = c6_cx(energy) * 1e4
        # print(sigma1_0)
    
    #Convert from cm^2 to m^2
    sigma1_0/=1e4
    
    return sigma1_0

def neutral_to_ion_cross_section(energy,ionSpecies='D',neutralSpecies='D'):
    """
    This function calculates the impact ionization cross section for a given 
    fast ion species and background neutral.
    
    D0 to D+

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigma0_1 : float
        Impact ionization cross section [m^2].
    """
    
    #Define the cross section variable
    sigma0_1=0
    
    if ionSpecies=='D' and neutralSpecies=='D':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ATOMIC AND PLASMA-MATERIAL INTERACTION DATA
        # FOR FUSION VOL. 4' which can be found here-
        # https://inis.iaea.org/collection/NCLCollectionStore/_Public/25/024/25024274.pdf
        # Section 2.2.1
        
        a1=12.899
        a2=61.897
        a3=9.2731e+3
        a4=4.9749e-4
        a5=3.989e-2
        a6=-1.59
        a7=3.1834
        a8=-3.7154
        
        sigma0_1=1e-16*a1 * ( (np.exp(-a2/E)*np.log(1+a3*E)/E) + (a4*np.exp(-a5*E)/(E**a6 + a7*E**a8)) )
        
    elif ionSpecies=='H' and neutralSpecies=='H2':
        
        #Convert energy to keV/amu
        E=energy/1e3
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 31
        
        Eth=2.0e-2
        c1=2.53e-4
        c2=1.728
        c3=2.164
        c4=7.74e-1
        c5=1.639
        c6=14.3
        
        E1=E-Eth
        
        sigma0_1= f_H2(E1, c1, c2, c3, c4) + c5*f_H2(E1/c6, c1, c2, c3, c4)
        
    elif ionSpecies=='D' and neutralSpecies=='D2':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 31
        
        Eth=2.0e-2
        c1=2.53e-4
        c2=1.728
        c3=2.164
        c4=7.74e-1
        c5=1.639
        c6=14.3
        
        E1=E-Eth
        
        sigma0_1= f_H2(E1, c1, c2, c3, c4) + c5*f_H2(E1/c6, c1, c2, c3, c4)
        
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=1.36e-2
        a1=4.4
        a2=1.79
        a3=25.2
        a4=0.3
        a5=200
        a6=1
        a7=1.8e-2
        a8=3.3e-2
        
        E1=E-Et
        
        sigma0_1=1e-16 * (f_metal_vapor(E1,a1,a2,a3,a4,a5,a6) + a7*f_metal_vapor(E1/a8,a1,a2,a3,a4,a5,a6))
        
    elif ionSpecies=='D' and neutralSpecies=='Li':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=1.36e-2
        a1=5.92
        a2=1.5
        a3=20
        a4=0.295
        a5=60
        a6=1
        
        E1=E-Et
        
        sigma0_1=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    #Convert from cm^2 to m^2
    sigma0_1/=1e4
    
    return sigma0_1

def neutral_to_negative_cross_section(energy,ionSpecies='D',neutralSpecies='Na'):
    """
    This function calculates the single electron capture cross section for a
    neutral ion on a background gas.
    
    D0 to D-

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigma0_n1 : float
        Single electron capture cross section [m^2].
    """
    
    #Define the cross section variable
    sigma0_n1=0
    
    if ionSpecies=='H' and neutralSpecies=='H2':
        
        #Convert energy to keV/amu
        E=energy/1e3
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 29
        
        Eth=2.1e-2
        c1=9.73e-3
        c2=2.38
        c3=1.39e-2
        c4=-5.51e-1
        c5=7.7e-2
        c6=2.12
        c7=1.97e-6
        c8=2.051
        c9=5.5
        c10=6.62e-1
        c11=20.2
        c12=3.62
        
        E1=E-Eth
        
        sigma0_n1= f_H3(E1, c1, c2, c3, c4, c5, c6) + f_H3(E1, c7, c8, c9, c10, c11, c12)
    
    elif ionSpecies=='D' and neutralSpecies=='D2':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 29
        
        Eth=2.1e-2
        c1=9.73e-3
        c2=2.38
        c3=1.39e-2
        c4=-5.51e-1
        c5=7.7e-2
        c6=2.12
        c7=1.97e-6
        c8=2.051
        c9=5.5
        c10=6.62e-1
        c11=20.2
        c12=3.62
        
        E1=E-Eth
        
        sigma0_n1= f_H3(E1, c1, c2, c3, c4, c5, c6) + f_H3(E1, c7, c8, c9, c10, c11, c12)
    
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=4.38e-3
        a1=6.88e5
        a2=4
        a3=0.97
        a4=-7.7e-2
        a5=2.54
        a6=3.4
        a7=8.1e-3
        a8=21
        
        E1=E-Et
        
        sigma0_n1=1e-16 * (f_metal_vapor(E1,a1,a2,a3,a4,a5,a6) + a7*f_metal_vapor(E1/a8,a1,a2,a3,a4,a5,a6))
        
    elif ionSpecies=='D' and neutralSpecies=='Li':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=4.64e-3
        a1=5.6e4
        a2=3.5
        a3=1
        a4=0.15
        a5=3.08
        a6=3.5
        a7=8e-3
        a8=20
        
        E1=E-Et
        
        sigma0_n1=1e-16 * (f_metal_vapor(E1,a1,a2,a3,a4,a5,a6) + a7*f_metal_vapor(E1/a8,a1,a2,a3,a4,a5,a6))
        
    #Convert from cm^2 to m^2
    sigma0_n1/=1e4
    
    return sigma0_n1

def ion_to_negative_ion_cross_section(energy,ionSpecies='D',neutralSpecies='Na'):
    """
    This function calculates the double electron capture cross section for a
    positive ion on a background gas.
    
    D+ to D-

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigma1_n1 : float
        Double electron capture cross section [m^2].
    """
    
    #Define the cross section variable
    sigma1_n1=0
    
    if ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=3.81e-2
        a1=82
        a2=1.45
        a3='omitted'
        a4='omitted'
        a5=4.47
        a6=1.94
        
        E1=E-Et
        
        sigma1_n1=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    elif ionSpecies=='D' and neutralSpecies=='Li':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=6.67e-2
        a1=1.68
        a2=0.83
        a3='omitted'
        a4='omitted'
        a5=8.4
        a6=3.97
        
        E1=E-Et
        
        sigma1_n1=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    #Convert from cm^2 to m^2
    sigma1_n1/=1e4
    
    return sigma1_n1

def negative_to_neutral_cross_section(energy,ionSpecies='D',neutralSpecies='Na'):
    """
    This function calculates the single electron capture loss section for a
    negative ion on a background gas.
    
    D- to D0

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigman1_0 : float
        Single electron loss cross section [m^2].
    """
    
    #Define the cross section variable
    sigman1_0=0
    
    if ionSpecies=='H' and neutralSpecies=='H2':
        
        #Convert energy to keV/amu
        E=energy/1e3
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 47
        
        Eth=2.25e-3
        c1=4.19e-2
        c2=1.89
        c3=0.178
        c4=-0.23
        c5=1.04
        c6=0.87
        c7=16.5
        c8=1.088
        c9=5.33e-3
        c10=0.166
        
        E1=E-Eth
        
        sigman1_0= f_H3(E1, c1, c2, c3, c4, c5, c6) + f_H2(E1, c7, c8, c9, c10)
    
    elif ionSpecies=='D' and neutralSpecies=='D2':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 47
        
        Eth=2.25e-3
        c1=4.19e-2
        c2=1.89
        c3=0.178
        c4=-0.23
        c5=1.04
        c6=0.87
        c7=16.5
        c8=1.088
        c9=5.33e-3
        c10=0.166
        
        E1=E-Eth
        
        sigman1_0= f_H3(E1, c1, c2, c3, c4, c5, c6) + f_H2(E1, c7, c8, c9, c10)
    
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=3.5e-4
        a1=4.56e4
        a2=1.4
        a3=0.23
        a4=0.298
        a5=1.5e2
        a6=1
        
        E1=E-Et
        
        sigman1_0=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    elif ionSpecies=='D' and neutralSpecies=='Li':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=1.7e-4
        a1=8.86e3
        a2=1
        a3=0.23
        a4=0.357
        a5=100
        a6=1
        
        E1=E-Et
        
        sigman1_0=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    #Convert from cm^2 to m^2
    sigman1_0/=1e4
        
    return sigman1_0

def negative_to_ion_cross_section(energy,ionSpecies='D',neutralSpecies='Na'):
    """
    This function calculates the double electron capture loss section for a
    negative ion on a background gas.
    
    D- to D+

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigman1_1 : float
        Double electron loss cross section [m^2].
    """
    
    #Define the cross section variable
    sigman1_1=0
    
    if ionSpecies=='H' and neutralSpecies=='H2':
        
        #Convert energy to keV/amu
        E=energy/1e3
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 49
        
        Eth=0
        c1=1.75e-8
        c2=3.88
        c3=0.906
        c4=-0.274
        c5=3.19
        c6=1.19
        
        E1=E-Eth
        
        sigman1_1= f_H3(E1, c1, c2, c3, c4, c5, c6)
    
    elif ionSpecies=='D' and neutralSpecies=='D2':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+
        # H2+, H3+, H, H2 AND H- WITH HYDROGEN MOLECULES, which can be found
        # here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X00908350
        # Eqn. 49
        
        Eth=0
        c1=1.75e-8
        c2=3.88
        c3=0.906
        c4=-0.274
        c5=3.19
        c6=1.19
        
        E1=E-Eth
        
        sigman1_1= f_H3(E1, c1, c2, c3, c4, c5, c6)
        
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=1.4e-2
        a1=1.07
        a2=2
        a3=23
        a4=0.154
        a5=200
        a6=1
        
        E1=E-Et
        
        sigman1_1=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    elif ionSpecies=='D' and neutralSpecies=='Li':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=1.4e-2
        a1=1.43
        a2=2
        a3=23
        a4=0.638
        a5=200
        a6=1
        
        E1=E-Et
        
        sigman1_1=1e-16 * f_metal_vapor(E1,a1,a2,a3,a4,a5,a6)
        
    #Convert from cm^2 to m^2
    sigman1_1/=1e4
    
    return sigman1_1

def cross_sections_array(energy, ionSpecies='D', neutralSpecies='D2'):
    """
    This function calculates all the relevant cross sections for a given fast
    ion species and background neutral, and returns them in an array.

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D2'.

    Returns
    -------
    sigmaArr : array
        Array containing all the relevant cross sections [m^2].

    """
    
    sigma_1_0  = ion_to_neutral_cross_section(energy, ionSpecies, neutralSpecies)
    sigma_0_1  = neutral_to_ion_cross_section(energy, ionSpecies, neutralSpecies)
    sigma_0_n1 = neutral_to_negative_cross_section(energy, ionSpecies, neutralSpecies)
    sigma_1_n1 = ion_to_negative_ion_cross_section(energy, ionSpecies, neutralSpecies)
    sigma_n1_0 = negative_to_neutral_cross_section(energy, ionSpecies, neutralSpecies)
    sigma_n1_1 = negative_to_ion_cross_section(energy, ionSpecies, neutralSpecies)
    
    sigmaArr = np.array([sigma_1_0, sigma_0_1, sigma_0_n1, sigma_1_n1, sigma_n1_0, sigma_n1_1])
    
    return sigmaArr

if __name__ == "__main__":

    # Plot the various cross sections vs. energy
    energyArr = np.logspace(2,5,1000) # eV

    sigma_1_0  = np.array([ion_to_neutral_cross_section(E, ionSpecies='D', neutralSpecies='D2') for E in energyArr])
    sigma_0_1  = np.array([neutral_to_ion_cross_section(E, ionSpecies='D', neutralSpecies='D2') for E in energyArr])
    sigma_0_n1 = np.array([neutral_to_negative_cross_section(E, ionSpecies='D', neutralSpecies='D2') for E in energyArr])
    sigma_1_n1 = np.array([ion_to_negative_ion_cross_section(E, ionSpecies='D', neutralSpecies='D2') for E in energyArr])
    sigma_n1_0 = np.array([negative_to_neutral_cross_section(E, ionSpecies='D', neutralSpecies='D2') for E in energyArr])
    sigma_n1_1 = np.array([negative_to_ion_cross_section(E, ionSpecies='D', neutralSpecies='D2') for E in energyArr])

    # Plot the cross sections
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(energyArr/1e3, sigma_1_0, label=r'D$^+$ to D$^0$', linewidth=3)
    ax.plot(energyArr/1e3, sigma_0_1, label=r'D$^0$ to D$^+$', linewidth=3)
    ax.plot(energyArr/1e3, sigma_0_n1, label=r'D$^0$ to D$^-$', linewidth=3)
    ax.plot(energyArr/1e3, sigma_1_n1, label=r'D$^+$ to D$^-$', linewidth=3)
    ax.plot(energyArr/1e3, sigma_n1_0, label=r'D$^-$ to D$^0$', linewidth=3)
    ax.plot(energyArr/1e3, sigma_n1_1, label=r'D$^-$ to D$^+$', linewidth=3)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Cross Section (m$^2$)')
    ax.legend()

    plt.show(block=True)