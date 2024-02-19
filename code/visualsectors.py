# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:11:08 2021

@author: Lozano

"""

import numpy as np
from matplotlib import pyplot as plt

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    
    """

    x = np.arange(0, size, 1, 'float32')
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def sector_mask(shape,centre,radiusLow,radiusHigh,angle_range):

    """
    source original code (only sector):
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    
    *Modification: I modified the code 
    so a LowAngle and a HighAngle can be defined for the sector
    
    """
    
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi
    
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    
    # circular mask
    circmaskLow = r2 >= radiusLow*radiusLow
    circmaskHigh = r2 <= radiusHigh*radiusHigh
    circmask = circmaskLow* circmaskHigh
    
    # angular mask
    anglemask = theta <= (tmax-tmin)
    
    return circmask*anglemask

#%% Complete gaussian
def complete_gauss(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=500, center=None, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)

    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    angle1 = -90
    angle2 = 90
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Outer ring
def outer_ring(windowsize=1000, fwhm=400, radiusLow=250, radiusHigh=500, center=None, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)

    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    angle1 = -90
    angle2 = 90
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Inner ring
def inner_ring(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=250, center=None, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)
    ms = matrix.shape
    ms_2 = int(ms[0]/2)

    angle1 = -90
    angle2 = 90
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Upper sector 45 degrees
def upper_sector(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=500, center=None, angle1=-90, angle2=-45, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)
    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
        
    return matrix

#%% Lower sector 45 degrees
def lower_sector(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=500, center=None, angle1=45, angle2=90, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)

    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix