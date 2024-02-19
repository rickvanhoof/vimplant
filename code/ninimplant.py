# -*- coding: utf-8 -*-
'''
Created on Wed Sep 16 13:20:28 2020

@author: Lozano & van Hoof

implant.xyz_intersection
implant.cart2pol
implant.get_xyz
implant.transform
implant.create_cube
implant.cube_from_points
implant.recover_mask_from_points
implant.get_polar_ecc_fromCube
implant.get_translation
implant.translate_cube
'''

import math
import mathutils
import numpy as np
from math import radians
from PIL import Image, ImageDraw


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def get_xyz(data, UNDERSAMPLING = 1):
    '''
    obtains separated vectors fro x,y,z with an option to undersample
    '''
    
    
    if UNDERSAMPLING == 1:
        
        x = data[0]
        y = data[1]
        z = data[2]
      
    else:

        x = data[0][::UNDERSAMPLING]
        y = data[1][::UNDERSAMPLING]
        z = data[2][::UNDERSAMPLING]    
        
    return x, y, z

def transform(points,
              TRANSLATION_X=0,TRANSLATION_Y=0,TRANSLATION_Z=0,
              ROTATION_ANGLE_X=0,ROTATION_ANGLE_Y=0,ROTATION_ANGLE_Z=0):
    
    '''
    # The input is a matrix like this:
    # (n_points, 3 + 1)
    #   [[x1, x2]
    #    [y1, y2]
    #    [z1, z2]
    #    [1,  1]]
    
    # The output is a matrix like this:
    #   [[x1', x2']
    #    [y1', y2']
    #    [z1', z2']]
    #    [1,   1]]
    '''

    mat_rot_x = mathutils.Matrix.Rotation(radians(ROTATION_ANGLE_X), 4, 'X')
    mat_rot_y = mathutils.Matrix.Rotation(radians(ROTATION_ANGLE_Y), 4, 'Y')
    mat_rot_z = mathutils.Matrix.Rotation(radians(ROTATION_ANGLE_Z), 4, 'Z')
    
    mat_trans_x = mathutils.Matrix.Translation(mathutils.Vector((TRANSLATION_X,0,0)))
    mat_trans_y = mathutils.Matrix.Translation(mathutils.Vector((0,TRANSLATION_Y,0)))
    mat_trans_z = mathutils.Matrix.Translation(mathutils.Vector((0,0,TRANSLATION_Z)))
    
    trans_x = np.array(mat_trans_x)
    trans_y = np.array(mat_trans_y)
    trans_z = np.array(mat_trans_z)
    
    rot_x = np.array(mat_rot_x)
    rot_y = np.array(mat_rot_y)
    rot_z = np.array(mat_rot_z)

    # Join transformation matrices
#     print('Warning: in function transform, we are first rotating, then translating')
    transform = rot_x @ rot_y @ rot_z @ trans_x @ trans_y @ trans_z

    # Apply transformations and return
    return (transform @ points).astype('float32')

def create_cube(xBase, yBase, height, xmax, ymax, zmax):
    
    '''
    returns xyz coordinates of a cube volume defined by the input arguments
    '''
    
    mask_result = np.zeros((xmax,ymax,zmax))
    
    img = Image.new('L', (xmax, ymax), 0)
    
    base = [(0, 0),(yBase, xBase)]
    
    
    for i in range(height):
        
        ImageDraw.Draw(img).rectangle(base, outline=1, fill=1)
        mask = np.array(img)
        mask_result[:,:,i] = mask
    
    coords = np.where(mask_result)
    
    points = np.array([coords[0],coords[1],coords[2]]).T.astype('float32')
    aux_ones = np.ones((points.shape[0],1)).astype('float32')
    points = np.hstack((points,aux_ones))
    
    points = points.T
    
    return points, mask_result.astype(np.int16)

def cube_from_points(x, y, z):
    
    '''
    returns the coordinates of the defined x, y, z in a (n_points,4) format, adding auxiliar ones vector
    so the coordinates can be transformed later on using e.g. translation and rotation operations
    '''

    points = np.array([x,y,z]).T.astype('float32')
    aux_ones = np.ones((x.shape[0],1)).astype('float32')
    points = np.hstack((points,aux_ones))
    
    points = points.T
    
    return points


def recover_mask_from_points(points, target_shape):
    
    '''
    Creates a masked volume with ones (and zeros otherwise) from a set of x, y, z points
    '''
    
    new_mask = np.zeros(target_shape).astype(np.int16)
    
    print('new mask shape ', new_mask.shape)
    for i in range(points.shape[1]):
        
        x = int(points[0,i])
        y = int(points[1,i])
        z = int(points[2,i])
        
        new_mask[x, y, z] = 1
        
    return new_mask.astype(np.int16)


def get_polar_ecc_fromCube(cube, polar_map, ecc_map, R2_map, R2_THRESHOLD = 0, 
                           maskValue = -99,
                           ANGLE_FORMAT = 'RADIANS'):   
    

    '''
    Extracts polar angles and eccentricity values for points within a specified cube, applying a mask and R^2 threshold.

    Parameters:
    - cube (numpy.ndarray): A 4xN array where the first three rows represent the x, y, and z coordinates of N points in space. The fourth row is not used.
    - polar_map (numpy.ndarray): A 3D array containing polar angle values for each voxel.
    - ecc_map (numpy.ndarray): A 3D array containing eccentricity values for each voxel.
    - R2_map (numpy.ndarray): A 3D array containing R^2 values for each voxel, used to apply a threshold filter.
    - R2_THRESHOLD (float, optional): The minimum R^2 value required for a voxel to be considered valid. Defaults to 0.
    - maskValue (int, optional): The value used in `polar_map` and `ecc_map` to indicate invalid or masked data. Defaults to -99.
    - ANGLE_FORMAT (str, optional): The format of the returned polar angles, 'RADIANS' or 'DEGREES'. Defaults to 'RADIANS'.

    Returns:
    - pol_list (list of float): The list of polar angles for points passing the mask and R^2 threshold criteria.
    - ecc_list (list of float): The list of eccentricity values for points passing the mask and R^2 threshold criteria.
    
    The function iterates through each point in the `cube`, rounds its coordinates to the nearest integer, and retrieves the corresponding polar angle, eccentricity, and R^2 values from the provided maps. Points are filtered based on the R^2 value and maskValue criteria. The polar angles are converted to the specified format (radians or degrees) as required.    
    '''
    
    ecc_list = []
    pol_list = []
    
    for i in range(cube.shape[1]):
       
        x = int(round(cube[0,i]))
        y = int(round(cube[1,i]))
        z = int(round(cube[2,i]))
        
        p = polar_map[x,y,z]
        e = ecc_map[x,y,z]
        r2 = R2_map[x,y,z]
        
        if p != maskValue and e != maskValue and r2 > R2_THRESHOLD: 
    
            if ANGLE_FORMAT == 'RADIANS':
                pol_list.append((p))
                ecc_list.append(e)
                
            elif ANGLE_FORMAT == 'DEGREES':
                pol_list.append(radians(p))
                ecc_list.append(e)
                
    return pol_list, ecc_list  


def get_translation(cube_center, desired_center):
    '''    
    Returns TRANSLATION_X, TRANSLATION_Y, TRANSLATION_Z given 
    a desired cube center
    
    '''
    return desired_center - cube_center[0:3]


def translate_cube(cube_center, desired_center, rotation_angles, points):
    
    '''
    
    Inputs:
        cube_center (x,y,z)
        desired_center (x, y, z)
        rotation_angles (rot_x, rot_y, rot_z)
        points: (4,n_points) # xyz1
        
    Output: x y z of translated cube
    '''
    needed_translation = desired_center - cube_center[0:3]
    ROTATION_ANGLE_X, ROTATION_ANGLE_Y, ROTATION_ANGLE_Z = rotation_angles
    TRANSLATION_X, TRANSLATION_Y, TRANSLATION_Z = get_translation(cube_center, desired_center)
    
    #transform = trans @ rot
    new_cube = transform(points,
                  TRANSLATION_X,TRANSLATION_Y,TRANSLATION_Z,
                  ROTATION_ANGLE_X,ROTATION_ANGLE_Y,ROTATION_ANGLE_Z)
    x_new_cube, y_new_cube, z_new_cube = get_xyz(new_cube)
    
    return x_new_cube, y_new_cube, z_new_cube