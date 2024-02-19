'''
Created on Wed Jul 13 08:20:28 2021

@author: van Hoof & Lozano

Phosphene Simulator functions
  
  phos_elect.create_grid
  phos_elect.reposition_grid
  phos_elect.implant_grid
  phos_elect.get_phosphenes
  phos_elect.prf_to_phos
  phos_elect.normalized_uv

'''

import numpy as np
import math
from math import radians
import trimesh # needed for convex hull
from lossfunc import makeGaussian

### needed for matrix rotation/translation ect
from ninimplant import pol2cart, cart2pol,get_xyz,transform,create_cube,cube_from_points,recover_mask_from_points,get_polar_ecc_fromCube,get_translation,translate_cube

#################
def normalized_uv(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

#################
## define grid ##
#################
def create_grid(desired_center, shank_length=12, n_contactpoints_shank=5, spacing_along_xy = 1, offset_from_origin=0):
    '''
        n_contactpoints = (n_contactpoints_shank * spacing_along_xy * )^3
        shank_length in mm along grid z-axis
        base_width in mm along grid x and y axis
        offset_from_origin in mm, offset from middle contact points to the base origin
    '''
    
    base_width = n_contactpoints_shank * spacing_along_xy
    HEIGHT_DIFFERENCE = 0 #mm # base pitch/offset
    shank_length = shank_length            
    spacing_between_combs=spacing_along_xy #mm along y-axis = HORIZONTAL_DIFFERENCE
    spacing_along_shank = shank_length / n_contactpoints_shank
    contacts_perShank  = n_contactpoints_shank  
    number_of_combs = n_contactpoints_shank     
    num_shanks_perComb = n_contactpoints_shank  

    allContactPointsList = []
    contacts_xyz = []
    
    BRAIN_ANGLE = 0 # angle of the brain with respect to the horizontal line
    # Creating vectors with the position of the contacts relative to the shank origin
    # This is one shank
    contacts_position_lenght_singleShank = np.linspace(offset_from_origin, shank_length , num = contacts_perShank)
    
    # Generating the positions for one comb
    contacts_position_lenght = np.zeros((contacts_position_lenght_singleShank.shape[0],
                                         num_shanks_perComb))
    
    ## Lets generate a simple set of shanks that will become a comb
    for i in range(num_shanks_perComb):
        contacts_position_lenght[i] = contacts_position_lenght_singleShank

    ## First, define the shanks vertically:
    shankList = []

    # for each shank
    for sh in range(num_shanks_perComb):

        # we initialize 3D coordinates
        coords_shank = np.zeros((contacts_perShank,3))

        # and we fill the z axis coordinates coming from the
        # 'contacts_position_lenght' array
        for contact in range(contacts_perShank):
            coords_shank[contact, 2] = contacts_position_lenght[sh, contact]

        shankList.append(coords_shank)

    ## Second, translate each shank so they match the inter shank spacing
    shank_spacing = spacing_along_xy 
    for i in range(num_shanks_perComb):
        shankList[i][:,0] += i * shank_spacing

    #########################################
    # In addition, we will add some OFFSET FROM ORIGIN
    comb_horizontal_angle = radians(BRAIN_ANGLE)
    tan = math.tan(comb_horizontal_angle)

    shankOriginList = []
    for i in range(num_shanks_perComb):
        hor_distance_origin = i * shank_spacing
        vert_distance_origin = hor_distance_origin * tan
        shankOrigin = np.array([i * shank_spacing, 0, vert_distance_origin])

        shankList[i][:,2] += vert_distance_origin + offset_from_origin
        shankOriginList.append(shankOrigin)

    ########################################
    # calculating absolute angle (electrode -z axis)
    # 0 for a straight cube
    ELECTRODE_ABS_ANGLE = 0 
    radians_ELECTRODE_ABS_ANGLE = radians(ELECTRODE_ABS_ANGLE)
    tan = np.abs(math.tan(radians_ELECTRODE_ABS_ANGLE))

    # for each shank
    for sh in range(num_shanks_perComb):
        x_origin = shankOriginList[sh][0]
        z_origin = shankOriginList[sh][2]

        # we modify the x position of each contact point
        # according to its angle
        for contact in range(contacts_perShank):
            x_contact = shankList[sh][contact][0]
            z_contact = shankList[sh][contact][2]
            z_dist_to_origin = np.abs(z_contact - z_origin)
            x_angle = z_dist_to_origin * tan
            x_contact = x_contact + x_angle

            # Changing the x position of the contact due to the
            # absolute angle (electrode-z axis)
            shankList[sh][contact][0] = x_contact

    # Now we have a comb of electrodes ready to translate and rotate   
    comb = np.asarray(shankList).reshape(num_shanks_perComb * contacts_perShank, 3)

    aux_ones = np.ones((comb.shape[0],1)).astype('float32')
    comb = np.hstack((comb,aux_ones)).T
    combOrigin = np.asarray(shankOriginList).reshape(num_shanks_perComb, 3)
    aux_ones = np.ones((combOrigin.shape[0],1)).astype('float32')
    combOrigin = np.hstack((combOrigin,aux_ones)).T
    comb_center = np.mean(comb,axis=1)
    combOrigin_center = np.mean(combOrigin,axis=1)

    ###############################
    
    # First, center our original comb at map center/desired location
    rotation_angles = (0,0,0)
    comb_center = np.mean(comb,axis=1)
    x_new_comb, y_new_comb, z_new_comb = translate_cube(comb_center,
                                                        desired_center, 
                                                        rotation_angles,
                                                        comb)
    comb = cube_from_points(x_new_comb, y_new_comb, z_new_comb)
    allContactPointsList.append(comb)
    contacts_xyz = np.asarray(comb)[0:3, :]

    # Now, copy and translate comb
    for i in range(number_of_combs - 1):

        # Translating to obtain a parallel COMB
        rotation_angles = (0,0,0)         
        x_new_comb, y_new_comb, z_new_comb = translate_cube(comb_center,
                                                            comb_center[0:3] + [0,spacing_between_combs,HEIGHT_DIFFERENCE],
                                                            rotation_angles,
                                                            comb)    
        comb = cube_from_points(x_new_comb, y_new_comb, z_new_comb)
        allContactPointsList.append(comb)
        contacts_xyz = np.hstack((contacts_xyz, np.asarray(comb)[0:3, :]))

    # cannot scatter a single point, so we add it twice
    med_x = np.asarray([desired_center[0], desired_center[0]])
    med_y = np.asarray([desired_center[1], desired_center[1]])
    med_z = np.asarray([desired_center[2], desired_center[2]])
    
    orig_grid = contacts_xyz
    
    return orig_grid

#######################################
#######################################
def reposition_grid(orig_grid, new_location=None, new_angle=None):
            
    points = np.vstack((orig_grid, np.ones((1,orig_grid.shape[1]))))
    x_c, y_c, z_c = np.mean(orig_grid, axis=1)    
    
# rotate
    # TRANSLATION - before rotating, first move grid to rotation axis/location    
    points = transform(points,
                  -x_c, -y_c, -z_c,
                  0, 0, 0)    
    # ROTATION
    points = transform(points,
                  0,0,0,
                  new_angle[0],new_angle[1],new_angle[2])

    # TRANSLATION - move back to derired_center
    points = transform(points,
                  x_c, y_c, z_c,
                  0, 0, 0)    

# translate
    # to center
    points = transform(points,
          -x_c, -y_c, -z_c,
          0, 0, 0)

    # to new location
    points = transform(points,
          new_location[0], new_location[1], new_location[2],
          0, 0, 0)            

    x_new_cube, y_new_cube, z_new_cube = get_xyz(points)
    contacts_xyz = np.asarray([x_new_cube, y_new_cube, z_new_cube])  
            
    return contacts_xyz


#######################################
#######################################
def implant_grid(gm_mask, orig_grid, start_location, new_angle, offset_from_base):
    '''
    # determines the insertion point of the center of the electrode grid, based on angle and target point.
    '''
    
    valid = False

    # start location
    ref_orig = np.array([start_location[0], start_location[1], start_location[2]]) # ref-line vector
    ref_orig_targ = np.array([start_location[0], start_location[1], 0.0]) # ref-line vector            

    # move reference line according to new_angle             
    refline = np.transpose(np.vstack((ref_orig, ref_orig_targ)))
    refline_moved = reposition_grid(refline, start_location, new_angle)
    
    # find direction between ref start and endpoint
    ab = refline_moved[:,1] - refline_moved[:,0]        
    ref_direction = normalized_uv(ab)
    
    # convert greymatter to pointcloud and compute convex hull
    mesh = trimesh.points.PointCloud(gm_mask)
    mesh = mesh.convex_hull
    
    # create unit vector that describes direction along z-axis of the electrode grid
    ray_origins = np.array([[ start_location[0], start_location[1] , start_location[2]],
                            [ start_location[0], start_location[1] , start_location[2]]])    
    ray_directions = np.array([[ref_direction[0][0], ref_direction[0][1], ref_direction[0][2]],[ref_direction[0][0], ref_direction[0][1], ref_direction[0][2]]])    
    ray_visualize = trimesh.load_path(np.hstack((ray_origins,
                                                 ray_origins + 30 * ray_directions)).reshape(-1, 2, 3))
        
    # compute base offset    
    offset_from_origin = 0 
    zdist_center_to_base = ((np.max(orig_grid[2,:]) - np.min(orig_grid[2,:])) / 2) + offset_from_base
    base_offset = [ref_direction[0][0] * zdist_center_to_base, ref_direction[0][1] * zdist_center_to_base, ref_direction[0][2] * zdist_center_to_base]
    
    # compute intersection direction unit and convex hull
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)        
    projection = locations[0]
    
    # position grid at point where line vector touches convex hull and move center of grid into the brain by zdist_center_to_base        
    new_location = [projection[0] - base_offset[0], projection[1] - base_offset[1], projection[2] - base_offset[2]]
    
    # move original electrode grid according to new_angle
    ref_contacts_xyz = reposition_grid(orig_grid, start_location, new_angle)
    
    # move electrode grid according to new_angle
    contacts_xyz_moved = reposition_grid(orig_grid, new_location, new_angle)
#     print(ref_contacts_xyz)
#     print(new_angle)

    # check whether grid contains points outside of convex hull    
    if np.sum(mesh.contains(contacts_xyz_moved.T)) < contacts_xyz_moved.shape[1]:
        grid_valid = False
    else:
        grid_valid = True
    
    return ref_contacts_xyz, contacts_xyz_moved, refline, refline_moved, projection, ref_orig, ray_visualize, new_location, grid_valid

#######################################
#######################################
def get_phosphenes(contacts_xyz, good_coords, polar_map, ecc_map, sigma_map):

    '''
    
    creates a list of pRFs for the valid contact points
    
    '''
    n_contacts = contacts_xyz.shape[1]
    # filter good coords
    b1 = np.round(np.transpose(np.array(contacts_xyz)))
    b2 = np.transpose(np.array(good_coords))
    indices_prf = []
    for i in range(b1.shape[0]):
        tmp = np.where(np.array(b2 == b1[i,:]).all(axis=1))
        if tmp[0].shape[0] != 0:
            indices_prf.append(tmp[0][0])

    num_points = len(indices_prf)

    sList = []
    pList = []
    eList = []
    for i in indices_prf:
        xp, yp, zp = good_coords[0][i], good_coords[1][i], good_coords[2][i]
        pol = polar_map[xp, yp, zp]
        ecc = ecc_map[xp, yp, zp]
        sigma = sigma_map[xp, yp, zp]

        pList.append(pol)
        eList.append(ecc)
        sList.append(sigma)

    # normalize to range(0, 2*pi)
    eccentricities = np.asarray(eList)
    polarAngles = np.asarray(pList)
    rfSizes = np.asarray(sList)
    
    # angle x ecc x rfsize
    phosphenes = np.vstack((polarAngles, eccentricities))
    phosphenes = np.vstack((phosphenes, rfSizes))
    phosphenes = phosphenes.T

    return phosphenes

def pol2cart(angle, ecc):
    #angle in radians
    return ecc*np.cos(angle), ecc*np.sin(angle)

def get_cortical_magnification(ecc, mapping_model='wedge-dipole'):
    #parameters found by Horten & Hoyt (1991)
    a = 0.75
    b = 120
    k = 17.3
    if mapping_model == 'monopole':
        return k / (ecc + a)
    if mapping_model in ['dipole', 'wedge-dipole']:
        return k * (1 / (ecc + a) - 1 / (ecc + b))
    raise NotImplementedError

def cortical_spread(amplitude):
    #returns the radius of cortical spread in mm given a stimulation amplitude
    return np.sqrt(amplitude / 675) #from Tehovnik et al. 2007

def makeGaussian(sigma, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    """
    size = sigma*5
    x = np.arange(0, size, 1, 'float32')
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

def makeGaussian_v1(size, fwhm = 3, center=None):
    ''' Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    '''
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def gen_dummy_phos(n_phos, view_angle):
    rng = np.random.default_rng()

    max_r = view_angle / 2
    valid_ecc = np.linspace(1e-3, max_r, 1000)
    weights = get_cortical_magnification(valid_ecc, 'wedge-dipole')

    probs = weights / np.sum(weights)
    r = rng.choice(valid_ecc, size=n_phos, replace=True, p=probs)
    #phi = 2 * np.pi * rng.random(n_phos)
    phi = np.pi * rng.random(n_phos)

    phosphenes = np.zeros((n_phos,2))
    phosphenes[:, 0] = phi
    phosphenes[:, 1] = r
    return phosphenes

#######################################
#######################################
def prf_to_phos(phospheneMap, phosphenes, view_angle = 90, phSizeScale = 1):
    '''
            1- map phosphenes to visual scene -> depends on distance eyes/cam to plot
            90deg eccentricity -> 90% of window size

            - phSizeScale = scaling phosphenes for easier visualization

        '''

    windowSize = phospheneMap.shape[1]

    #degrees to pixels
    scaledEcc = windowSize/ view_angle #pixels per degree of visual angle

    for i in range(0, phosphenes.shape[0]):
        s = int(phosphenes[i, 2]*scaledEcc) #phosphene size in pixels

        c_x, c_y = pol2cart(radians(phosphenes[i, 0]), phosphenes[i, 1])
        x = int(c_x * scaledEcc + windowSize / 2)
        y = int(c_y * scaledEcc + windowSize / 2)

        if s < 2:            
            s = 2 # print('Tiny phosphene: artificially making size == 2')

        elif (s % 2) != 0:
            s = s + 1
        else:
            None

        g = makeGaussian(sigma=s, center=None) #so this assumes that sigma is equal to the size of the
        g /= g.max()
        half_gauss = g.shape[0]//2

        try:
            phospheneMap[y - half_gauss:y + half_gauss, x - half_gauss:x + half_gauss] += g
        except:
            None #print('error... (probably a phosphene on the edge of the image')

    # rotate by 90 degrees to match orientation visual field
    phospheneMap = np.rot90(phospheneMap, 1)

    return phospheneMap

#######################################
#######################################
def phos_density(phospheneMap, phosphenes, max_eccentricity = 180, phSizeScale = 1):
    '''
        - map phosphenes to visual scene -> depends on distance eyes/cam to plot
        90deg eccentricity -> 90% of window size
        
        - phSizeScale = scaling phosphenes for easier visualization
        
    '''

    # create phosphene map
    windowSize = phospheneMap.shape[1]
    scaledEcc =  windowSize / max_eccentricity   

    # for i in range(total_phosphenes):
    for i in range(0,phosphenes.shape[0]):
        
        s = int(phosphenes[i,2]) * phSizeScale
        c_x, c_y = pol2cart(radians(phosphenes[i,0]) ,phosphenes[i,1])

        x = int(c_x * scaledEcc + windowSize/2)
        y = int(c_y * scaledEcc + windowSize/2)
        
        if s < 2:
            # print('Tiny phosphene: artificially making size == 2')
            s = 2

        elif (s % 2) != 0:
            s =  s + 1
        else:
            None

        halfs = s // 2        

        g = makeGaussian(size = s , fwhm = s / 3, center=None)
        g /= g.max()
        g /= g.sum()
        
        g = np.expand_dims(g,-1)

        try:
            phospheneMap [y-halfs:y+halfs, x-halfs:x+halfs] = phospheneMap[y-halfs:y+halfs, x-halfs:x+halfs] + g
        except:
            None
    
    # rotate by 90 degrees to match orientation visual field
    phospheneMap = np.rot90(phospheneMap, 1)    
    
    # get density values
    density_vals = []
    for i in range(0,phosphenes.shape[0]):
        c_x, c_y = pol2cart(radians(phosphenes[i,0]) ,phosphenes[i,1])

        x = int(c_x * scaledEcc + windowSize/2)
        y = int(c_y * scaledEcc + windowSize/2)    
        
        density_vals.append(phospheneMap[x, y])
        
        
    return phospheneMap, density_vals