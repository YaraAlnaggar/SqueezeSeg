#!/usr/bin/python2
# -*- coding: utf-8 -*-


import sys
import os.path
import numpy as np
from utils.util import *


ZENITH_LEVEL = 64
AZIMUTH_LEVEL = 512
INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 12.12]]]) # Need to be changed to accomodate airsmith environment
INPUT_STD = np.array([[[11.47, 6.91, 0.86, 12.32]]])  # Need to be changed to accomodate airsmith environment




def spherical_project(point_cloud):

    np_p = np.load(point_cloud)
    # perform fov filter by using hv_in_range
    cond = self.hv_in_range(x=np_p[:, 0],
                            y=np_p[:, 1],
                            z=np_p[:, 2],
                            fov=[-45, 45])

    np_p_ranged = np_p[cond]
    # get depth map
    lidar = self.pto_depth_map(velo_points=np_p_ranged, C=7)
    lidar_f = lidar.astype(np.float32)

    # to perform prediction
    lidar_mask = np.reshape(
        (lidar[:, :, 6] > 0),
        [ZENITH_LEVEL, AZIMUTH_LEVEL, 1]
    )
    lidar_f = (lidar_f - INPUT_MEAN) / INPUT_STD
    return lidar_f

def hv_in_range( x, y, z, fov, fov_type='h'):
    """
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit

    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`

    Return:
    `cond`:condition of points within fov or not

    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' "
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), \
                              np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), \
                              np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")

def pto_depth_map(velo_points, H=64, W=512, C=7, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):

    x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]
    r, g, b = velo_points[:, 3], velo_points[:, 4], velo_points[:, 5]
    
    d = np.sqrt(x ** 2 + y ** 2 + z**2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d==0] = 0.000001
    r[r==0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y/r)
    phi_ = (phi/dphi).astype(int)
    phi_[phi_<0] = 0
    phi_[phi_>=512] = 511

    theta = np.radians(2.) - np.arcsin(z/d)
    theta_ = (theta/dtheta).astype(int)
    theta_[theta_<0] = 0
    theta_[theta_>=64] = 63


    depth_map = np.zeros((H, W, C))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = r
    depth_map[theta_, phi_, 4] = g
    depth_map[theta_, phi_, 5] = b
    depth_map[theta_, phi_, 6] = d

    return depth_map


spherical_project(".npy")