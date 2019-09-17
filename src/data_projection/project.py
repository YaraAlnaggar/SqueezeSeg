import numpy as np
from collections import Counter
from PIL import Image

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def hv_in_range(x, y, z, fov, fov_type='h'):
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

def pto_depth_map(velo_points, H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
        """
        Project velodyne points into front view depth map.

        :param velo_points: velodyne points in shape [:,4]
        :param H: the row num of depth map, could be 64(default), 32, 16
        :param W: the col num of depth map
        :param C: the channel size of depth map
            3 cartesian coordinates (x; y; z),
            an intensity measurement and
            range r = sqrt(x^2 + y^2 + z^2)
        :param dtheta: the delta theta of H, in radian
        :param dphi: the delta phi of W, in radian
        :return: `depth_map`: the projected depth map of shape[H,W,C]
        """

        x, y, z, label = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2],velo_points[:, 3]
        d = np.sqrt(x ** 2 + y ** 2 + z**2)
        r = np.sqrt(x ** 2 + y ** 2)
        d[d==0] = 0.000001
        r[r==0] = 0.000001
        phi = np.radians(45.)  - np.arcsin(y/r) 
        phi_ = (phi/dphi).astype(int)
        phi_[phi_<0] = 0
        phi_[phi_>=W] = W-1

        theta = np.radians(3.0) - np.arcsin(z/d)
        theta_ = (theta/dtheta).astype(int)
        theta_[theta_<0] = 0
        theta_ =  theta_
        theta_[theta_>=H] = H-1
        depth_map = np.zeros((H, W, C))

        if C == 5:
            depth_map[theta_, phi_, 0] = x
            depth_map[theta_, phi_, 1] = y
            depth_map[theta_, phi_, 2] = z
            depth_map[theta_, phi_, 3] = d
            depth_map[theta_, phi_, 4] = label
        else:
            depth_map[theta_, phi_, 0] = label

        return depth_map


def get_2d(points_3d, W = 512, H = 64, C = 5, visualize = False):

	cond = hv_in_range(x=points_3d[:, 0],
	                   y=points_3d[:, 1],
	                   z=points_3d[:, 2],
	                   fov=[-45, 45])

	points_3d_ranged = points_3d[cond]
	lidar = pto_depth_map(points_3d_ranged)

	lidar_f = lidar.astype(np.float32)

	if visualize:
		pick_color = {
			0: (0,0,0), 
			1 : (107, 46, 5),      # cement 
			2  : (255, 0, 0),    # light brown
			3:  (64, 148, 4),      # green
			4:  (165, 115, 14),    # brown
		    5 :  (29, 61, 0) ,      # dark green
		    6 : (179, 179, 177),   # light grey 
		    7: (102, 102, 101) ,     # dark grey road
		    8 : (163, 91, 163),     # light brown
		    9 :  (0,0,255),        # red 
		    }    
		file_name = 'visual.png'
		print("in pic", Counter(lidar[:, :, 4].flatten()))
		colors = [ pick_color[int(num)] for num in lidar[:, :, 4].flatten() ]
		img = Image.new('RGB',(512,64))
		img.putdata(colors)
		img.save(file_name)
		print(file_name, " as label is saved")
			
    # generated depth map from LiDAR data

	return lidar_f        


