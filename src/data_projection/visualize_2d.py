
import numpy as np
from project import *

file = "/home/admin1/yara/old_airsim/AirSim/PythonClient/lidar_collecting/data_1e6/counter_cw/0/0_20_S.npy"

point_cloud = np.load(file)
projected_2d = get_2d(point_cloud, visualize = True)

