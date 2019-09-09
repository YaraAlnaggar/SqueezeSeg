import numpy as np
from project import *
from os import listdir
from os.path import isfile, join

input_dir = '/home/admin1/yara/old_airsim/AirSim/PythonClient/lidar_collecting/data'
out_dir = '../../data/lidar_2d_NH_Airsim'

input_files = [f for f in listdir(join(input_dir, sub)) for sub in listdir(input_dir) if isfile(join(input_dir, f))]
print(input_files)

for file in input_files:
	point_cloud = np.load(join(input_dir, file))
	projected_2d = get_2d(point_cloud)
	np.save(join(out_dir, f))

