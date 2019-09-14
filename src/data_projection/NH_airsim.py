import numpy as np
from project import *
from os import listdir
from os.path import isfile, join

input_dir = '/home/admin1/yara/old_airsim/AirSim/PythonClient/lidar_collecting/data_1e6/counter_cw'
# input_dir = '/home/admin1/yara/old_airsim/AirSim/PythonClient/lidar_collecting/data'

out_dir = '../../data/lidar_1e6_2d_NH_Airsim/counter_cw'
# out_dir = '../../data/lidar_2d_NH_Airsim'

input_files = [join(input_dir,sub,f) for sub in listdir(input_dir) for f in listdir(join(input_dir, sub)) if isfile(join(input_dir,sub,f))]
# print(input_files)

for file in input_files:
	point_cloud = np.load(file)
	projected_2d = get_2d(point_cloud)
	np.save(join(out_dir, file.split("/")[-1]),projected_2d )

