import numpy as np
from project import *
from os import listdir
from os.path import isfile, join


def discard(x):
	if x in classes_ignore:
		return 0
	else:
		return x

def reorder(x):

	if x in classes_consider[1:]:
		return x-1
	else:
		return x



input_dir = '/home/admin1/yara/old_airsim/AirSim/PythonClient/lidar_collecting/data_1e6/cw'
# input_dir = '/home/admin1/yara/old_airsim/AirSim/PythonClient/lidar_collecting/data'

# out_dir = '../../data/lidar_1e6_2d_NH_deg3_Airsim/cw'
out_dir = '/home/admin1/yara/local/data/lidar_1e6_2d_NH_deg3_Airsim/cw'

input_files = [join(input_dir,sub,f) for sub in listdir(input_dir) for f in listdir(join(input_dir, sub)) if isfile(join(input_dir,sub,f))]
# print(input_files)

classes_ignore = [2]  #  sign
classes_consider = [1,3,4,5]



vdiscard = np.vectorize(discard)
vreorder = np.vectorize(reorder)



for file in input_files:
	point_cloud = np.load(file)
	projected_2d = get_2d(point_cloud)
	projected_2d[:,:,4] = vdiscard(projected_2d[:,:,4])
	projected_2d[:,:,4] = vreorder(projected_2d[:,:,4])
	np.save(join(out_dir, file.split("/")[-1]),projected_2d )

