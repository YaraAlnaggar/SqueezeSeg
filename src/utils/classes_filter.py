import numpy as np
from os import listdir
from os.path import isfile, join


# 	NH = {
#"ignore" : 0,
#"building" : 1,
#"sign" : 2,
#"tree" : 3,
#"car":  4,
#"road" : 5  }


input_dir = "./data/lidar_1e6_2d_NH_Airsim/cw"
output_dir = "./data/lidar_1e6_2d_NH_Airsim_wr_L1/cw" 


#L0 
# classes_ignore = [1,2]  #  buidling, sign,road  ---> 1, 2, 5
# classes_consider = [3,4,5]

# #L1
classes_ignore = [2,3]  #  sign, tree,road 
classes_consider = [1,4,5]

input_files = [ f for f in listdir(input_dir) if isfile(join(input_dir,f)) ] 

def discard(x):
	if x in classes_ignore:
		return 0
	else:
		return x

def reorder(x):

	if x in classes_consider[1:]:
		return x-2
	else:
		return x

for f in input_files:
	cloud_2d  = np.load(join(input_dir,f))
	vdiscard = np.vectorize(discard)
	vreorder = np.vectorize(reorder)
	cloud_2d[:,:,4] = vdiscard(cloud_2d[:,:,4])
	cloud_2d[:,:,4] = vreorder(cloud_2d[:,:,4])
	np.save(join(output_dir,f), cloud_2d)
	