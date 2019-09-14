import numpy as np
from os import listdir
from os.path import isfile, join

input_dir = '../../data/lidar_1e6_2d_NH_Airsim/cw'
out_dir = '../../data/ImageSet_NH_Airsim_cw'
shuffled = False

if shuffled:
	input_files = [f for f in listdir(input_dir) if isfile(join(input_dir,f)) ] 
	train_size = int(len(input_files) * 0.8)
	train, val = input_files[:train_size], input_files[train_size:]

	with open( join(out_dir,"all.txt"),"w" ) as f:
		for pcl in input_files:
			f.write(pcl.split(".")[0]+"\n") 

	with open( join(out_dir,"train.txt"),"w" ) as f:
		for pcl in train:
			f.write(pcl.split(".")[0]+"\n") 

	with open( join(out_dir,"val.txt"),"w" ) as f:
		for pcl in val:
			f.write(pcl.split(".")[0]+"\n") 

else: 
	# validation segments are 2 and 6
	input_files = [ join("cw",f) for f in listdir(input_dir) ]
	train_files = [f for f in input_files if f.split("/")[-1][0] not in ["2","6"]]
	val_files = [f for f in input_files if f not in train_files]

	with open( join(out_dir,"all.txt"),"w" ) as f:
		for pcl in input_files:
			f.write(pcl+"\n") 

	with open( join(out_dir,"train.txt"),"w" ) as f:
		for pcl in train_files:
			f.write(pcl+"\n") 

	with open( join(out_dir,"val.txt"),"w" ) as f:
		for pcl in val_files:
			f.write(pcl+"\n") 