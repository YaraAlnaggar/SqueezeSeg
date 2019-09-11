import numpy as np
from os import listdir
from os.path import isfile, join

input_dir = '../../data/lidar_2d_NH_Airsim'
out_dir = '../../data/ImageSet_NH_Airsim'
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
	input_files = []
	
	input_files.extend([f for f in listdir(input_dir) if f.split("_")[-1].split(".")[0]!="S" ])
	input_files.extend([f for f in listdir(input_dir) if f.split("_")[-1].split(".")[0]=="S" ])
	train_size = int(len(input_files) * 0.75)
	train, val = input_files[:765], input_files[765:]

	with open( join(out_dir,"all.txt"),"w" ) as f:
		for pcl in input_files:
			f.write(pcl.split(".")[0]+"\n") 

	with open( join(out_dir,"train.txt"),"w" ) as f:
		for pcl in train:
			f.write(pcl.split(".")[0]+"\n") 

	with open( join(out_dir,"val.txt"),"w" ) as f:
		for pcl in val:
			f.write(pcl.split(".")[0]+"\n") 