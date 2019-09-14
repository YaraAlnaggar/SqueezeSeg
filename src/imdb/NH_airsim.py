# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017

"""Image data base class for kitti"""

import os 
import numpy as np
import subprocess

from .imdb import imdb

class NH_airsim(imdb):
	def __init__(self, image_set_dir, image_set, data_path, mc, level):
		imdb.__init__(self, 'NH_airsim_'+image_set, mc)
		self._image_set = image_set
		self._image_set_dir = image_set_dir
		self._data_root_path = data_path
		if level == 5:
			self._lidar_2d_path = os.path.join(self._data_root_path, 'lidar_1e6_2d_NH_Airsim')
		else :
			self._lidar_2d_path = os.path.join(self._data_root_path, 'lidar_1e6_2d_NH_Airsim_L' + str(level))

		#self._gta_2d_path = os.path.join(self._data_root_path, 'gta')

		# a list of string indices of images in the directory
		self._image_idx = self._load_image_set_idx() 
		# a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
		# the image width and height

		## batch reader ##
		self._perm_idx = None
		self._cur_idx = 0
		# TODO(bichen): add a random seed as parameter
		self._shuffle_image_idx()

	def _load_image_set_idx(self):
		image_set_file = os.path.join(
			self._data_root_path, self._image_set_dir, self._image_set+'.txt')
		assert os.path.exists(image_set_file), \
			'File does not exist: {}'.format(image_set_file)

		with open(image_set_file) as f:
		  image_idx = [x.strip() for x in f.readlines()]
		return image_idx

	def _lidar_2d_path_at(self,idx):
		lidar_2d_path = os.path.join(self._lidar_2d_path, idx+'.npy')
		assert os.path.exists(lidar_2d_path),'File does not exist: {}'.format(lidar_2d_path)
		return lidar_2d_path

	def read_batch(self, shuffle=True):
		"""Read a batch of lidar data including labels. Data formated as numpy array
		of shape: height x width x {x, y, z, intensity, range, label}.
		Args:
		  shuffle: whether or not to shuffle the dataset
		Returns:
		  lidar_per_batch: LiDAR input. Shape: batch x height x width x 5.
		  lidar_mask_per_batch: LiDAR mask, 0 for missing data and 1 otherwise.
			Shape: batch x height x width x 1.
		  label_per_batch: point-wise labels. Shape: batch x height x width.
		  weight_per_batch: loss weights for different classes. Shape:
			batch x height x width
		"""
		mc = self.mc

		if shuffle:
			if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
				self._shuffle_image_idx()
			batch_idx = self._perm_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
			self._cur_idx += mc.BATCH_SIZE
		else:
			if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
				batch_idx = self._image_idx[self._cur_idx:] \
							+ self._image_idx[:self._cur_idx + mc.BATCH_SIZE - len(self._image_idx)]
				self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
			else:
				batch_idx = self._image_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
				self._cur_idx += mc.BATCH_SIZE

		lidar_per_batch = []
		lidar_mask_per_batch = []
		label_per_batch = []
		weight_per_batch = []

		for idx in batch_idx:
			# load data
			# loading from npy is 30x faster than loading from pickle
			record = np.load(self._lidar_2d_path_at(idx)).astype(np.float32, copy=False)

			if mc.DATA_AUGMENTATION:
				if mc.RANDOM_FLIPPING:
					if np.random.rand() > 0.5:
						# flip y
						record = record[:, ::-1, :]
						record[:, :, 1] *= -1

			lidar = record[:, :, :4]  # x, y, z, r
			lidar_mask = np.reshape(
				(lidar[:, :, 3] > 0),
				[mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
			)
			# normalize
			lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD
			label = record[:, :, 4]



			weight = np.zeros(label.shape)
			for l in range(mc.NUM_CLASS):
				weight[label == l] = mc.CLS_LOSS_WEIGHT[int(l)]

			# Append all the data
			lidar_per_batch.append(lidar)
			lidar_mask_per_batch.append(lidar_mask)
			label_per_batch.append(label)
			weight_per_batch.append(weight)

		return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
			   np.array(label_per_batch), np.array(weight_per_batch)
