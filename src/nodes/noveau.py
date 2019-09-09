#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Segmentation ROS node on LiDAR point cloud using SqueezeSeg Neural Network
"""

import sys
import os.path
import numpy as np
from PIL import Image

# lib_path = os.path.abspath(os.path.join('..'))
# print lib_path
# sys.path.append(lib_path)
import tensorflow as tf



sys.path.append("..")
from config import *
from nets import SqueezeSeg
from utils.util import *
from utils.clock import Clock

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

pick_color = {
    0: (163, 162, 158) ,     # dark grey road
    1 :  (255,0,0),        # red
    2 :  (29, 61, 0) ,      # dark green
    3:  (64, 148, 4),      # green
    4:  (165, 115, 14),    # brown
    5  : (255, 191, 64),    # light brown
    6 : (107, 46, 5),      # cement 
    7 : (214, 165, 133),   # wall 
    8 : (222, 174, 90),     # light brown
    9 : (215, 88, 232),    # pink,
    10 : (89, 5, 5),  #  dark red
    11: (0,0,0)
}    


class SegmentNode():
    """LiDAR point cloud segment ros node"""

    def __init__(self,
                 sub_topic, pub_topic, pub_feature_map_topic, pub_label_map_topic,
                 FLAGS):
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        self._mc = kitti_squeezeSeg_config()
        self._mc.LOAD_PRETRAINED_MODEL = False
        # TODO(bichen): fix this hard-coded batch size.
        self._mc.BATCH_SIZE = 1
        self._model = SqueezeSeg(self._mc)
        self._saver = tf.train.Saver(self._model.model_params)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)
        self.run()

    def run(self):


        np_p = np.load("/home/ubi-comp/yara/AirSim/PythonClient/multirotor/scene1_90fov_color.npy")
        # perform fov filter by using hv_in_range

        np_p[:,2] = np_p[:,2] + 3
        # np_p =  np_p[np_p[:,2] >= 0]
        cond = self.hv_in_range(x=np_p[:, 0],
                                y=np_p[:, 1],
                                z=np_p[:, 2],
                                fov=[-45, 45])


        np_p_ranged = np_p[cond]

        # get depth map
        lidar = self.pto_depth_map(velo_points=np_p_ranged, C=5, H= self._mc.ZENITH_LEVEL)
        lidar_f = lidar.astype(np.float32)

        # to perform prediction
        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [self._mc.ZENITH_LEVEL, self._mc.AZIMUTH_LEVEL, 1]
        )
        lidar_f = (lidar_f - self._mc.INPUT_MEAN) / self._mc.INPUT_STD
        pred_cls = self._session.run(
            self._model.pred_cls,
            feed_dict={
                self._model.lidar_input: [lidar_f],
                self._model.keep_prob: 1.0,
                self._model.lidar_mask: [lidar_mask]
            }
        )
        label = pred_cls[0]
        print(label)
        # # generated depth map from LiDAR data
        depth_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))

        label_3d = np.zeros((label.shape[0], label.shape[1], 3))
        label_3d[np.where(label==0)] = [1., 1., 1.]
        label_3d[np.where(label==1)] = [0., 1., 0.]
        label_3d[np.where(label==2)] = [1., 1., 0.]
        label_3d[np.where(label==3)] = [0., 1., 1.]


        ## point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)


        # save the data
        file_name = "hello_fov90"
        # generated depth map from LiDAR data
        depth_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
        # classified depth map with label
        label_map = Image.fromarray(
            (255 * visualize_seg(pred_cls, self._mc)[0]).astype(np.uint8))
        blend_map = Image.blend(
            depth_map.convert('RGBA'),
            label_map.convert('RGBA'),
            alpha=0.4
        )
        # save classified depth map image with label
        
        blend_map.save( file_name + '.png')
        # export input img
        test = depth_map.convert('RGBA')
        test.save(
        os.path.join('input_' + file_name + '.png'))

        colors = [ pick_color[int(num)] for num in lidar[:, :, 3].flatten() ]




        img = Image.new('RGB',(512,64))
        img.putdata(colors)
        img.save('somefile.png')

    def hv_in_range(self, x, y, z, fov, fov_type='h'):
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

    def pto_depth_map(self, velo_points,
                      H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
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

        x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
        #print("z", 255 * _normalize(z))
        d = np.sqrt(x ** 2 + y ** 2 + z**2)
        r = np.sqrt(x ** 2 + y ** 2)
        d[d==0] = 0.000001
        r[r==0] = 0.000001
        # phi = np.radians(45.) - np.arcsin(y/r)
        phi = np.arcsin(y/r) + np.radians(45.) 
        phi_ = (phi/dphi).astype(int)
        phi_[phi_<0] = 0
        phi_[phi_>=512] = 511

        # print(np.min(phi_))
        # print(np.max(phi_))
        #
        # print z
        # print np.radians(2.)
        # print np.arcsin(z/d)
        # theta = np.radians(2.) - np.arcsin(z/d)
        theta = np.arcsin(z/d)
        # print theta
        theta_ = (theta/dtheta).astype(int)
        # print theta_
        theta_[theta_<0] = 0
        theta_[theta_>=H] = H-1
        #print theta,phi,theta_.shape,phi_.shape
        # print(np.min((phi/dphi)),np.max((phi/dphi)))
        #np.savetxt('./dump/'+'phi'+"dump.txt",(phi_).astype(np.float32), fmt="%f")
        #np.savetxt('./dump/'+'phi_'+"dump.txt",(phi/dphi).astype(np.float32), fmt="%f")
        # print(np.min(theta_))
        # print(np.max(theta_))

        depth_map = np.zeros((H, W, C))
        # 5 channels according to paper
        if C == 5:
            depth_map[theta_, phi_, 0] = x
            depth_map[theta_, phi_, 1] = y
            depth_map[theta_, phi_, 2] = z
            depth_map[theta_, phi_, 3] = i
            depth_map[theta_, phi_, 4] = d
        else:
            depth_map[theta_, phi_, 0] = i
        return depth_map