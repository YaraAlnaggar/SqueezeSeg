# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from .config import base_model_config

def NH_airsim_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('NH_airsim')

  # mc.CLASSES            = ['ignore', 'building', 'sign', 'tree','car', 'road']
  mc.CLASSES            = ['ignore','tree','car']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  # mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  10.0, 10.0, 10.0, 1/15.0 ]) #check
  mc.CLS_LOSS_WEIGHT    = np.array([1/20.0, 5.0,  10.0 ])

  # mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
  #                                   [ 0.12,  0.56,  0.37],
  #                                   [ 0.66,  0.55,  0.71],
  #                                   [ 0.40,  0.72,  0.88],
  #                                   [ 0.58,  0.30,  0.50],
  #                                   [ 0.70,  0.10,  0.60],
  #                                   [ 0.35,  0.20,  0.40]
  #                                   ])
  
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37],
                                    [ 0.66,  0.55,  0.71]
                                    ])


  mc.INPUT_CHANNEL_SIZE = 4
  mc.BATCH_SIZE         = 32
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64
# ??????
  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  # mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6, .6, .9])
  mc.BILATERAL_THETA_A  = np.array([.9, .6, .6])
  # mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01, .01, .015])
  mc.BILATERAL_THETA_R  = np.array([.015, .01, .01]) 
  mc.BI_FILTER_COEF     = 0.1
  # mc.ANG_THETA_A        = np.array([.9, .9, .6, .6, .6, .9])
  mc.ANG_THETA_A        = np.array([.9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02
# ???????

  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         =  np.array([[[7.904, -0.016, -1.028, 8.979]]])
  mc.INPUT_STD          =  np.array( [[[5.93, 3.749, 0.37, 5.944]]] )

  return mc
