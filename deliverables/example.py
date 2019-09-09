import tensorflow as tf
from config import *
from encoder import SqueezeSeg
from utils.util import *



config = kitti_squeezeSeg_config()
model = SqueezeSeg(config)
session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# reading point cloud


# generate projection



# Inference

pred_cls = session.run(
    model.pred_cls,
    feed_dict={
        model.lidar_input: [lidar_f],
        model.keep_prob: 1.0,
        model.lidar_mask: [lidar_mask]
    }
)
