from easydict import EasyDict
import os

cfg = EasyDict()

# ext/superpoint.py
cfg.weights_path = '/home/yskim/projects/optival_vo_v2/checkpoint/superpoint_v6_from_tf.pth'
cfg.max_keypoints = 500
cfg.device = 'cuda'

cfg.feature_type = 'superpoint'

cfg.match_threshold = 1

cfg.vp_config_path = 'ext/neurvps/config/tmm17.yaml'
cfg.vp_ckpt_path = '/home/yskim/projects/optival_vo_v2/checkpoint/checkpoint_latest.pth.tar'

cfg.odometry_home = '/home/yskim/projects/vo-labs/data/kitti_odometry/'
cfg.proj_home = '/home/yskim/projects/optival_vo_v2/'
cfg.model = 'VO'

cfg.logdir = os.path.join(cfg.proj_home, 'checkpoint', cfg.model)

cfg.color_subdir = 'datasets/sequences/'
cfg.calib_subdir = 'datasets/sequences/'
cfg.poses_subdir = 'poses/'


cfg.traintxt = 'kitti-train.txt'
cfg.valtxt = 'kitti-val.txt'

cfg.trainsequencelist = ['00']
cfg.valsequencelist = ['09', '10']


