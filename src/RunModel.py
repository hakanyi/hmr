""" Evaluates a trained model using placeholders. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import exists

from .tf_smpl import projection as proj_util
from .tf_smpl.batch_smpl import SMPL
from .models import get_encoder_fn_separate

from collections import OrderedDict

class RunModel(object):
    def __init__(self, config, second_load_path=None, sess=None):
        """
        Args:
          config
        """
        self.config = config
        self.load_path = config.load_path
        self.second_load_path = second_load_path

        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You need to specify `load_path` to load a pretrained model"
            )
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path

        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type
        self.joint_type = config.joint_type
        # Camera
        self.num_cam = 3
        self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        print(self.joint_type)
        self.smpl = SMPL(self.smpl_model_path, joint_type=self.joint_type)

        # self.theta0_pl = tf.placeholder_with_default(
        #     self.load_mean_param(), shape=[self.batch_size, self.total_params], name='theta0')
        # self.theta0_pl = tf.placeholder(tf.float32, shape=[None, self.total_params], name='theta0')

        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # Load data.
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50')
        # self.saver_resnet = tf.train.Saver(variables)
        variables = tf.contrib.framework.get_variables('resnet_v2_50')
        self.saver_resnet = tf.train.Saver(variables)
        self.saver = tf.train.Saver()
        self.prepare()


    def build_test_model_ief(self):
        # Load mean value
        self.mean_var = tf.Variable(tf.zeros((1, self.total_params)), name="mean_param", dtype=tf.float32)

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)
        # Extract image features.
        self.img_feat, self.E_var, self.img_end_points = img_enc_fn(self.images_pl,
                                                                   is_training=False,
                                                                   reuse=False)

        # print(self.img_end_points)
        # print(self.img_feat)
        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_poses = []
        self.all_shapes = []
        self.all_Js = []
        self.final_thetas = []
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
            else:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            # print(cams.shape)
            # print(poses.shape)
            # print(shapes.shape)
            verts, Js, _ = self.smpl(shapes, poses, get_skin=True)

            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_poses.append(poses)
            self.all_shapes.append(shapes)
            self.all_Js.append(Js)
            # save each theta.
            self.final_thetas.append(theta_here)
            # Finally)update to end iteration.
            theta_prev = theta_here


    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)
        if self.second_load_path is not None:
            self.saver_resnet.restore(self.sess, self.second_load_path)
        self.mean_value = self.sess.run(self.mean_var)

    def predict(self, images, get_theta=False):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        results = self.predict_dict(images)
        if get_theta:
            return results['joints'], results['verts'], results['cams'], results[
                'joints3d'], results['theta'], results['layer_activations']
        else:
            return results['joints'], results['verts'], results['cams'], results[
                'joints3d']

    def predict_dict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        feed_dict = {
            self.images_pl: images,
            # self.theta0_pl: self.mean_var,
        }
        fetch_dict = {
            'joints': self.all_kps[-1],
            'verts': self.all_verts[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Js[-1],
            'theta': self.final_thetas[-1],
            'layer_activations': OrderedDict(
            {'B1U1' : self.img_end_points['resnet_v2_50/block1/unit_1/bottleneck_v2'],
             'B1U2' : self.img_end_points['resnet_v2_50/block1/unit_2/bottleneck_v2'],
             'B1U3' : self.img_end_points['resnet_v2_50/block1/unit_3/bottleneck_v2'],
             'B2U1' : self.img_end_points['resnet_v2_50/block2/unit_1/bottleneck_v2'],
             'B2U2' : self.img_end_points['resnet_v2_50/block2/unit_2/bottleneck_v2'],
             'B2U3' : self.img_end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'],
             'B2U4' : self.img_end_points['resnet_v2_50/block2/unit_4/bottleneck_v2'],
             'B3U1' : self.img_end_points['resnet_v2_50/block3/unit_1/bottleneck_v2'],
             'B3U2' : self.img_end_points['resnet_v2_50/block3/unit_2/bottleneck_v2'],
             'B3U3' : self.img_end_points['resnet_v2_50/block3/unit_3/bottleneck_v2'],
             'B3U4' : self.img_end_points['resnet_v2_50/block3/unit_4/bottleneck_v2'],
             'B3U5' : self.img_end_points['resnet_v2_50/block3/unit_5/bottleneck_v2'],
             'B3U6' : self.img_end_points['resnet_v2_50/block3/unit_6/bottleneck_v2'],
             'B4U1' : self.img_end_points['resnet_v2_50/block4/unit_1/bottleneck_v2'],
             'B4U2' : self.img_end_points['resnet_v2_50/block4/unit_2/bottleneck_v2'],
             'B4U3' : self.img_end_points['resnet_v2_50/block4/unit_3/bottleneck_v2'],
             'GPOOL': self.img_feat,
             'joint': self.all_kps[-1],
             'vert': self.all_verts[-1],
             'cam': self.all_cams[-1],
             'joint3D': self.all_Js[-1],
             'pose' : self.all_poses[-1],
             'shape' : self.all_shapes[-1]})
        }

        results = self.sess.run(fetch_dict, feed_dict)

        # Return joints in original image space.
        joints = results['joints']
        results['joints'] = ((joints + 1) * 0.5) * self.img_size

        return results
