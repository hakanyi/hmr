from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.tf_smpl.batch_smpl import SMPL
from src.tf_smpl import projection as proj_util

import pdb
tf.reset_default_graph()

def visualize(proc_param, renderer, joints, verts, cam, img_size=(230,230)):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img_size)

    # Render results
    rend_img = renderer(vert_shifted, cam=cam_for_render, img_size=img_size)
    rend_img = renderer.rotated(vert_shifted, 90, cam=cam_for_render, img_size=img_size)

    fig = plt.figure(1)
    plt.imshow(rend_img)
    return fig

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
args = parser.parse_args()

# define some paths
smpl_model_path = '/gpfs/milgram/project/yildirim/hakan/hmr/models/neutral_smpl_with_cocoplus_reg.pkl'
smpl_face_path = '/gpfs/milgram/project/yildirim/hakan/hmr/src/tf_smpl/smpl_faces.npy'

# define SMPL model and renderer
smpl = SMPL(smpl_model_path)
renderer = vis_util.SMPLRenderer(face_path=smpl_face_path)

# define camera, position and shape parameters
cam = np.load('latents/cam'+str(args.num)+'.npy').flatten()
pose = np.load('latents/pose'+str(args.num)+'.npy')
shape = np.load('latents/shape'+str(args.num)+'.npy')
cam_tf = tf.Variable(cam)
pose = tf.Variable(pose)
shape = tf.Variable(shape)

# define pre-processing dict, no clue why they require it
proc_param = {'end_pt': np.array([339, 339]), 'img_size': 230, 'scale': 0.9739130434782609, 'start_pt': np.array([115, 115])}
# proc_param = {'end_pt': array([230, 230]), 'img_size': 230, 'scale': 0.9739130434782, 'start_pt': array([0, 0])}
# proc_param = {'end_pt': np.array([230, 230]), 'img_size': 230, 'scale': 1., 'start_pt': np.array([0, 0])}

# run SMPL and get 3D
verts, Js, _ = smpl(shape, pose, get_skin=True)
pred_kp = proj_util.batch_orth_proj_idrot(Js, cam_tf, name='proj_2d')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    verts, pred_kp = sess.run([verts, pred_kp])
# visualize the whole thing
fig = visualize(proc_param, renderer, pred_kp[0], verts[0], cam)
fig.savefig('render_results/'+str(args.num)+'.png')
