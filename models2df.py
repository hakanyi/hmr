from PIL import Image
import pandas as pd
import os
import sys
import numpy as np
import shutil
import tensorflow as tf

import pdb

sys.path.append('./')
sys.path.append('../')
sys.path.append('../EIG_body_perception')

from config import Options
from demo import main

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def attend_crop_img(img, attend_size=350, save_path=None):
    """
    Cropping procedure: find bounding box, make bounding box quadratic and scale to fixed crop size
    """
    img_np = np.mean(np.array(img), axis=2).T
    xs, ys = np.where(img_np != img_np[0, 0])  # find non-background pixels
    x0_bb, x1_bb = min(xs), max(xs)  # extract bounding box + padding
    y0_bb, y1_bb = min(ys), max(ys)

    len = max(x1_bb - x0_bb, y1_bb - y0_bb)
    x_center, y_center = int((x1_bb + x0_bb) / 2), int((y1_bb + y0_bb) / 2)

    cropped_img = img.crop(
        (x_center - int(len / 2), y_center - int(len / 2), x_center + int(len / 2), y_center + int(len / 2)))
    attended_img = cropped_img.resize([attend_size, attend_size])

    if save_path is not None:
        make_path(os.path.dirname(save_path))
        attended_img.save(save_path)

    return attended_img


def center_crop_img(img, crop_size=350, save_path=None):
    width, height = img.size  # Get dimensions
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    cropped_img = img.crop((left, top, right, bottom))

    if save_path is not None:
        make_path(os.path.dirname(save_path))
        cropped_img.save(save_path)

    return cropped_img


def models2df(model_names, opt, save_dir=None):
    """
    Applies provided models to provided stimuli and outputs network activation
    data as a DataFrame for further processing.
    """
    activation_data = []
    columns = ["StimulusName", "ImageMode", "ModelName", "Region", "Activations"]

    for img_mode in opt.feeding_modes:
        stim_source, mode = str.split(img_mode, "-")
        img_paths = []
        if stim_source == "original":
            img_dir = opt.stimuli_dir
            img_paths = sorted(os.listdir(img_dir))
        elif stim_source == "rendered":
            if os.path.exists(opt.stimuli_dir+'-trunk-centric'):
                img_dir = opt.stimuli_dir+"-trunk-centric"
            else:
                img_dir = opt.stimuli_dir+"-trunk-centric-uncentered"
            img_paths = sorted(os.listdir(img_dir))
            img_paths = [path for path in img_paths if "-" not in path]
        else:
            print("Source in image mode {} is not implemented.".format(img_mode))

        # loop through all stimuli/images in this mode
        make_path('tmp')
        for img_path in img_paths:
            stimulus_name = str.split(os.path.basename(img_path), ".")[0]

            # loop through image modes (e.g. "raw" or "cropped")
            if mode == "raw":
                img_path = os.path.join(img_dir, img_path)
            elif mode == "cropped":
                img_path = os.path.join(img_dir+'-cropped', img_path)
            # elif mode == "attended":
            #     img = attend_crop_img(img, attend_size=opt.attend_size)
            else:
                print("Mode in image mode {} is not implemented.".format(mode))

            # loop through models
            for model_name in model_names:
                print('\nComputing output for model {} and image {}.\n'.format(model_name, stimulus_name))
                output = main(img_path, model_type=model_name)
                # loop through labels
                for model_region in opt.model_regions[model_name]:
                    out = output[model_region].flatten()
                    activation_data.append([stimulus_name, img_mode, model_name, model_region, out])

    return pd.DataFrame(activation_data, columns=columns)

opt = Options()

# load pre-trained models
model_names = ['ResNet50-ImageNet', 'ResNet50-HMR']
print("\t Model Data")

# load activations of pre-trained models applied to stimuli
model_data = models2df(model_names, opt)
make_path(opt.model_data_dir)
model_data.to_pickle(os.path.join(opt.model_data_dir, 'ResNets.pkl'))
