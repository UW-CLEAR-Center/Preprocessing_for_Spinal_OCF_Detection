import os
from os.path import exists, join
import shutil
import SimpleITK as sitk
import tifffile as tiff
import csv
import json
from .image_preprocessing import *
import numpy as np
from .image_normalization import *
import sys
import pandas as pd
import threading
from sklearn.model_selection import KFold
sys.path.append('black_bone_conversion')
from ..black_bone_conversion.robust_white_black_bone_detection import *
import argparse

os.nice(5)


""" This file is used for image augmentation """
parser = argparse.ArgumentParser()
parser.add_argument('--needed_vbs_dir', type=str)
parser.add_argument('--black_bone_converting', action='store_true')
parser.add_argument('--num_threads', type=int, default=10)
parser.add_argument('--num_augmentations', type=int, default=10)
args = parser.parse_args()

###################### parameter ###########################
needed_vbs_dir = args.needed_vbs_dir # the dir used for reference
black_bone_converting = args.black_bone_converting
num_threads = args.num_threads
num_augmentations = args.num_augmentations # number of augmented images
image_normalizing = False
############################################################

if black_bone_converting and image_normalizing:
    save_dir = os.path.join(needed_vbs_dir, 'black_bone_converted_image_norm_augmented') # vb saved dir
elif black_bone_converting:
    save_dir = os.path.join(needed_vbs_dir, 'black_bone_converted_augmented') # vb saved dir
elif image_normalizing:
    save_dir = os.path.join(needed_vbs_dir, 'image_norm_augmented') # vb saved dir
else:
    save_dir = os.path.join(needed_vbs_dir, 'augmented') # vb saved dir

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

image_dir = '/data/Spine_ML_Data/SOF_MrOS_Data/MrOSFinal/'# the directory containing all images
bit_depth = 16 # the bit depth we want the image to be
default_rotate = 0 # the basic degree we want the vb to rotate
default_expand_w = 1 # basic expanding of the width of the tightest bounding box
default_expand_h = default_expand_w # basic expanding the height of the tightest bounding box
square = True

# annotations of all images
annotation_dir = 'mros_annotation_files'
annotation_file = 'merged_annotation_file.csv'
annotation_path = os.path.join(annotation_dir, annotation_file)
annotation_df = pd.read_csv(annotation_path)

# image preprocessing related
image_norm_low = 0.05
image_norm_high = 0.95

# data augmentation related
rotate_range = (-5, 5) # the range of rotation degree
scale_x_range = (0.9, 1.1) # the range of the vb scaling along x-axis
scale_y_range = scale_x_range # the range of the vb scaling along y-axis, as we keep the vb patch as a square, this attribute won't have effects
translate_x_range = (-0.05, 0.05) # the range of the vb translation in a bounding box along x-axis
translate_y_range = translate_x_range # the range of the vb translation in a bounding box along y-axi
deformation_coeff_range = (0, 0) # the range of the vb deformation
brightness_coeff_range = (0.4, 0.6) # the range of brightness adjustment
contrast_coeff_range = (4, 8) # the range of contrast adjustment
blur_coeff_range = (0, 0) # the range of Gaussian blurring
sharpen_coeff_range = (0, 0) # the range of sharpening
noise_coeff_range = (0, 8e-4 * (2**bit_depth - 1)) # the range of noise adding
gray_inverse_prob = 0 # the probability that the augmented patch will invert grayscale
horizontal_flip_prob = 0 # the probability that the augmented patch will be flipped horizontally
vertical_flip_prob = 0 # the probability that the augmented patch will be flipped vertically

def _vb_scale_to_bb_expand(vb_scaling, default_expanding):
    return 1 / vb_scaling * (default_expanding + 1) - 1
def _vb_scale_tuple_to_bb_expand_tuple(vb_scalings, default_expanding):
    return (_vb_scale_to_bb_expand(vb_scalings[1], default_expanding),
            _vb_scale_to_bb_expand(vb_scalings[0], default_expanding))
expand_w_range = _vb_scale_tuple_to_bb_expand_tuple(scale_x_range, default_expand_w)
expand_h_range = _vb_scale_tuple_to_bb_expand_tuple(scale_y_range, default_expand_h)

# store all of papameters into a json file
param_dict = {
    'num_augmentations': num_augmentations, 
    'black_bone_converting': black_bone_converting,
    'default_rotate': default_rotate,
    'default_expand_w': default_expand_w,
    'default_expand_h': default_expand_h,
    'square': square,
    'image_normalizing': image_normalizing,
    'image_norm_low': image_norm_low,
    'image_norm_high': image_norm_high,
    'rotate_range': rotate_range,
    'scale_x_range': scale_x_range,
    'scale_y_range': scale_y_range,
    'translate_x_range': translate_x_range,
    'translate_y_range': translate_y_range,
    'deformation_coeff_range': deformation_coeff_range,
    'brightness_coeff_range': brightness_coeff_range,
    'contrast_coeff_range': contrast_coeff_range,
    'blur_coeff_range': blur_coeff_range,
    'sharpen_coeff_range': sharpen_coeff_range,
    'noise_coeff_range': noise_coeff_range,
    'gray_inverse_prob': gray_inverse_prob,
    'horizontal_flip_prob': horizontal_flip_prob,
    'vertical_flip_prob': vertical_flip_prob,
}
param_output_path = os.path.join(save_dir, 'params.json')
with open(param_output_path, 'w') as f:
    json.dump(param_dict, f)

"""
read the csv files
"""
image_annotation_dict_for_black_bone_dectection = spine_image_info_dict_builder(annotation_df)
image_annotation_dict = {'1':image_annotation_dict_for_black_bone_dectection['V1'],
                         '2':image_annotation_dict_for_black_bone_dectection['V2']}

"""
get all vbs needed augmentation
"""
# the filenames of the targeted vbs
new_dirs = []
needed_vb_info_dict = {}
train_files = set()
needed_vbs_subdir = os.path.join(needed_vbs_dir, 'original_data')
for root, dirs, files in os.walk(needed_vbs_subdir):
    if root.split('/')[-2] == 'train':
        for f in files:
            if f in train_files:
                continue
            train_files.add(f)
            vb = f.split('.')[0]
            info = vb.split('_')
            label = int(info[0])
            version = int(info[1][1])
            case = info[2]
            annatomic_level = info[3]
            L_input_image_path = join(image_dir, case[0:2], case, 'MR' + case + 'V' + str(version) + 'L.dcm')
            T_input_image_path = join(image_dir, case[0:2], case, 'MR' + case + 'V' + str(version) + 'T.dcm')
            needed_vb_info_dict[vb] = {}
            needed_vb_info_dict[vb]['vb_filename'] = f
            needed_vb_info_dict[vb]['vb_dir'] = root
            needed_vb_info_dict[vb]['output_dir'] = join(save_dir, str(label), vb)
            needed_vb_info_dict[vb]['label'] = label
            needed_vb_info_dict[vb]['version'] = version
            needed_vb_info_dict[vb]['case'] = case
            needed_vb_info_dict[vb]['vb'] = annatomic_level
            if exists(L_input_image_path):
                needed_vb_info_dict[vb]['L_input_image_path'] = L_input_image_path
            if exists(T_input_image_path):
                needed_vb_info_dict[vb]['T_input_image_path'] = T_input_image_path
# print(needed_vb_info_dict)

def split_task(task_dict, num):
    num_cases = len(task_dict)
    kf = KFold(n_splits=num)
    k_folds_indices = kf.split(range(num_cases))
    num_subcases = []
    for _, index in k_folds_indices:
        num_subcases.append(len(index))
    subtasks = []
    co = 0
    i = 0
    subtask_dict = {}
    for key in task_dict:
        if co >= num_subcases[i]:
            subtasks.append(subtask_dict)
            co = 0
            i += 1
            subtask_dict = {}
        subtask_dict[key] = task_dict[key]
        co += 1
    subtasks.append(subtask_dict)
    return subtasks

def data_augmentation_func(needed_vb_info_dict, thread_seq_num):
    num_all_images = len(needed_vb_info_dict)
    for kkk, vb_code in enumerate(needed_vb_info_dict):
        print('thread {}: {} out of {} starts: {}'.format(thread_seq_num, kkk, num_all_images, vb_code))
        vb_info_dict = needed_vb_info_dict[vb_code]
        image_id = vb_info_dict['case']
        version = str(vb_info_dict['version'])

        """
        read pairs of L and T images
        """
        if 'L_input_image_path' in vb_info_dict:
            image_file = vb_info_dict['L_input_image_path']
            imgL = sitk.ReadImage(image_file)[:,:,0]
            npaL = sitk.GetArrayViewFromImage(imgL)
            hL, wL = npaL.shape
            npaL = np.flip(npaL, axis=1)

        if 'T_input_image_path' in vb_info_dict:
            image_file = vb_info_dict['T_input_image_path']
            imgT = sitk.ReadImage(image_file)[:,:,0]
            npaT = sitk.GetArrayViewFromImage(imgT)
            hT, wT = npaT.shape
            npaT = np.flip(npaT, axis=1)
            # a special case which should be flipped vertically
            if image_id == 'PO6538' and version == '1':
                npaT = np.flip(npaT, axis=0)

        """
        make images to the same bit_depth
        """
        npaL = regulate_bit_depth(npaL, bit_depth)
        npaT = regulate_bit_depth(npaT, bit_depth)

        """
        flip the annotation points and check which image contain such annotations
        """
        if image_id not in image_annotation_dict[version]:
            continue
        annotation = image_annotation_dict[version][image_id]
        corners_dict = {'L':{}, 'T':{}}
        for vb_name in annotation:
            flipped_corners = annotation[vb_name]['corners']
            if annotation[vb_name]['which_image'] == 'L':
                corners = np.array([wL-flipped_corners[:,0],flipped_corners[:,1]]).T
                corners_dict['L'][vb_name] = corners
            else:
                corners = np.array([wT-flipped_corners[:,0],flipped_corners[:,1]]).T
                corners_dict['T'][vb_name] = corners

        """
        extract the vb
        """
        Lvb_pixel_dict = extract_spine_vbs(npaL, corners_dict['L'], default_rotate, default_expand_w, default_expand_h, square=square)
        Tvb_pixel_dict = extract_spine_vbs(npaT, corners_dict['T'], default_rotate, default_expand_w, default_expand_h, square=square)
        vb = vb_info_dict['vb']
        if vb in Lvb_pixel_dict:
            vb_pixel = Lvb_pixel_dict[vb]
        elif vb in Tvb_pixel_dict:
            vb_pixel = Tvb_pixel_dict[vb]

        """
        image preprocessing on the origianl image
        """
        vb_file = vb_info_dict['vb_filename']
        vb_dir = vb_info_dict['vb_dir']
        if black_bone_converting:
            bone_gray = robust_write_black_bone_checker(vb_file, vb_dir, image_dir, image_annotation_dict_for_black_bone_dectection)
            if bone_gray == 'black':
                vb_pixel = 2**bit_depth - 1 - vb_pixel
        else:
            bone_gray = None
        if image_normalizing:
            vb_pixel = my_image_normalize(vb_pixel, bit_depth, image_norm_low, image_norm_high)

        """
        output the images
        """
        current_save_dir = vb_info_dict['output_dir']
        if not exists(current_save_dir):
            os.makedirs(current_save_dir)
        output_vb_image_name = vb_code + '_original.tiff'
        save_path = join(current_save_dir, output_vb_image_name)
        tiff.imsave(save_path, vb_pixel)

        """
        do brightness adjustment, contrast adjustment, gaussian blurring/sharpening, and gaussian noise adding to on vb patch
        """
        Lvb_augment_dict = affine_vb_augmentation(npaL, num_augmentations, corners_dict['L'], rotate_range, expand_w_range, expand_h_range, translate_x_range, translate_y_range, square=square)
        Tvb_augment_dict = affine_vb_augmentation(npaT, num_augmentations, corners_dict['T'], rotate_range, expand_w_range, expand_h_range, translate_x_range, translate_y_range, square=square)
        if vb in Lvb_augment_dict:
            vb_augment = Lvb_augment_dict[vb]
        elif vb in Tvb_augment_dict:
            vb_augment = Tvb_augment_dict[vb]

        aug_vbs = vb_patch_augmentation(vb_augment,
                                        deformation_coeff_range,
                                        brightness_coeff_range,
                                          contrast_coeff_range,
                                          blur_coeff_range,
                                          sharpen_coeff_range,
                                          noise_coeff_range,
                                          bit_depth=bit_depth,
                                        gray_inverse_prob=gray_inverse_prob,
                                        horizontal_flip_prob=horizontal_flip_prob,
                                        vertical_flip_prob=vertical_flip_prob,
                                        black_bone_converting=black_bone_converting,
                                        bone_gray=bone_gray,
                                        image_normalizing=image_normalizing,
                                        image_norm_low=image_norm_low,
                                        image_norm_high=image_norm_high
                                       )
        for jj, aug in enumerate(aug_vbs):
            output_vb_image_name = vb_code + '_augmentation' + str(jj) + '.tiff'
            save_path = join(current_save_dir, output_vb_image_name)
            tiff.imsave(save_path, aug)

# data augmentation via multi-threads
needed_vb_info_subdicts_list = split_task(needed_vb_info_dict, num=num_threads)
threads = []
for i, subdict in enumerate(needed_vb_info_subdicts_list):
    thread = threading.Thread(target=data_augmentation_func, args=(subdict, i))
    threads.append(thread)
    thread.start()
for t in threads:
    t.join()


