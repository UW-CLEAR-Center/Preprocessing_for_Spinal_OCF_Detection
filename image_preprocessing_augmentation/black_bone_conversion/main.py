from .robust_white_black_bone_detection import *
import shutil
import pandas as pd
import os
from sklearn.model_selection import KFold
import threading
import argparse
import tifffile as tiff

os.nice(5)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--num_threads', type=int, default=10)
args = parser.parse_args()

###############################
input_dir = args.input_dir
num_threads = args.num_threads
##############################

bit_depth = 16
vb_dir = os.path.join(input_dir, 'original_data')
output_dir = os.path.join(input_dir, 'black_bone_converted')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

MrOS_spine_dir = '/data/Spine_ML_Data/SOF_MrOS_Data/MrOSFinal'
annotation_dir = 'mros_annotation_files'
annotation_file = 'merged_annotation_file.csv'
annotation_path = os.path.join(annotation_dir, annotation_file)
annotation_df = pd.read_csv(annotation_path)
image_annotation_dict = spine_image_info_dict_builder(annotation_df)

def count_dir_depth(d):
    dir_depth = len(d.split('/'))
    if d[-1] == '/':
        dir_depth -= 1
    return dir_depth

def black_bone_converting_func(sdir, ddir, files):
    for vb_file in files:
        src_vb_path = os.path.join(sdir, vb_file)
        des_vb_path = os.path.join(ddir, vb_file)
        bone_gray = robust_write_black_bone_checker(vb_file, sdir, MrOS_spine_dir, image_annotation_dict, bit_depth=bit_depth)
        if bone_gray == 'white':
            shutil.copyfile(src_vb_path, des_vb_path)
        elif bone_gray == 'black':
            img = tiff.imread(src_vb_path)
            img = 2**bit_depth - 1 - img
            tiff.imsave(des_vb_path, img)

def splitting_files(files, num):
    files = np.array(files)
    kf = KFold(n_splits=num)
    k_folds_indices = kf.split(files)
    split_files = []
    for _, index in k_folds_indices:
        split_files.append(files[index])
    return split_files

def black_bone_converting_multithreading(sdir, ddir, files):
    print(ddir)
    os.makedirs(ddir)
    split_files = splitting_files(files, num=num_threads)
    threads = []
    for files_ in split_files:
        thread = threading.Thread(target=black_bone_converting_func, args=(sdir, ddir, files_))
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()

vb_dir_depth = count_dir_depth(vb_dir)
for root, dirs, files_ in os.walk(vb_dir):
    files = []
    for f in files_:
        if f.endswith('.tiff'):
            files.append(f)
    if len(files) == 0:
        continue
    root_layers = root.split('/')
    vb_sub_dir_layers = root_layers[vb_dir_depth:]
    vb_sub_dir = '/'.join(vb_sub_dir_layers)
    new_root = os.path.join(output_dir, vb_sub_dir)
    black_bone_converting_multithreading(root, new_root, files)
