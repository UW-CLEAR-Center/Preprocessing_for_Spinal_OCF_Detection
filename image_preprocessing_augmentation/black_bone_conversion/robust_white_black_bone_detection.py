from .image_preprocessing import *
from .white_black_bone_detection import *
import tifffile as tiff
import os
import SimpleITK as sitk


def spine_image_info_dict_builder(annotation_df):
    image_annotation_dict = {'V1':{}, 'V2':{}}
    for index, line in annotation_df.iterrows():
        visit = 'V' + str(line['VISIT'])
        image_name = line['ID']
        if image_name not in image_annotation_dict[visit]:
            image_annotation_dict[visit][image_name] = {}
            comparator = np.inf
            which_image = 'L'
            if visit == 'V2' and image_name == 'PO7267':
                which_image = 'T'
        vb_name = line['XQVERT']
        image_annotation_dict[visit][image_name][vb_name] = {}
        if float(line['XQQMY1']) > comparator:
            which_image = 'T'
        comparator = float(line['XQQMY1'])
        corners = np.array([[float(line['XQQMX1']), float(line['XQQMY1'])],
                                   [float(line['XQQMX3']), float(line['XQQMY3'])],
                                   [float(line['XQQMX4']), float(line['XQQMY4'])],
                                   [float(line['XQQMX6']), float(line['XQQMY6'])]])
        image_annotation_dict[visit][image_name][vb_name]['corners'] = corners
        image_annotation_dict[visit][image_name][vb_name]['which_image'] = which_image
            
    return image_annotation_dict


def check_LT_image(vb_level, case, visit, image_annotation_dict):
    """
    check whether the vb is in the L image or the T image
    """
    which_image = image_annotation_dict[visit][case][vb_level]['which_image']
    return which_image


def original_spine_search(vb_filename, search_dir, image_annotation_dict):
    """
    search the spine image file containing a certain vb
    """
    image_annotation_dict = image_annotation_dict.copy()
    vb_code = vb_filename.split('.')[0]
    vb_code_parts = vb_code.split('_')
    visit = vb_code_parts[1]
    case = vb_code_parts[2]
    case_type = case[:2]
    vb_level = vb_code_parts[3]
    which_image = check_LT_image(vb_level, case, visit, image_annotation_dict)
    spine_file = 'MR' + case + visit + which_image + '.dcm'
    spine_path = os.path.join(search_dir, case_type, case, spine_file)
    return spine_path


def extract_spine_all_vbs(vb_filename, search_dir, image_annotation_dict, bit_depth=16, rotate=0, expand_w=1, expand_h=1, square=True):
    image_annotation_dict = image_annotation_dict.copy()
    spine_path = original_spine_search(vb_filename, search_dir, image_annotation_dict)
    img = sitk.ReadImage(spine_path)[:,:,0]
    npa = sitk.GetArrayViewFromImage(img)
    h, w = npa.shape
    npa = np.flip(npa, axis=1)
    vb_code = vb_filename.split('.')[0]
    vb_code_parts = vb_code.split('_')
    visit = vb_code_parts[1]
    case = vb_code_parts[2]
    vb_level = vb_code_parts[3]
    which_image = check_LT_image(vb_level, case, visit, image_annotation_dict)
    npa = regulate_bit_depth(npa, bit_depth)
    annotation = image_annotation_dict[visit][case]
    corners_dict = {}
    for vb_name in annotation:
        flipped_corners = annotation[vb_name]['corners']
        if annotation[vb_name]['which_image'] == which_image:
            corners = np.array([w-flipped_corners[:,0],flipped_corners[:,1]]).T
            corners_dict[vb_name] = corners
    vb_pixel_dict = extract_spine_vbs(npa, corners_dict, rotate, expand_w, expand_h, square=square)
    return vb_pixel_dict


def robust_write_black_bone_checker(vb_filename, vb_dir, search_dir, image_annotation_dict, bit_depth=16, rotate=0, expand_w=1, expand_h=1, square=True):
    try:
        vb_pixel_dict = extract_spine_all_vbs(vb_filename, search_dir, image_annotation_dict, bit_depth=bit_depth, rotate=rotate, expand_w=expand_w, expand_h=expand_h, square=square)
    except:
        img = tiff.imread(os.path.join(vb_dir, vb_filename))
        rst, _, _ = black_white_bone_checker(img)
        return rst

    results = []
    for vb in vb_pixel_dict:
        vb_pixel = vb_pixel_dict[vb]
        rst, _, _ = black_white_bone_checker(vb_pixel)
        # print(vb, rst)
        results.append(rst)
    black_count = results.count('black')
    white_count = results.count('white')
    if black_count == white_count:
        img = tiff.imread(os.path.join(vb_dir, vb_filename))
        rst, _, _ = black_white_bone_checker(img)
        return rst
    if black_count > white_count:
        return 'black'
    return 'white'


