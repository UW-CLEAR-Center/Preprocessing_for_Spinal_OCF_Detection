import argparse

from .dataset_splitting_functions import *
from .utils import *

os.nice(5)

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', type=int, default=10)
parser.add_argument('--train_val_vbs_output_dir', type=str)
parser.add_argument('--generating_new_testset', action='store_true')
parser.add_argument('--test_percent', type=float, default=0.15)
parser.add_argument('--test_vbs_output_dir', type=str)
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--validation_percent', type=float, default=0.1)
parser.add_argument('--normal_frac_ratio', type=float, default=1)
parser.add_argument('--existing_test_dir', type=str)
parser.add_argument('--resampling_trainset', action='store_true')
parser.add_argument('--existing_split_folds_dir', type=str)
args = parser.parse_args()

##############
test_percent = args.test_percent # the ratio of # of test instances to # of all cases
use_existing_test_set = not args.generating_new_testset # whether use a existing test set
test_vbs_output_dir = args.test_vbs_output_dir # the directory storing the vbs in the test set, must be set if use_existing_test_set = False.
existing_test_dir = args.existing_test_dir # the dir containing existing test set, must be set if use_existing_test_set == True

num_folds = args.num_folds # number of rounds in kfold cross validation
validation_percent = args.validation_percent # the ratio of # of validation instances to (# of validation + # of training instances); if num_folds > 1, this flag will be ignored
normal_frac_ratio = args.normal_frac_ratio # the ratio of normal to frac when downsampling the normal cases in training set
train_val_vbs_output_dir = args.train_val_vbs_output_dir # the dir storing the csv files generated when splitting k rounds of kfold cross validation
use_existing_split_folds = not args.resampling_trainset  # when downsampling, whether we want to use existing rounds that have been already split, or split from scratch
existing_split_folds_dir = args.existing_split_folds_dir # the directory containing existing round splitting schema, must be set if use_existing_split_folds == True
num_threads = args.num_threads
#############

class_lumping_dict = {
    0: 0, 1: 0, 2: 0, 2.5: 0,
    3: 1, 4: 1, 4.5: 1, 5: 1, 6: 1, 7: 1
}
# print(os.getcwdb())
annotation_dir = 'mros_annotation_files'
v1_vbs_dir = os.path.join(annotation_dir, 'V1_extracted_vbs_no_augmentation')
v2_vbs_dir = os.path.join(annotation_dir, 'V2_extracted_vbs_no_augmentation')
annotation_file = 'merged_annotation_file.csv'
annotation_path = os.path.join(annotation_dir, annotation_file)
annotation_df = pd.read_csv(annotation_path)
cleaned_annotation_df = annotation_df.loc[(annotation_df['XQSQ']!='N') & (annotation_df['XQSQ']!='A') & (annotation_df['XQSQ']!='M')]

cleaned_annotation_df = cleaned_annotation_df.astype({'XQSQ': 'float64'})
cleaned_annotation_df = df_add_lumped_class(cleaned_annotation_df, class_lumping_dict)
cleaned_annotation_df = df_add_vb_filenames(cleaned_annotation_df)
cleaned_annotation_df.reset_index(inplace=True, drop=True)

# sample test set
if not use_existing_test_set:
    test_df, train_val_df = sampling_testset(cleaned_annotation_df, percent=test_percent)
    test_df.reset_index(inplace=True, drop=True)
    copy_test_files(v1_vbs_dir, v2_vbs_dir, test_vbs_output_dir, test_df, train_val_df, num_threads=num_threads)
else:
    train_val_df = pd.read_csv(os.path.join(existing_test_dir, 'all_unsampled_train_val_instances.csv'))
train_val_df.reset_index(inplace=True, drop=True)

# generate train and valid set
train_dfs, val_dfs, raw_train_dfs = splitting_folds(
    train_val_df,
    num_folds=num_folds,
    normal_frac_ratio=normal_frac_ratio,
    use_existing_split_folds=use_existing_split_folds,
    existing_split_folds_dir=existing_split_folds_dir,
    validation_percent=validation_percent
)

# uncommented this later
copy_train_val_files(v1_vbs_dir, v2_vbs_dir, train_val_vbs_output_dir, train_dfs, val_dfs, raw_train_dfs, class_lumping_dict, num_threads=num_threads)

