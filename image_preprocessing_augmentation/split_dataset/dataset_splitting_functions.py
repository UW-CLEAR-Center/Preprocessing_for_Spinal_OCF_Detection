import random
import pandas as pd

from .utils import *

"""
This file defines functions for splitting the dataset into train, validation and test.
"""


def sampling_testset(cleaned_annotation_df, percent):
    """
    sample the test set
    """
    all_patients = list(set(cleaned_annotation_df['ID']))
    num_all_patients = len(all_patients)
    num_test_patients = int(round(num_all_patients * percent))
    patients_in_testset = random.sample(all_patients, num_test_patients)
    sampled_df = cleaned_annotation_df[cleaned_annotation_df['ID'].isin(patients_in_testset)].reset_index(drop=True)
    remained_df = cleaned_annotation_df.merge(sampled_df, how = 'outer' ,indicator=True) \
            .query('_merge=="left_only"').drop(columns=['_merge']).reset_index(drop=True)
    return sampled_df, remained_df


def copy_test_files(v1_input_dir, v2_input_dir, output_dir, test_df, train_val_df, num_threads=10):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    data_subdir = os.path.join(output_dir, 'original_data')
    os.makedirs(data_subdir)
    copy_files_multithreading(v1_input_dir, v2_input_dir, data_subdir, test_df, num_threads=num_threads)
    test_df.to_csv(os.path.join(output_dir, 'test_instances.csv'), index=False)
    train_val_df.to_csv(os.path.join(output_dir, 'all_unsampled_train_val_instances.csv'), index=False)


def splitting_folds_from_scratch(df, num_folds, validation_percent=None):
    """
    given that there are no csv files containing unsampled_train instances and valid instances,
    split the data into several rounds for kfold cross validation
    if num_folds > 1, then validation_percent argument will be ignored
    """
    val_dfs = []
    raw_train_dfs = []
    # TODO write the code of num_folds != 1 later, if needed
    if num_folds != 1:
        pass
        # kf = KFold(n_splits=num_folds, shuffle=True)
        # k_folds_indices = kf.split(df.index)
        # for raw_train_index, val_index in k_folds_indices:
        #     val_df = df.iloc[val_index]
        #     raw_train_df = df.iloc[raw_train_index]
        #     val_df.reset_index(inplace=True, drop=True)
        #     val_dfs.append(val_df)
        #     raw_train_df = delete_same_vb_diff_visit(raw_train_df, val_df)
        #     raw_train_df.reset_index(inplace=True, drop=True)
        #     raw_train_dfs.append(raw_train_df)
    else:
        all_patients = list(set(df['ID']))
        num_all_patients = len(all_patients)
        num_val_patients = int(round(num_all_patients * validation_percent))
        patients_in_valstset = random.sample(all_patients, num_val_patients)
        val_df = df[df['ID'].isin(patients_in_valstset)].reset_index(drop=True)
        raw_train_df = df.merge(val_df, how = 'outer' ,indicator=True) \
            .query('_merge=="left_only"').drop('_merge', axis=1)
        print(raw_train_df[raw_train_df['LUMPED_LABEL']==1].shape[0])
        val_dfs.append(val_df)
        raw_train_dfs.append(raw_train_df)
    return raw_train_dfs, val_dfs


def read_write_folds(mode, dir_, train_dfs=None, val_dfs=None, raw_train_dfs=None, num_folds=None):
    """
    read or write csv files contains train, valid, unsampled train instances for each training round
    """
    train_subdir = os.path.join(dir_, 'summaries', 'train')
    val_subdir = os.path.join(dir_, 'summaries', 'val')
    raw_train_subdir = os.path.join(dir_, 'summaries', 'raw_train')
    if mode == 'w':
        if os.path.exists(dir_):
            shutil.rmtree(dir_)
        os.makedirs(train_subdir)
        os.makedirs(val_subdir)
        os.makedirs(raw_train_subdir)
        for i in range(len(raw_train_dfs)):
            raw_train_df = raw_train_dfs[i]
            val_df = val_dfs[i]
            train_df = train_dfs[i]
            train_path = os.path.join(train_subdir, str(i)+'.csv')
            val_path = os.path.join(val_subdir, str(i)+'.csv')
            raw_train_path = os.path.join(raw_train_subdir, str(i)+'.csv')
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            raw_train_df.to_csv(raw_train_path, index=False)
    elif mode == 'r':
        train_dfs = [None for _ in range(num_folds)]
        for root, _, files in os.walk(train_subdir):
            for f in files:
                if f.endswith('.csv'):
                    fold = int(f.split('.')[0])
                    train_dfs[fold] = pd.read_csv(os.path.join(root, f))
        val_dfs = [None for _ in range(num_folds)]
        for root, _, files in os.walk(val_subdir):
            for f in files:
                if f.endswith('.csv'):
                    fold = int(f.split('.')[0])
                    val_dfs[fold] = pd.read_csv(os.path.join(root, f))
        raw_train_dfs = [None for _ in range(num_folds)]
        for root, _, files in os.walk(raw_train_subdir):
            for f in files:
                if f.endswith('.csv'):
                    fold = int(f.split('.')[0])
                    raw_train_dfs[fold] = pd.read_csv(os.path.join(root, f))
        
        return train_dfs, val_dfs, raw_train_dfs

def splitting_folds(df, num_folds, normal_frac_ratio, use_existing_split_folds=False, existing_split_folds_dir=None, validation_percent=None):
    """
    split data set into k folds
    """
    if use_existing_split_folds:
        _, val_dfs, raw_train_dfs = read_write_folds(mode='r', dir_=existing_split_folds_dir, num_folds=num_folds)
    else:
        raw_train_dfs, val_dfs = splitting_folds_from_scratch(df=df, num_folds=num_folds, validation_percent=validation_percent)
    
    # train set
    train_dfs = []
    for raw_train_df in raw_train_dfs:
        train_frac_T_df = raw_train_df.loc[(raw_train_df['LUMPED_LABEL']==1) & (raw_train_df['SPINE_TYPE']=='T')]
        train_frac_L_df = raw_train_df.loc[(raw_train_df['LUMPED_LABEL']==1) & (raw_train_df['SPINE_TYPE']=='L')]
        num_frac_Ts = round(normal_frac_ratio * train_frac_T_df.shape[0])
        num_frac_Ls = round(normal_frac_ratio * train_frac_L_df.shape[0])
        train_nonfrac_T_df = raw_train_df.loc[(raw_train_df['LUMPED_LABEL']==0) & (raw_train_df['SPINE_TYPE']=='T')].sample(n=num_frac_Ts, replace=False)
        train_nonfrac_L_df = raw_train_df.loc[(raw_train_df['LUMPED_LABEL']==0) & (raw_train_df['SPINE_TYPE']=='L')].sample(n=num_frac_Ls, replace=False)
        train_df = train_nonfrac_T_df.append(train_nonfrac_L_df)
        train_df = train_df.append(train_frac_T_df)
        train_df = train_df.append(train_frac_L_df)
        train_df.reset_index(inplace=True, drop=True)
        train_dfs.append(train_df)
        # test_group = train_df.groupby(['LUMPED_LABEL', 'SPINE_TYPE']).count()['ID']
        # print(test_group)

    # print some information
    for i in range(len(raw_train_dfs)):
        raw_train_df = raw_train_dfs[i]
        val_df = val_dfs[i]
        train_df = train_dfs[i]
        print("UNSAMPLED_TRAIN:", raw_train_df.shape[0], "TRAIN:", train_df.shape[0], "VALIDATION:", val_df.shape[0])

    return train_dfs, val_dfs, raw_train_dfs

def copy_train_val_files(v1_input_dir, v2_input_dir, output_dir, train_dfs, val_dfs, raw_train_dfs, class_lumping_dict, num_threads=10):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    summary_subdir = os.path.join(output_dir, 'summaries')
    data_subdir = os.path.join(output_dir, 'original_data')
    os.makedirs(summary_subdir)
    os.makedirs(data_subdir)
    labels = set(class_lumping_dict.values())

    # write the results csv files out
    read_write_folds(mode='w', dir_=output_dir, train_dfs=train_dfs, val_dfs=val_dfs, raw_train_dfs=raw_train_dfs)

    # generate the train and val set for each training round
    for i in range(len(train_dfs)):
        # train
        train_df = train_dfs[i]
        val_df = val_dfs[i]
        for label in labels:
            train_label_df = train_df.loc[train_df['LUMPED_LABEL']==label]
            train_output_subdir = os.path.join(data_subdir, 'fold'+str(i), 'train', str(label))
            os.makedirs(train_output_subdir)
            copy_files_multithreading(v1_input_dir, v2_input_dir, train_output_subdir, train_label_df, num_threads=num_threads)

            val_label_df = val_df.loc[val_df['LUMPED_LABEL']==label]
            val_output_subdir = os.path.join(data_subdir, 'fold'+str(i), 'validation', str(label))
            os.makedirs(val_output_subdir)
            copy_files_multithreading(v1_input_dir, v2_input_dir, val_output_subdir, val_label_df, num_threads=num_threads)
