import shutil
import os
from sklearn.model_selection import KFold
import threading


def df_add_lumped_class(df, class_lumping_dict):
    for sq in class_lumping_dict:
        df.loc[df['XQSQ']==sq, 'LUMPED_LABEL'] = class_lumping_dict[sq]
    df = df.astype({'LUMPED_LABEL': 'int64'})
    return df


def df_add_vb_filenames(df):
    filenames = []
    for index, row in df.iterrows():
        label = str(row['LUMPED_LABEL'])
        visit = 'V' + str(row['VISIT'])
        image = str(row['ID'])
        vb = str(row['XQVERT'])
        filename = '_'.join([label, visit, image, vb]) + '.tiff'
        filenames.append(filename)
    df['FILENAME'] = filenames
    return df


def split_dataframe(df, num):
    kf = KFold(n_splits=num)
    k_folds_indices = kf.split(df.index)
    split_dfs = []
    for _, index in k_folds_indices:
        split_dfs.append(df.iloc[index])
    return split_dfs


def copy_files_func(v1_input_dir, v2_input_dir, output_dir, df):
    for index, row in df.iterrows():
        visit = str(row['VISIT'])
        image = str(row['ID'])
        vb = str(row['XQVERT'])
        if visit == '1':
            dir_ = v1_input_dir
        elif visit == '2':
            dir_ = v2_input_dir
        spath = os.path.join(dir_, image, vb+'.tiff')
        filename = row['FILENAME']
        dpath = os.path.join(output_dir, filename)
        shutil.copyfile(spath, dpath)


def copy_files_multithreading(v1_input_dir, v2_input_dir, output_dir, df, num_threads=10):
    split_dfs = split_dataframe(df, num_threads)
    print(output_dir)
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=copy_files_func, args=(v1_input_dir, v2_input_dir, output_dir, split_dfs[i]))
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()

