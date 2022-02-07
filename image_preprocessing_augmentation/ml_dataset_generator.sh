#!/bin/bash

# number of threads used
# should be set bigger than 1
num_threads=10
# the ratio of # of normal instances to fx # of instances when downsampling the normal instances in training set
normal_frac_ratio=25e-1
# the directory storeing train and valid sets
train_val_output_dir=/data/Spine_ML_Data/mros_ml_train_val_sets/normal_fx_ratio_25e-1
# whether to generate a new test set
generating_new_testset=false

num_augmentations=15

if $generating_new_testset ; then
	# the ratio of # of test instances to # of all instances
	test_percent=0.15
	# the directory storing the vbs in the test set
	test_vbs_output_dir=/data/Spine_ML_Data/mros_ml_test_sets/testset10p
	# the number of rounds to be generate for kfold cross validation
	num_folds=1
	# when num_folds == 1, set the ratio of # of validation instances to (# of valid instances + # of training instances)
	validation_percent=0.1
else
	# in each round, whether re-downsampling the train set (while keep the validation set unchanged)
	resampling_trainset=false
	# the directory containing existing round splitting schema, must be set if use_existing_split_folds == True
	existing_split_folds_dir=/data/Spine_ML_Data/mros_ml_train_val_sets/normal_fx_ratio_1
	# the directory containing the existing test set
	existing_test_dir=/data/Spine_ML_Data/mros_ml_test_sets/testset10p
fi


if $generating_new_testset ; then
	python split_dataset/main.py --num_threads $num_threads --train_val_vbs_output_dir $train_val_output_dir --generating_new_testset --test_percent $test_percent --test_vbs_output_dir $test_vbs_output_dir --num_folds $num_folds --normal_frac_ratio $normal_frac_ratio --resampling_trainset --validation_percent $validation_percent
	python black_bone_conversion/main.py --input_dir $test_vbs_output_dir --num_threads $num_threads
else
	if $resampling_trainset ; then
		python split_dataset/main.py --num_threads $num_threads --train_val_vbs_output_dir $train_val_output_dir --normal_frac_ratio $normal_frac_ratio --existing_split_folds_dir $existing_split_folds_dir --resampling_trainset --existing_test_dir $existing_test_dir
	else
		python split_dataset/main.py --num_threads $num_threads --train_val_vbs_output_dir $train_val_output_dir --normal_frac_ratio $normal_frac_ratio --existing_split_folds_dir $existing_split_folds_dir --existing_test_dir $existing_test_dir
	fi

fi

python black_bone_conversion/main.py --input_dir $train_val_output_dir --num_threads $num_threads
python augmentation/main.py --needed_vbs_dir $train_val_output_dir --num_threads $num_threads --num_augmentations $num_augmentations --black_bone_converting
