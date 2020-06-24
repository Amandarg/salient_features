'''
you will need to change PATH to your own paths for getting the zappos data and for recording the results
to run this file: `python zappos_exp.py ATTRIBUTE# SPLIT# HYPERPARAMETER#`. For example: `python zappos_exp.py 1.0 0 0` will train a model on attribute 1.0 with split 0 of the data with the 0-th choice for t in the top t-selection function in line 57. The choices for the first number are 1.0, 2.0, 3.0, 4.0 (all possible attributes), 0-9 for the splits is what was done in our paper since we did 10 splits, and 0-len(thresholds) for the choice of hyper parameters

'''

import preference_utils
import fit_model_class
import pandas as pd

import numpy as np
import os
import sys

# change paths
# input is the attribute

attribute = sys.argv[1]
split = sys.argv[2]
idx = sys.argv[3]

print('The attribute is', attribute)
print('the split is', split)

name = '{}_{}'.format(str(attribute), str(split))
train_features = np.array(pd.read_csv('PATH/train_features_%s.csv' % (name), header=None))
validate_features = np.array(pd.read_csv('PATH/validate_features_%s.csv' % (name), header=None))
test_features = np.array(pd.read_csv('PATH/test_features_%s.csv' % (name), header=None))

print(train_features.shape, validate_features.shape, test_features.shape)

d, n = train_features.shape
# read in train/validate/test
comparison_train = np.array(pd.read_csv('PATH/comparison_train_%s.csv' % (name), header=None))
print(np.sum(comparison_train))
comparison_validate = np.array(pd.read_csv('PATH/comparison_validate_%s.csv' % (name), header=None))
print(np.sum(comparison_validate))
comparison_test = np.array(pd.read_csv('PATH/comparison_test_%s.csv' % (name), header=None))
print(np.sum(comparison_test))


used_train = np.array(pd.read_csv('PATH/used_train_%s.csv' % (name), header=None))
print(np.sum(used_train))
used_train_coarse = np.array(pd.read_csv('PATH/used_train_coarse_%s.csv' % (name), header=None))
print(np.sum(used_train))
used_validate = np.array(pd.read_csv('PATH/used_validate_coarse_%s.csv' %(name), header=None))
print(np.sum(used_validate))
used_test = np.array(pd.read_csv('PATH/used_test_coarse_%s.csv' %(name), header=None))
print(np.sum(used_test))

train_pairwise_data = fit_model_class.pairwise_comparisons(comparison_train, train_features, used_train, {}, standardized = True)
train_coarse_pairwise_data = fit_model_class.pairwise_comparisons(comparison_train, train_features, used_train_coarse, {}, standardized = True, StandardScalarObject = train_pairwise_data.StandardScalerObject)
validate_pairwise_data = fit_model_class.pairwise_comparisons(comparison_validate, validate_features, used_validate, {}, standardized = True, StandardScalarObject = train_pairwise_data.StandardScalerObject)
test_pairwise_data = fit_model_class.pairwise_comparisons(comparison_test, test_features, used_test, {}, standardized = True, StandardScalarObject = train_pairwise_data.StandardScalerObject)


thresholds = range(10, d+1, 10)
# for i in range(1, d+1, 10):
threshold = thresholds[idx]
print('The threshold is', threshold)

Cs = [.000001, .00001, .0001, .001, .01, .1, 1, 10]

LR_l2, SVM_l2, seen_h, seen_h_no_sat = train_pairwise_data.fit_threshold(threshold, [1 for i in range(d)], -1, Cs, 'top', 'var', False, print_info = False, penalty = 'l2')

model_info = {
    'threshold_function': 'var',
    'threshold_type': 'top',
    'threshold': threshold,
    'relative_flag': False
}

for c in Cs:
    acc_coarse_train = train_coarse_pairwise_data.compute_accuracy(LR_l2[c][2], model_info, train = True)
    print('train acc (coarse) for regularization {} and threshold {} is {}'.format(c, threshold, np.round(acc_coarse_train,3)))
    # acc_fine_train = train_coarse_pairwise_data.compute_accuracy(LR_l2[c][2], model_info, train = False)
    # print('test acc (fine) for regularization {} and threshold {} is {}'.format(c, threshold, np.round(acc_fine_train,3)))
    acc_fine_train = 0

    acc_coarse_validate = validate_pairwise_data.compute_accuracy(LR_l2[c][2], model_info, train = True)
    print('validate acc (coarse) for regularization {} and threshold {} is {}'.format(c, threshold, np.round(acc_coarse_validate,3)))
    acc_fine_validate = validate_pairwise_data.compute_accuracy(LR_l2[c][2], model_info, train = False)
    print('test acc (fine) for regularization {} and threshold {} is {}'.format(c, threshold, np.round(acc_fine_validate,3)))

    acc_coarse_test = test_pairwise_data.compute_accuracy(LR_l2[c][2], model_info, train = True)
    print('acc (coarse) for regularization {} and threshold {} is {}'.format(c, threshold, np.round(acc_coarse_test,3)))
    acc_fine_test = test_pairwise_data.compute_accuracy(LR_l2[c][2], model_info, train = False)
    print('acc (fine) for regularization {} and threshold {} is {}'.format(c, threshold, np.round(acc_fine_test,3)))

    text_file = open("PATH/zappos_results/zappos_{}/svm/{}/{}_{}.txt".format(str(attribute), str(split), str(threshold), str(c)), "w")
    #text_file = open("/Users/amandarg/Documents/2019/code/cleaned_preferences/zappos_results/zappos_{}/{}_{}.txt".format(name, str(threshold), str(c)), "w")
    msg = str(acc_coarse_validate) + '\n' + str(acc_fine_validate) + '\n' + str((acc_coarse_validate + acc_fine_validate) / 2)
    msg += '\n' + str(acc_coarse_test) + '\n' + str(acc_fine_test) + '\n' + str((acc_coarse_test + acc_fine_test) / 2)
    msg += '\n' + str(acc_coarse_train) + '\n' + str(acc_fine_train) + '\n' + str((acc_coarse_train + acc_fine_train) / 2)
    text_file.write(msg)
    text_file.close()
