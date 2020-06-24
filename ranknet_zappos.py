'''
you will need to change PATH to the appropriate path for all the files based on the create_zappos_split.py file
to run this file: `python ranknet_zappos.py ATTRIBUTE# SPLIT# HYPERPARAMETER#`. For example: `python ranknet_zappos.py 1.0 0 0` will train a model on attribute 1.0 with split 0 of the data with the 0-th item in the parameters array on line 58. The choices for the first number are 1.0, 2.0, 3.0, 4.0 (all possible attributes), 0-9 for the splits is what was done in our paper since we did 10 splits, and 0-len(parameters) for the choice of hyper parameters
'''

import preference_utils
import fit_model_class
import pandas as pd

import numpy as np
import os
import sys
import ranknet

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

# first number is the l2 regularization strength, second number is the number of hidden nodes
parameters = [(0.001, 400), (0.0001, 50), (0.0001, 200), (0.0001, 400), (0.0001, 600), (0.001, 50), (0.001, 200), (0.001, 600), (0.01, 50), (0.01, 200), (0.01, 400), (0.01, 600)]

c, nodes = parameters[idx]

print('c is', float(c))
print('nodes is', int(nodes))
total_comparisons = np.sum(train_pairwise_data.comparison_data)
total_comparisons = int(total_comparisons)
d, num_items = train_pairwise_data.standardized_features.shape
ct_to_item1 = []
ct_to_item2 = []

X_1 = np.zeros((total_comparisons, d))
X_2 = np.zeros((total_comparisons, d))
y = np.ones((X_1.shape[0], 1))

ct = 0
for i in range(num_items):
    for j in range(num_items):
        if i == j:
            continue
        else:
            for k in range(int(train_pairwise_data.comparison_data[i,j])):
                ct_to_item1.append(i)
                ct_to_item2.append(j)
                X_1[ct, :] = train_pairwise_data.standardized_features[:, i]
                X_2[ct, :] = train_pairwise_data.standardized_features[:, j]
                ct += 1


train_coarse_acc, train_fine_acc, validate_coarse_acc, validate_fine_acc, test_coarse_acc, test_fine_acc = ranknet.run_ranknet(train_pairwise_data, validate_pairwise_data, test_pairwise_data, int(nodes), float(c), np.copy(X_1), np.copy(X_2), np.copy(y))

text_file = open("PATH/zappos_ranknet/zappos_{}/{}/{}_{}.txt".format(str(attribute), str(split), str(nodes), str(c)), "w")
msg = str(train_coarse_acc)
msg += '\n' + str(train_fine_acc)
msg += '\n' + str(validate_coarse_acc)
msg += '\n' + str(validate_fine_acc)
msg += '\n' + str(test_coarse_acc)
msg += '\n' + str(test_fine_acc)
text_file.write(msg)
text_file.close()
