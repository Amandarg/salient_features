'''
this file parses the zappos data and splits into a train/validation/test set
you will have to change PATH to the appropriate path to the zappos data and a path to where you want to save the train/validation/test splits
to run this file you will need to specify a split number, i.e. `python create_zappos_split.py 0` creates a split named 0. We use 10 splits in the paper.
'''
import random
import pandas as pd
import sys
import scipy.io
import numpy as np
import pickle
split = sys.argv[1]
print('the split is', split)

for attribute in [1.0, 2.0, 3.0, 4.0]:
    data = scipy.io.loadmat('PATH/zappos-labels.mat')
    label = 'mturkOrder'

    columns=['item1','item2','attribute','comparison','confidence','agreement', 'type']

    df = pd.DataFrame(columns=columns)
    df = df.fillna(0)

    for j in range(len(data[label][0])):
        for k in range(len(data[label][0][j])):
            if data[label][0][j][k][3] != 3.0 and data[label][0][j][k][2] == attribute:
                temp = list(data[label][0][j][k])
                temp.extend(['coarse'])
                df.loc[len(df)] = temp
            elif data[label][0][j][k][3] == 3.0:
                print('is')

    data = scipy.io.loadmat('PATH/zappos-labels-fg.mat')
    label = 'mturkHard'

    for j in range(len(data[label][0])):
        for k in range(len(data[label][0][j])):
            if data[label][0][j][k][3] != 3.0 and data[label][0][j][k][2] == attribute:
                temp = list(data[label][0][j][k])
                temp.extend(['fine'])
                df.loc[len(df)] = temp
            elif data[label][0][j][k][3] == 3.0:
                print('is')

    id_to_num = {'train':{}, 'validate': {}, 'test':{}}
    num_to_id = {'train':{}, 'validate': {}, 'test':{}}

    # assing to train/validate/test
    train_prob = .7
    validate_prob = .15

    train = []
    validate = []
    test = []

    for i in range(len(df)):
        x = np.random.uniform(0,1,1)[0]
        if x < train_prob:
            train.append(i)
        elif x >= train_prob and x < train_prob + validate_prob:
            validate.append(i)
        else:
            test.append(i)

    with open("PATH/zappos_results/splits/train_split_{}_{}.txt".format(attribute, split), 'wb') as f:
        pickle.dump(train, f)

    with open("PATH/zappos_results/splits/validate_split_{}_{}.txt".format(attribute, split), 'wb') as f:
        pickle.dump(validate, f)

    with open("PATH/zappos_results/splits/test_split_{}_{}.txt".format(attribute, split), 'wb') as f:
        pickle.dump(test, f)

    # map shoe id to a number
    print('Confidences', set(df['confidence']))
    for index,row in df.iterrows():

        label = 'train'
        if index in train:
            label = 'train'
        elif index in validate:
            label = 'validate'
        elif index in test:
            label = 'test'
        else:
            print('something is broken!')
        shoe1 = row['item1']
        shoe2 = row['item2']

        if shoe1 not in num_to_id[label].values():
            num_to_id[label][len(num_to_id[label])] = int(shoe1)
            id_to_num[label][int(shoe1)] = len(num_to_id[label]) - 1
        if shoe2 not in num_to_id[label].values():
            num_to_id[label][len(num_to_id[label])] = int(shoe2)
            id_to_num[label][int(shoe2)] = len(num_to_id[label]) - 1

    print('There are', len(num_to_id['train']), 'in the train comparisons.')
    print('There are', len(num_to_id['validate']), 'in the validate comparisons.')
    print('There are', len(num_to_id['test']), 'in the test comparisons.')

    # you are training on everything in the traning set and not keeping track of coarse and fine
    comparison_train = np.zeros((len(num_to_id['train'].values()),len(num_to_id['train'].values())))
    used_train = np.zeros((len(num_to_id['train'].values()),len(num_to_id['train'].values())))
    used_train_coarse = np.zeros((len(num_to_id['train'].values()),len(num_to_id['train'].values())))

    # you wlil use the "train" here to refer to fine grained; and used for "coarse". same as below
    comparison_validate = np.zeros((len(num_to_id['validate'].values()),len(num_to_id['validate'].values())))
    used_validate_coarse = np.zeros((len(num_to_id['validate'].values()),len(num_to_id['validate'].values())))

    comparison_test = np.zeros((len(num_to_id['test'].values()),len(num_to_id['test'].values())))
    used_test_coarse = np.zeros((len(num_to_id['test'].values()),len(num_to_id['test'].values())))

    for index, row in df.iterrows():

        label = 'train'
        if index in train:
            label = 'train'
        elif index in validate:
            label = 'validate'
        elif index in test:
            label = 'test'
        else:
            print('something is broken!')

        item1=id_to_num[label][row['item1']]
        item2=id_to_num[label][row['item2']]

        num_agree = int(5*row['agreement'])
        num_disagree = 5 - num_agree

        if row['comparison'] == 2.0:
    #         temp = item1
    #         item1 = item2
    #         item2 = temp
            temp = num_agree
            num_agree = num_disagree
            num_disagree = temp
        if row['comparison'] == 3.0:
            print('there is a prob')
        if label == 'train':
            comparison_train[item1,item2]+= num_agree
            comparison_train[item2,item1]+= num_disagree
            used_train[item1,item2] = 1
            used_train[item2,item1] = 1
            if row['type'] == 'coarse':
                used_train_coarse[item1,item2] = 1
                used_train_coarse[item2,item1] = 1

        elif label == 'validate':
            comparison_validate[item1,item2]+= num_agree
            comparison_validate[item2,item1]+= num_disagree

            if row['type'] == 'coarse':
                used_validate_coarse[item1,item2]=1
                used_validate_coarse[item2,item1]=1

        elif label == 'test':
            comparison_test[item1,item2]+= num_agree
            comparison_test[item2,item1]+= num_disagree

            if row['type'] == 'coarse':
                used_test_coarse[item1,item2]=1
                used_test_coarse[item2,item1]=1
        else:
            print('something went wrong')

    name = '{}_{}'.format(str(attribute), str(split))
    np.savetxt('PATH/comparison_train_%s.csv'  %(name), comparison_train, delimiter=',')
    np.savetxt('PATH/used_train_%s.csv' %(name), used_train,delimiter=',')
    np.savetxt('PATH/used_train_coarse_%s.csv' %(name), used_train_coarse,delimiter=',')

    np.savetxt('PATH/comparison_validate_%s.csv' %(name), comparison_validate,  delimiter=',')
    np.savetxt('PATH/used_validate_coarse_%s.csv' %(name), used_validate_coarse,  delimiter=',')

    np.savetxt('PATH/comparison_test_%s.csv' %(name), comparison_test,  delimiter=',')
    np.savetxt('PATH/used_test_coarse_%s.csv' %(name), used_test_coarse,  delimiter=',')

    gist = scipy.io.loadmat('PATH/zappos-gist.mat')['gistfeats']
    color = scipy.io.loadmat('PATH/zappos-color.mat')['colorfeats']
    features = np.concatenate((gist, color), axis = 1)
    _, d = features.shape
    rows = [num_to_id['train'][i] for i in range(len(num_to_id['train']))]
    train_features = features[rows,:].T

    rows = [num_to_id['validate'][i] for i in range(len(num_to_id['validate']))]
    validate_features = features[rows,:].T

    rows = [num_to_id['test'][i] for i in range(len(num_to_id['test']))]
    test_features = features[rows,:].T

    np.savetxt('PATH/train_features_%s.csv' %(str(name)), train_features, delimiter=',')
    np.savetxt('PATH/test_features_%s.csv' %(str(name)), test_features, delimiter=',')
    np.savetxt('PATH/validate_features_%s.csv' %(str(name)), validate_features , delimiter=',')
