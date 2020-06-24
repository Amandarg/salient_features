import numpy as np
import pandas as pd
import fit_model_class

'''
This file contains code that is useful for the synthetic_pair_class and the fit_model_class and also the main code
'''

def compute_probability(x, y, w_proj, model_info, b = 0):
    dim = len(x)

    w_proj = w_proj.reshape(dim,1)

    threshold_function = model_info['threshold_function']
    threshold_type =  model_info['threshold_type']
    threshold = model_info['threshold']
    relative_flag = model_info['relative_flag']

    h = evaluate_threshold(x, y, threshold_function, threshold_type, threshold, relative_flag)
    not_h = [idx for idx in range(dim) if idx not in h]
    dif = x-y
    dif[not_h] = 0
    w_proj[not_h] = 0

    temp = np.dot(w_proj.T, dif.reshape(dim,1))[0][0]
    return 1 / (1+np.exp(-temp+b))

def evaluate_threshold(x, y, threshold_function, threshold_type, threshold, relative_flag):

    if threshold_function == 'dif':
        vecs = np.abs(x-y)
    elif threshold_function == 'var':
        vecs = np.std(np.column_stack((x,y)), axis = 1)**2
    else:
        raise ValueError('You did not specifiy a valid threshold_function. Options are: (1) dif; (2) var')

    if relative_flag:
        vecs = vecs / np.sum(vecs)

    if threshold_type == 'additive':
        h = []
        sorted_scores_idx = np.argsort(np.array(vecs))[::-1]
        bucket = 0
        for i in range(len(sorted_scores_idx)):
            bucket += vecs[sorted_scores_idx[i]]
            h.append(sorted_scores_idx[i])
            if bucket >= threshold:
                break
#             print('-', threshold)
#             print(vecs)
#             print(h)
    elif threshold_type == 'individual':
        n = len(vecs)
        h = [i for i in range(n) if vecs[i] > threshold]

        if len(h) == 0:
            h = [i for i in range(n)]
    elif threshold_type == 'top':
        sorted_scores_idx = np.argsort(np.array(vecs))[::-1]
        h = sorted_scores_idx[:int(threshold)]
        # print(x-y)
        # print(vecs)
        # print(h)
    elif threshold_type == 'random':
        h = np.random.choice(len(x), threshold, replace = False)
    elif threshold_type == 'random_coord':
        h = []
        for idx,i in enumerate(np.random.uniform(0,1,len(x))):
            if i <= threshold:
                h.append(idx)
    else:
        raise ValueError('You did not specifiy a valid threshold_type. Options are: (1) additive; (2) individual')

    return np.sort(h)

def prepare_ranking_object(name, avoid = []):
    usable_features = ['points', 'var_xcoord', 'var_ycoord', 'varcoord_ratio', 'avgline', 'varline', 'boyce', 'lenwid', 'jagged', 'parts', 'hull', 'bbox', 'reock', 'polsby', 'schwartzberg', 'circle_area', 'circle_perim', 'hull_area', 'hull_perim', 'orig_area', 'district_perim', 'corners', 'xvar', 'yvar', 'cornervar_ratio', 'sym_x', 'sym_y']
    features_df = pd.read_csv('district_data/subset_features.csv')
    if name == 'mturk':
        ranking_df = pd.read_csv('district_data/mturk_ranking_data')
    elif name == 'shiny2':
        ranking_df = pd.read_csv('district_data/shiny2_ranking')
    elif name == 'shiny1':
        ranking_df = pd.read_csv('district_data/shiny1_ranking')
    elif name[0:3] == 'ug1':
        study = name[3:]
        ug1_df = pd.read_csv('district_data/ug1_ranking')
        ranking_df = ug1_df.loc[ug1_df['district_set'] == study].iloc[:,1:]
        ranking_df = ranking_df.reset_index(drop = True)
    else:
        raise ValueError('Name of district.')

    districts = list(ranking_df.iloc[0][1:])
    total_districts = len(districts)
    print('There are', total_districts, 'before removing the ones with no features and the ones to avoid because they show up in training.')
    districts = [d for d in districts if d in features_df['district'].values and d not in avoid]
    print('There are', len(districts), 'with features and not avoided.')

    district_dict = {}
    their_ranking = []
    for idx, district in enumerate(districts):
        district_dict[idx] = district
        district_dict[district] = idx

    features = np.zeros((len(usable_features),len(districts)))

    for district_idx,district in enumerate(districts):
        for idx, feat in enumerate(usable_features):
            features[idx, district_idx] = features_df.loc[features_df['district'] == district_dict[district_idx]][feat]
        their_ranking.append(features_df.loc[features_df['district'] == district_dict[district_idx]]['compactness'].values[0])

    print('-')
    ranking_array = np.zeros((len(ranking_df), len(districts)))

    their_ranking = np.argsort(their_ranking) # ranking most compact to least

    for index, row in ranking_df.iterrows():
        ct = 0
        for i in range(total_districts):
            if row[str(i)] in districts:
                ranking_array[index, ct ] = district_dict[row[str(i)]]
                ct +=1

    return ranking_array, features, district_dict, their_ranking

# prepare data
'''
processes district pairwise comparison data into a form used for the fit_model class
'''
def get_df(name, min_comparisons, confidence = 0):
    usable_features = ['points', 'var_xcoord', 'var_ycoord', 'varcoord_ratio', 'avgline', 'varline', 'boyce', 'lenwid', 'jagged', 'parts', 'hull', 'bbox', 'reock', 'polsby', 'schwartzberg', 'circle_area', 'circle_perim', 'hull_area', 'hull_perim', 'orig_area', 'district_perim', 'corners', 'xvar', 'yvar', 'cornervar_ratio', 'sym_x', 'sym_y']
    df = pd.read_csv('district_data/paired_comparisons.csv')
    features_df = pd.read_csv('district_data/subset_features.csv')

    if name != 'all':
        df = df.loc[df['study'] == name]

    df = df.loc[df['alternate_id_1'] != df['alternate_id_2']]
    districts_in_shiny = set(df.alternate_id_1.unique()).union(set(df.alternate_id_2.unique()))
    districts_in_shiny_item_num_dict = {}
    for idx, district in enumerate(districts_in_shiny):
        districts_in_shiny_item_num_dict[district] = idx
        districts_in_shiny_item_num_dict[idx] = district

    comparisons = np.zeros([len(districts_in_shiny), len(districts_in_shiny)])
    seen_pairs = np.zeros([len(districts_in_shiny), len(districts_in_shiny)])
    features = np.zeros([len(usable_features), len(districts_in_shiny)])

    for index, row in df.iterrows():
        district1 = districts_in_shiny_item_num_dict[row['alternate_id_1']]
        district2 = districts_in_shiny_item_num_dict[row['alternate_id_2']]
        winner = districts_in_shiny_item_num_dict[row['alternate_id_winner']]
        if winner == district1:
            comparisons[district1, district2] += 1
        else:
            comparisons[district2, district1] += 1
        seen_pairs[district1, district2] = 1
        seen_pairs[district2, district1] = 1

    their_compactness_measure = []

    for i in range(len(districts_in_shiny)):
        their_compactness_measure.append(float(features_df.loc[features_df['district'] == districts_in_shiny_item_num_dict[i]]['compactness']))

    their_ranking = np.argsort(np.array(their_compactness_measure))[::-1]
    print('their ranking', their_ranking)

    for i in range(len(districts_in_shiny)):
        for idx, feat in enumerate(usable_features):
            features[idx, i] = features_df.loc[features_df['district'] == districts_in_shiny_item_num_dict[i]][feat]

    for i in range(len(districts_in_shiny)):
        for j in range(i+1, len(districts_in_shiny)):
            if comparisons[i,j] + comparisons[j,i] < min_comparisons:
                comparisons[i,j] = 0
                comparisons[j,i] = 0
                seen_pairs[i,j] = 0
                seen_pairs[j,i] = 0

    if confidence != 0:
        for i in range(len(districts_in_shiny)):
            for j in range(i+1, len(districts_in_shiny)):
                if comparisons[i,j] + comparisons[j,i]> 0:
                    prob_i_beats_j = comparisons[i,j] / (comparisons[j,i] + comparisons[i,j])
                    if not (prob_i_beats_j > confidence or prob_i_beats_j < 1-confidence): #or comparisons[i,j] + comparisons[j,i] < max_compares:
                        print(i,j, prob_i_beats_j, comparisons[i,j] + comparisons[j,i] )
                        comparisons[i,j] = 0
                        comparisons[j,i] = 0
                        seen_pairs[i,j] = 0
                        seen_pairs[j,i] = 0

    pairwise_data = fit_model_class.pairwise_comparisons(comparisons, features, seen_pairs, districts_in_shiny_item_num_dict, standardized = True)
    return pairwise_data, their_ranking, districts_in_shiny_item_num_dict
