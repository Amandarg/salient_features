import embedding_class
import synthetic_pair_class
import numpy as np
import fit_model_class

'''
This file contains code for running the synthetic experiments in the paper.
'''
def get_results(num_items, dim, num_exp, num_results, thresholds):
    '''
    this function gets the parameters in thm 1 + stochastic transitivity rates and pairwise inconsistency rates
    '''
    results = [[[] for i in range(len(thresholds))] for k in range(num_results)]

    for i in range(num_exp):
        print('on exp', i)
        embedding_object = embedding_class.embedding(num_items, dim, 0, np.sqrt(1/dim))
        original_U = np.copy(embedding_object.U)
        original_w = np.copy(embedding_object.w)
        for idx,k in enumerate(thresholds):

            # #for scaling
            # embedding_object.U = (np.sqrt(dim)/ np.sqrt(idx +1))*original_U
#             embedding_object.U = (np.sqrt(dim)/ np.sqrt(k))*original_U
#             embedding_object.w = (np.sqrt(dim)/ np.sqrt(idx+1))*original_w
            print('on exp', i, 'and k', k)
            synthetic_object = synthetic_pair_class.synthetic_pairs(embedding_object,
                                   {'threshold_function': 'var',
                                    'threshold_type':'top',
                                    'threshold': k,
                                    'relative_flag': True})
            parameters = synthetic_object.get_parameters()
            parameters = [j for j in parameters]

            temp = synthetic_object.analyze_transitivity_and_pairwise_inconsistencies()
            for j in temp:
                parameters.append(j)

            for j in range(len(parameters)):
                results[j][idx].append(parameters[j])
    return results

def sample_sweep(dim, num_items, num_exp, threshold, samples):
    '''
    this function gets the estimation error, kendall tau correlation, and the pairwise accuracy rates as the number of samples vary
    '''
    threshold_function = 'var'
    threshold_type = 'top'
    results = [[] for i in range(len(samples))]
    btl_results = [[] for i in range(len(samples))]
    prediction_results = [[] for i in range(len(samples))]

    kt_results = [[] for i in range(len(samples))]
    btl_kt_results = [[] for i in range(len(samples))]
    btl_prediction_results = [[] for i in range(len(samples))]

    embedding_object = embedding_class.embedding(num_items, dim, 0, np.sqrt(1/dim))
    synthetic_object = synthetic_pair_class.synthetic_pairs(embedding_object,
           {'threshold_function': threshold_function,
            'threshold_type': threshold_type,
            'threshold': threshold,
            'relative_flag': True})

    embedding_object_unseen = embedding_class.embedding(num_items, dim, 0, np.sqrt(1/dim))
    embedding_object_unseen.w = np.copy(synthetic_object.w)
    parameters = synthetic_object.get_parameters()
    synthetic_object_unseen = synthetic_pair_class.synthetic_pairs(embedding_object_unseen,
           {'threshold_function': threshold_function,
            'threshold_type': threshold_type,
            'threshold': threshold,
            'relative_flag': True})
    print('object unsen@')
    parameters = synthetic_object_unseen.get_parameters()
    parameters = synthetic_object.get_parameters()
    bound = [[parameters[5]*(1 / np.sqrt(i)) for j in range(10)] for i in samples]
    print('samples', parameters[6], parameters[7])

    model_info =  {'threshold_function': threshold_function,
                'threshold_type': threshold_type,
                'threshold': threshold,
                'relative_flag': True}

    for i in range(num_exp):
        for idx,s in enumerate(samples):
            print('on exp {} and samples {}'.format(i,s))

            comparison_data = synthetic_object.get_comparison_matrix(s)
            fit_model_instance = fit_model_class.pairwise_comparisons(comparison_data, embedding_object.U, np.ones((num_items, num_items)), {})
            result = fit_model_instance.fit_threshold(threshold, [1 for i in range(dim)], -1, [1000000000], threshold_type, threshold_function, False, solver = 'sag')
            results[idx].append(synthetic_object.get_w_est_error(result[0][1000000000][2][0]))
            kt, _ = synthetic_object_unseen.get_ranking_error_and_score_rsme(result[0][1000000000][2][0])
            kt_results[idx].append(kt)
            prediction_results[idx].append(synthetic_object_unseen.get_prediction_error(result[0][1000000000][2][0], model_info, btl = False))

            fit_model_instance = fit_model_class.pairwise_comparisons(comparison_data, embedding_object.U, np.ones((num_items, num_items)), {})
            result = fit_model_instance.fit_threshold(dim, [1 for i in range(dim)], -1, [1000000000], threshold_type, threshold_function, False, solver = 'sag')
            btl_results[idx].append(synthetic_object.get_w_est_error(result[0][1000000000][2][0]))
            kt, _ = synthetic_object_unseen.get_ranking_error_and_score_rsme(result[0][1000000000][2][0])
            btl_kt_results[idx].append(kt)
            btl_prediction_results[idx].append(synthetic_object_unseen.get_prediction_error(result[0][1000000000][2][0], model_info, btl = True))


    return results, btl_results, bound, kt_results, btl_kt_results, prediction_results, btl_prediction_results

def sample_mispecified(dim, num_items, num_exp, threshold, samples, thresholds_try):
    '''
    this function gets the estimation error, kendall tau correlation, and the pairwise accuracy rates as the number of samples vary
    '''
    threshold_function = 'var'
    threshold_type = 'top'
    results = {t: [[] for i in range(len(samples))] for t in thresholds_try}
    prediction_results = {t: [[] for i in range(len(samples))] for t in thresholds_try}

    kt_results = {t: [[] for i in range(len(samples))] for t in thresholds_try}

    embedding_object = embedding_class.embedding(num_items, dim, 0, np.sqrt(1/dim))
    synthetic_object = synthetic_pair_class.synthetic_pairs(embedding_object,
           {'threshold_function': threshold_function,
            'threshold_type': threshold_type,
            'threshold': threshold,
            'relative_flag': True})

    embedding_object_unseen = embedding_class.embedding(num_items, dim, 0, np.sqrt(1/dim))
    embedding_object_unseen.w = np.copy(synthetic_object.w)
    parameters = synthetic_object.get_parameters()
    synthetic_object_unseen = synthetic_pair_class.synthetic_pairs(embedding_object_unseen,
           {'threshold_function': threshold_function,
            'threshold_type': threshold_type,
            'threshold': threshold,
            'relative_flag': True})
    print('object unseen')
    parameters = synthetic_object_unseen.get_parameters()
    parameters = synthetic_object.get_parameters()
    bound = [[parameters[5]*(1 / np.sqrt(i)) for j in range(10)] for i in samples]
    print('samples', parameters[6], parameters[7])



    for i in range(num_exp):
        for idx,s in enumerate(samples):
            print('on exp {} and samples {}'.format(i,s))

            for fit_threshold in thresholds_try:

                comparison_data = synthetic_object.get_comparison_matrix(s)
                fit_model_instance = fit_model_class.pairwise_comparisons(comparison_data, embedding_object.U, np.ones((num_items, num_items)), {})
                result = fit_model_instance.fit_threshold(fit_threshold, [1 for i in range(dim)], -1, [1000000000], threshold_type, threshold_function, False, solver = 'sag')
                results[fit_threshold][idx].append(synthetic_object.get_w_est_error(result[0][1000000000][2][0]))
                kt, _ = synthetic_object_unseen.get_ranking_error_and_score_rsme(result[0][1000000000][2][0])
                kt_results[fit_threshold][idx].append(kt)

                model_info =  {'threshold_function': threshold_function,
                            'threshold_type': threshold_type,
                            'threshold': fit_threshold,
                            'relative_flag': True}
                prediction_results[fit_threshold][idx].append(synthetic_object_unseen.get_prediction_error(result[0][1000000000][2][0], model_info, btl = False))

    return results, bound, kt_results, prediction_results


def threshold_sweep_samples(dim, num_items, num_exp, thresholds, num_samples, rescale = False):
    '''
    this function gets the estimation error and the kendall tau correlation as the threshold varies
    '''
    threshold_function = 'var'
    threshold_type = 'top'
    results = [[] for i in range(len(thresholds))]
    btl_results = [[] for i in range(len(thresholds))]

    kt_results = [[] for i in range(len(thresholds))]
    btl_kt_results = [[] for i in range(len(thresholds))]

    for i in range(num_exp):
        embedding_object = embedding_class.embedding(num_items, dim, 0, np.sqrt(1/dim))
        original_U = np.copy(embedding_object.U)
        for idx,k in enumerate(thresholds):
            print('on exp {} and threshold {}'.format(i,k))
            if rescale:
                embedding_object.U = (np.sqrt(dim)/ np.sqrt(k))*original_U
            synthetic_object = synthetic_pair_class.synthetic_pairs(embedding_object,
                   {'threshold_function': threshold_function,
                    'threshold_type': threshold_type,
                    'threshold': k,
                    'relative_flag': True})
            comparison_data = synthetic_object.get_comparison_matrix(num_samples)
            fit_model_instance = fit_model_class.pairwise_comparisons(comparison_data, embedding_object.U, np.ones((num_items, num_items)), {})
            result = fit_model_instance.fit_threshold(k, [1 for i in range(dim)], -1, [1000000000], threshold_type, threshold_function, False, solver = 'sag')

            orig_norm = np.linalg.norm(embedding_object.w)
            results[idx].append(synthetic_object.get_w_est_error(result[0][1000000000][2][0]) / orig_norm)
            kt, my_kt= synthetic_object.get_ranking_error_and_score_rsme(result[0][1000000000][2][0])
            kt_results[idx].append(my_kt)

            fit_model_instance = fit_model_class.pairwise_comparisons(comparison_data, embedding_object.U, np.ones((num_items, num_items)), {})
            result = fit_model_instance.fit_threshold(dim, [1 for i in range(dim)], -1, [1000000000], threshold_type, threshold_function, False, solver = 'sag')

            orig_norm = np.linalg.norm(embedding_object.w)
            btl_results[idx].append(synthetic_object.get_w_est_error(result[0][1000000000][2][0]) / orig_norm)
            kt, my_kt= synthetic_object.get_ranking_error_and_score_rsme(result[0][1000000000][2][0])
            btl_kt_results[idx].append(my_kt)

    return results, btl_results, kt_results, btl_kt_results
