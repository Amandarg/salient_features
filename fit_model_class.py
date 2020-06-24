import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import preference_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
import networkx as nx
from sklearn.preprocessing import StandardScaler

#district_dict and _plot_triplet_districts
class pairwise_comparisons(object):
    '''
    This class contains code that takes features and pairwise comparisons and fits the salient feature preference model with the top t selection function. It can then fit the features BTL model by setting t = ambient dimension.
    It also can compute the empirical number of stochastic transitivity violations.

    To initiate an object you need:

    comparison_data: num_items x num_items numpy array such that comparison_data[i,j] = num times item i beat item j
    features: d x num_items matrix wher eeach column is the features for one item
    training_data: num_items x num_items numpy array such that training_data[i,j] = 1 if the pair is used in the training data
    district_dict: set to {} unless working with district data in which case you want a dictionary d where d[i] = name of district
    standardized: True means that the data will be standardized, ie subtract mean and divide by standard deviation

    Note, once initiated an object will have:
    preference_matrix: num_items x num_items numpy array such that preference_matrix[i,j] = Prob(item i beats item j)
    num_compared_matrix: num_items x num_items numpy array such that num_compared_matrix[i,j] = num times item i and j compared

    Note: an item never gets compared to itself and so preference_matrix[i,i] = 0 for instance
    '''

    def __init__(self, comparison_data, features, training_data, district_dict, StandardScalarObject = 0, standardized = False):

        self.num_items, _ = comparison_data.shape
        self.comparison_data = comparison_data
        self.training_data = training_data
        self.standardized_features = features
        self.num_compared_matrix = self._compute_num_compared_matrix()
        self.preference_matrix = self._compute_preference_matrix()

        if standardized:
            if StandardScalarObject != 0:
                print('using our standardization!!!!!')
                self.standardized_features = StandardScalarObject.transform(features.T).T
            else:
                self.StandardScalerObject = StandardScaler(copy=True, with_mean=True, with_std=True)
                self.StandardScalerObject.fit(features[:, self._items_used_in_training()].T)
                self.standardized_features = self.StandardScalerObject.transform(features.T).T

        self.tot_samples = self._compute_total_training_samples()
        self.district_dict = district_dict

#################
#INITIALIZATIONS#
#################
    def _items_used_in_training(self):
        items_used = []
        for i in range(self.num_items):
            for j in range(i+1, self.num_items):
                if self.training_data[i,j] == 1:
                    items_used.append(i)
                    items_used.append(j)
        return list(set(items_used))

    def _compute_num_compared_matrix(self):
        num_compared_matrix = np.zeros([self.num_items, self.num_items])

        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    continue
                num_compared_matrix[i,j] = self.comparison_data[i,j] + self.comparison_data[j,i]

        return num_compared_matrix

    def _compute_preference_matrix(self):
        preference_matrix = np.zeros([self.num_items, self.num_items])

        for i in range(self.num_items):
            for j in range( self.num_items):
                if i == j:
                    continue
                if self.num_compared_matrix[i,j] > 0:
                    preference_matrix[i,j] = self.comparison_data[i,j] / self.num_compared_matrix[i,j]

        return preference_matrix

    def _compute_total_training_samples(self):

        ct = 0
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    continue
                if self.training_data[i,j] == 1:
                    ct += self.comparison_data[i,j]
        return int(ct)

    def _plot_triplet_districts(self, i,j,k):
        img_A = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (i))
        img_B = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (j))
        img_C = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (k))

        images = [img_A, img_B, img_C]
        num_models = len(images)
        fig, ax = plt.subplots(1,num_models, figsize = (15,15))

        for i in range(num_models):
            ax[i].axis('off')
            ax[i].imshow(1-images[i])

        plt.axis('off')
        plt.show()

    def plot_pair_districts(self, i,j):
        img_A = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (self.district_dict[i]))
        img_B = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (self.district_dict[j]))

        images = [img_A, img_B]
        num_models = len(images)
        fig, ax = plt.subplots(1,num_models, figsize = (10,10))

        for i in range(num_models):
            ax[i].axis('off')
            ax[i].imshow(1-images[i])

        plt.axis('off')
        plt.show()

#######################################
#STATISTICS AND OTHER QUANTITIES#
#######################################

    def get_connected_comp_and_graph(self, min_compare, pairwise_data):
        G = nx.Graph()

        for i in range(self.num_items):
            for j in range(self.num_items):
                if i != j and self.comparison_data[i,j] + self.comparison_data[j,i] > min_compare:
                    G.add_edge(i,j)
        cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        print(cc, len(cc))
        largest_components=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
        for index,component in enumerate(largest_components):
            nx.draw(component, with_labels=True)
            #nx.savefig('fig{}.pdf'.format(index))
            #plt.clf()
            plt.draw()
            plt.show()

    def plot_number_times_each_compared(self):
        values = []
        M = self.comparison_data + self.comparison_data.T
        for i in range(self.num_items):
            for j in range(i+1, self.num_items):
                if M[i,j] != 0:
                    values.append(M[i,j])

        print('There are', self.num_items, 'items.')
        print('There are', len(values), 'seen pairs.')
        plt.title('Histogram of # of times each pair compared \n out of those with at least one sample')
        plt.hist(values)
        plt.show()

    def plot_empirical_pairwise_probabilities(self):
        probs = []
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    continue
                if self.training_data[i,j] == 1:
                    probs.append(self.preference_matrix[i,j])
        plt.hist(probs)
        plt.title('Histogram of empirical pairwise probabilities')
        plt.show()

    def st_violations(self, min_compared, plot = False, limit_set = []):
        num_valid_trips = 0
        weak_st_violations = []
        moderate_st_violations = []
        sst_violations = []
        triplets = {}

        if len(limit_set) > 0:
            iterate_set = limit_set
        else:
            iterate_set = range(self.num_items)

        for i in iterate_set:
            # print(i / self.num_items)
            for j in iterate_set:
                if i == j:
                    continue
                for k in iterate_set:
                    if i == k or j == k:
                        continue

                    # do not consider any triplets where one of the three possible paris were not even compared
                    if self.num_compared_matrix[i,j] < min_compared or self.num_compared_matrix[j,k] < min_compared or self.num_compared_matrix[i,k] < min_compared:
                        continue
                    if self.preference_matrix[i,j] > .5 and self.preference_matrix[j,k] > .5:
                        trip = str(np.sort([i,j,k]))
                        if trip not in triplets:
                            triplets[trip] = 0
                        triplets[trip] += 1
                        num_valid_trips +=1
                        # print(i,j,k)
                        # print('Pij, Pjk, Pik:', self.preference_matrix[i,j], self.preference_matrix[j,k], self.preference_matrix[i,k])
                        # print('Num comparisons', self.num_compared_matrix[j,i], self.num_compared_matrix[k,j], self.num_compared_matrix[k,i])
                        if plot:
                            self._plot_triplet_districts(self.district_dict[i],self.district_dict[j],self.district_dict[k])

                        if self.preference_matrix[i,k] < .5:
                            # print('^ WEAK TRANSITIVITY VIOLATION')
                            if trip not in weak_st_violations:
                                weak_st_violations.append(trip)
                        if self.preference_matrix[i,k] < max(self.preference_matrix[i,j], self.preference_matrix[j,k]):
                            # print('^ STRONG TRANSITIVITY VIOLATION')
                            if trip not in sst_violations:
                                sst_violations.append(trip)
                        if self.preference_matrix[i,k] < min(self.preference_matrix[i,j], self.preference_matrix[j,k]):
                            # print('^ MODERATE TRANSITIVITY VIOLATION')
                            if trip not in moderate_st_violations:
                                moderate_st_violations.append(trip)

        print('Weak ST violations', len(weak_st_violations))
        print('Strong ST violations', len(sst_violations))
        print('Moderate ST violations', len(moderate_st_violations))
        print('out of', num_valid_trips, 'valid triplets that could have been violated.')
        return {'moderate_st_violations': len(moderate_st_violations), 'strong_st_violations': len(sst_violations), 'weak_st_violations': len(weak_st_violations), 'num_valid_trips': len(triplets)}, triplets

    def compute_accuracy(self, w, model_info, b = 0, train = True):

        d, _ = self.standardized_features.shape
        acc = 0
        num_compare = 0

        for i in range(self.num_items):
            for j in range(i + 1, self.num_items):
                if i == j:
                    continue
                if train and self.training_data[i,j] == 0:
                    continue
                if not train and self.training_data[i,j] == 1:
                    continue

                if self.comparison_data[i,j] + self.comparison_data[j, i] > 0:
                    num_compare += 1
                    x = np.copy(self.standardized_features[:, i])
                    y = np.copy(self.standardized_features[:, j])
                    predicted_prob = preference_utils.compute_probability(x, y, np.copy(w), model_info)
                    if (self.preference_matrix[i,j] - .5)*(predicted_prob - .5) >0:
                        acc +=1

        return acc / num_compare

#################
#HELPER FUNCTIONS#
#################
    # used by fit_btl_features to get the final ranking
    def _compute_ranking_from_w(self, w):
        scores = (w@self.standardized_features)[0]
        btl_ranking = np.argsort(scores, axis = 0)
        return btl_ranking, scores

    def _plot_districts(self, i,j,h, feature_names):
        print('h', [feature_names[k] for k in h])

        img_A = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (i))
        img_B = mpimg.imread("../../../../Dropbox/grey maps/%s.jpg" % (j))

        images = [img_A, img_B]
        num_models = len(images)
        fig, ax = plt.subplots(1,num_models, figsize = (15,15))

        for i in range(num_models):
            ax[i].axis('off')
            ax[i].imshow(1-images[i])

        plt.axis('off')
        plt.show()

############
#FIT MODELS#
############

    def _train(self, X, y, Cs, penalty, solver):
        LR_l2 = {}
        SVM_l2 = {}

        for c in Cs:
            lr = LogisticRegression(penalty = penalty, tol = 1e-10, dual=False, C = c, fit_intercept = False, max_iter = 10000, verbose = 2, solver = solver).fit(X, y)
            ranking, scores = self._compute_ranking_from_w(lr.coef_)
            LR_l2[c] = [np.copy(ranking), np.copy(scores), np.copy(lr.coef_), np.copy(lr.intercept_)]

            svm_model = svm.LinearSVC( dual=False,  max_iter = 10000, C = c, tol = 1e-10, fit_intercept = False).fit(X, y)
            svm_ranking, svm_scores = self._compute_ranking_from_w(svm_model.coef_)
            SVM_l2[c] = [np.copy(svm_ranking), np.copy(svm_scores), np.copy(svm_model.coef_)]

        return LR_l2, SVM_l2

    def _count_features(self, h, feature_counts):
        if len(h) != len(feature_counts):
            for i in h:
                feature_counts[i] += 1
        return feature_counts

    def fit_threshold(self, threshold, feature_names, h_limit, Cs, threshold_type, threshold_function, relative_flag, print_info = False, penalty = 'none', solver = 'liblinear'):
        feature_counts = [0 for i in range(len(feature_names))]
        d, num_items = self.standardized_features.shape
        seen_h = []
        seen_h_no_sat = []

        # self.tot_samples = self._compute_total_training_samples()
        tot_samples = self.tot_samples
        X = np.zeros((tot_samples, d))
        y = []
        idx = 0

        for i in range(self.num_items):
            for j in range(self.num_items):
                if self.training_data[i,j] == 1 and i != j:

                    h = preference_utils.evaluate_threshold(self.standardized_features[:, i], self.standardized_features[:,j], threshold_function, threshold_type, threshold, relative_flag)

                    feature_counts = self._count_features(h, feature_counts)

                    if len(h) < h_limit:
                        self._plot_districts(self.district_dict[i], self.district_dict[j], h, feature_names)

                    seen_h.append(len(h))

                    if len(h) < len(feature_names):
                        seen_h_no_sat.append(len(h))

                    for k in range(int(self.comparison_data[i,j])):

                        X[idx,h] = self.standardized_features[h, i] - self.standardized_features[h,j]
                        if idx % 2 == 0:
                            y.append(1)
                        else:
                            X[idx,:] = -X[idx,:]
                            y.append(-1)
                        idx += 1

        LR_l2, SVM_l2 = self._train(X, y, Cs, penalty = penalty, solver = solver)

        if print_info:
            plt.hist(seen_h)
            print('average size of h--not counting repeated pairs', np.mean(np.array(seen_h)))
            print('average size of h--not counting repeated pairs and saturation', np.mean(np.array(seen_h_no_sat)))
            plt.title('Histogram of length of coordinates used only over unique pairs', size = 20)
            plt.ylabel('frequency', size = 20)
            plt.xlabel('number coordinates used', size = 20)
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 10))

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(feature_names)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(feature_names, size = 15)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", size = 15,
                     rotation_mode="anchor")

            plt.bar([i for i in range(len(feature_names))], feature_counts)
            plt.title('Frequency of used features', size = 20)
            plt.ylabel('frequency', size = 20)
            plt.xlabel('feature name', size = 20)
            plt.tight_layout()
            plt.savefig('figs/paper/district/feature_distribution.pdf')
            plt.show()


            print(feature_counts)

        return LR_l2, SVM_l2, seen_h, seen_h_no_sat
