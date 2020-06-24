import preference_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class synthetic_pairs(object):
    '''
    embedding_object: an embedding object
    num_samples: how many pairwise comparisons to sample
    model_info:
                threshold_function = model_info['threshold_function']
                threshold_type = model_info['threshold_type']
                threshold = model_info['threshold']
                relative_flag = model_info['relative_flag']

    This class computes the bounds in Thm 1, stochastic transitivity violations, pairwise inconsistency rates, and given an estimated w_est, it computes the estimation error to the truth, the pairwise accuracy error, and the kendall tau correlation with the true ranking
    '''
    def __init__(self, embedding_object, model_info, get_sst = False):
        self.H = []

        self.U = embedding_object.U
        self.w = embedding_object.w
        self.model_info = model_info
        dim, num_items = self.U.shape
        self.num_items = num_items
        self.dim = dim
        self.P = np.zeros((self.num_items, self.num_items))
        self.inconsistent_pairs = []


    def get_parameters(self):
        '''
        This function computes all the relevant terms from Theorem 1 in the sample complexity result
        b, C, lambda, B, R, m_1, m_2, and the upper bound and builds P, the pairwise preference matrix
        '''

        b = 0 # max |w, U_i^{\tau(i,j) - U_j^{\tau(i,j)}|
        C = np.zeros([self.dim, self.dim]) # for computing C, second moment
        avg = np.zeros([self.dim,self.dim]) # for computing the min eigenvalue
        B = 0 # largest difference in the infinity norm
        num_pairs = (self.num_items)*(self.num_items-1) / 2
        R = 0 # for largest eigenvalue
        hs = []
        probs = []
        bs = []
        Bs = []
        Rs = []
        dif_in_probs = []

        for i in range(self.num_items):
            for j in range(i+1, self.num_items):
                x = np.copy(self.U[:, i])
                y = np.copy(self.U[:, j])
                w_proj = np.copy(self.w)

                threshold_function = self.model_info['threshold_function']
                threshold_type = self.model_info['threshold_type']
                threshold = self.model_info['threshold']
                relative_flag = self.model_info['relative_flag']

                h = preference_utils.evaluate_threshold(x, y, threshold_function, threshold_type, threshold, relative_flag)
                hs.append(np.copy(h))
                not_h = [idx for idx in range(self.dim) if idx not in h]
                dif = x-y
                dif[not_h] = 0
                # dif = dif / np.linalg.norm(dif)
                w_proj[not_h] = 0

                temp = np.linalg.norm(dif.reshape(self.dim,1), ord = np.inf)
                Bs.append(temp)
                if temp > B:
                    B = np.copy(temp)

                temp = np.dot(w_proj.T, dif.reshape(self.dim,1))[0][0]
                bs.append(temp)
                if np.abs(temp) > b :
                    b = np.copy(np.abs(temp))

                self.P[i,j] = 1/(1+np.exp(-temp))
                self.P[j,i] = 1/(1+np.exp(temp))

                full_prob = np.dot(self.w.T, (self.U[:,i] - self.U[:,j]).reshape(self.dim,1))[0][0]
                full_prob = 1/(1+np.exp(-full_prob))
                dif_in_probs.append(np.abs(self.P[i,j] - full_prob))
                dif_in_probs.append(np.abs(self.P[j,i] - (1-full_prob)))

                probs.append(self.P[i,j])
                probs.append(self.P[j,i])

                temp_c = dif.reshape(self.dim,1)@dif.reshape(self.dim,1).T
                avg += temp_c
                C += temp_c@temp_c

        avg = avg / num_pairs
        C = C / num_pairs

        # you have to compute the average first so that's why its like this
        ct = -1
        for i in range(self.num_items):
            for j in range(i+1, self.num_items):
                ct += 1

                x = np.copy(self.U[:, i])
                y = np.copy(self.U[:, j])
                w_proj = np.copy(self.w)

                threshold_function = self.model_info['threshold_function']
                threshold_type = self.model_info['threshold_type']
                threshold = self.model_info['threshold']
                relative_flag = self.model_info['relative_flag']

                h = hs[ct]
                not_h = [idx for idx in range(self.dim) if idx not in h]
                dif = x - y
                dif[not_h] = 0
                # dif = dif / np.linalg.norm(dif)
                w_proj[not_h] = 0

                temp = dif.reshape(self.dim,1)@dif.reshape(self.dim,1).T
                eigs, _ = np.linalg.eig(avg-temp)
                temp = np.max(eigs)
                Rs.append(temp)
                if temp > R:
                    R = temp

        weights = np.ones_like(dif_in_probs)/ float(len(dif_in_probs))
        plt.hist(dif_in_probs, weights = weights)
        plt.show()

        weights = np.ones_like(probs)/float(len(probs))
        plt.hist(probs, weights=weights)
        plt.show()
        #
        # weights = np.ones_like(bs)/float(len(bs))
        # plt.hist(bs, weights=weights)
        # plt.show()
        #
        # weights = np.ones_like(Bs)/float(len(Bs))
        # plt.hist(Bs, weights=weights)
        # plt.show()
        #
        # weights = np.ones_like(Rs)/float(len(Rs))
        # plt.hist(Rs, weights=weights)
        # plt.show()


        e_vals, _ = np.linalg.eig(avg)
        lam = np.min(e_vals)
        print('lambda is', lam)
        print('b is', b)
        print('R is', R)
        print('B is', B)
        C = np.linalg.norm(C - avg@avg, ord = 2)
        print('C is', C)
        m1 = (3*B**2*np.log(2*self.dim**2)*self.dim+4*B*np.log(2*self.dim**2)*np.sqrt(self.dim)) / 6
        print('m at least', m1)
        m2 = 8*np.log(self.dim)*(6*C + lam*R)/(3*lam**2)
        print('m at least', m2)
        print('with probability', 1- 2/self.dim)
        t = np.sqrt((3*B**2*np.log(2*self.dim**2)*self.dim+4*B*np.sqrt(self.dim)*np.log(2*self.dim**2))/(6))
        upper_bound = 4*(1+ np.exp(b))**2 / (np.exp(b)*lam)*t
        print('upper bound is', upper_bound)

        return lam, b, R, B, C, upper_bound, m1, m2

    def analyze_transitivity_and_pairwise_inconsistencies(self):
        num_valid_trips = 0
        weak_st_violations = 0
        moderate_st_violations = 0
        sst_violations = 0
        pairwise_inconsistencies = 0
        if np.sum(self.P) == 0:
            raise ValueError('The preference matrix P is all 0. Please run get_parameters')
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    continue
                i_utility = np.dot(self.w.T, self.U[:,i].reshape(self.dim,1))[0][0]
                j_utility = np.dot(self.w.T, self.U[:,j].reshape(self.dim,1))[0][0]
                if (self.P[i,j] - .5)*(i_utility - j_utility) < 0:
                    if j<i:
                        continue
                    pairwise_inconsistencies += 1
                    # print(self.P[i,j], 1/(1+np.exp(-i_utility + j_utility)), self.P[i,j] - 1/(1+np.exp(-i_utility + j_utility)))
                    self.inconsistent_pairs.append((i,j))
                for k in range(self.num_items):
                    if i == k or j == k:
                        continue

                    if self.P[i,j] >= .5 and self.P[j,k] >= .5:
                        num_valid_trips +=1

                        if self.P[i,k] < .5:
                            weak_st_violations += 1
                        if self.P[i,k] < max(self.P[i,j], self.P[j,k]):
                            sst_violations += 1
                        if self.P[i,k] < min(self.P[i,j], self.P[j,k]):
                            moderate_st_violations += 1
        return moderate_st_violations / num_valid_trips, sst_violations / num_valid_trips, weak_st_violations / num_valid_trips, pairwise_inconsistencies / (self.num_items*(self.num_items)/2)

    def get_comparison_matrix(self, num_comparisons):
        comparison_matrix = np.zeros((self.num_items, self.num_items))

        for _ in range(num_comparisons):
            # select the items
            i = np.random.randint(0, self.num_items)
            j = np.random.randint(0, self.num_items)
            while i == j:
                j = np.random.randint(0, self.num_items)

            prob = preference_utils.compute_probability(np.copy(self.U[:,i]), np.copy(self.U[:,j]), np.copy(self.w), self.model_info)
            if np.random.uniform(0,1) < prob:
                comparison_matrix[i,j] += 1
            else:
                comparison_matrix[j,i] += 1
        return comparison_matrix

    def get_w_est_error(self, w_est):
        return np.linalg.norm(w_est.T - self.w.T)

    def get_prediction_error(self, w_est, model_info, btl = False):
        correct = 0
        if np.sum(self.P) == 0:
            raise ValueError('The preference matrix P is all 0. Please run get_parameters')
        for i in range(self.num_items):
            for j in range(i+1, self.num_items):
                if btl:
                    est_prob = 1/(1+np.exp(-np.dot(w_est, self.U[:,i] - self.U[:,j])))
                else:
                    est_prob = preference_utils.compute_probability(np.copy(self.U[:,i]), np.copy(self.U[:,j]), np.copy(w_est), model_info)
                if (self.P[i,j] - .5)*(est_prob -.5)>0:
                    correct +=1
        print('prediction error', correct / (self.num_items*(self.num_items-1)/2))
        return correct / (self.num_items*(self.num_items-1)/2)

    def get_ranking_error_and_score_rsme(self, w_est):
        true_scores = (self.w.T@self.U)[0]

        est_scores = w_est.T@self.U

        return stats.kendalltau(true_scores, est_scores)[0], np.linalg.norm(true_scores - est_scores) / self.num_items
