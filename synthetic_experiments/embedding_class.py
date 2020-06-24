import numpy as np

'''
This class generates an embedding of items and judgement vector.

The item embeddings are sampled two different ways.
_generate_basic_embedding(self, num_items, dim): Each item's coordinates are sampled from N(0, 1/dim). Each judge's coordinate are sampled from N(0,4/dim)
_generate_clustered_embedding(self, num_items, dim, num_clusters): First num_clusters mean vectors are drawn from N(0, 1/d). Then to get the embedding for each item, a center is sampled \mu, and then the item embedding is drawn from N(\mu, 1/d).
'''

class embedding(object):
    def __init__(self, num_items, dim, num_clusters, std_deviation):
        '''
        num_items: how many items
        dim: embedding dimension
        num_clusters: how many clusters to sample from
        '''
        if num_clusters == 0:
            U, w = self._generate_basic_embedding(num_items, dim, std_deviation)
        elif num_clusters == -1:
            U, w = self._generate_orthognal_embedding(num_item)
        else:
            U, w = self._generate_clustered_embedding(num_items, dim, num_clusters)
        self.U = U
        self.w = w

    def _generate_basic_embedding(self, num_items, dim, std_deviation):
        U = np.random.normal(0, scale=std_deviation, size = (dim,num_items))
        w = 2*np.random.normal(0, scale=1/np.sqrt(dim), size = (dim,1))
        # w = w / np.linalg.norm(w)
        # w = np.random.uniform(0, 1, size = (dim,1))
        # U = np.random.uniform(0, std_deviation, size = (dim,num_items))
        # w = np.random.uniform(0, np.sqrt(dim), size = (dim,1))

        return U, w

    def _generate_clustered_embedding(self, num_items, dim, num_clusters):
        U = np.zeros((dim,num_items))
        w = np.random.normal(0, scale=1/np.sqrt(dim), size = (dim,1))
        w = w / np.linalg.norm(w)

        cov = (1/np.sqrt(dim))*np.eye(dim,dim)

        clust_to_center = [1/ np.sqrt(np.sqrt(dim))*np.random.randn(dim, 1).reshape(dim,1).T[0] for i in range(num_clusters)]

        for i in range(num_items):
            clust_mean = clust_to_center[np.random.randint(0,num_clusters)]
            U[:, i] = np.random.multivariate_normal(clust_mean, cov).T

        return U, w
