from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import types
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering
class KMeansClusters(BaseEstimator, TransformerMixin):
    def __init__(self, k:int=7):
        """_summary_

        Args:
            k (int, optional): Number of clusters. Defaults to 7.
        """
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance, \
            avoid_empty_clusters=True)
    
    def fit(self, documents, labels=None):
        return self
        
    def transform(self, documents:list):
        """Fits documents to the K-Means model.

        Args:
            documents (list): List that contains document vectors (frequency-, \
                one hot- or tfidf vectors).
        """
        if isinstance(documents, types.GeneratorType):
            documents_ = list(documents)
        else:
            documents_ = documents
        return self.model.cluster(documents_, assign_clusters=True)
    
class HierarhicalClusters(object):
    def __init__(self):
        self.model = AgglomerativeClustering()
        self.labels = None
        self.children = None
    
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents:list):
        """Fit the agglomerative models to the given data.

        Args:
            documents (list): List that contains document vectors.
        """
        if isinstance(documents, types.GeneratorType):
            documents_ = list(documents)
        clusters = self.model.fit_predict(documents_)
        self.labels = self.model.labels_
        self.children = self.model.children_
        return clusters
    
    def plot_dendrogram(self, ids):
        # Distance between each pair of children
        distance = np.arange(self.children.shape[0])
        position = np.arange(self.children.shape[0])
        
        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([self.children, distance, position]).\
            astype(float)
        
        # Plot the corresponding dendrogram
        fig, ax = plt.subplots(figsize=(10, 5)) # set size
        ax = dendrogram(linkage_matrix, labels=ids)
        plt.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
        plt.xticks(fontsize=14)
        plt.tight_layout()
        plt.show()