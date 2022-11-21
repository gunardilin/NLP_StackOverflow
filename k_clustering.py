from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import types

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