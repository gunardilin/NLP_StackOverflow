from m_wordcloud import pipeline_normalizer
import os
from k_clustering import KMeansClusters
import gensim
import numpy as np
import matplotlib.pyplot as plt
import time
from l_pipeline import timer

from sklearn.metrics.pairwise import pairwise_distances

PATH =  "DB/StackOverflow.sqlite"
LEXICON_PATH = "other/lexicon.pkl"

def elbow_method(path:str, lexicon_path:str, year:int, end_year:int=None,\
    max_clusters:int=10, metric="cosine"):
    start_time = time.time()
    # Delete existing lexicon to keep the matrix as small as possible
    if os.path.exists(lexicon_path):
        os.remove(lexicon_path)
        print("Existing lexicon deleted...")
    else:
        print("No existing lexicon.")
    print("Generating tfidf matrix")
    tfidf_matrix = pipeline_normalizer(path, lexicon_path, year)
    lexicon = gensim.corpora.Dictionary.load(lexicon_path)
    tfidf_matrix.columns = list(lexicon.token2id.keys())
    # Initiate cluster range
    cluster_range = range(2, max_clusters+1)
    sum_squared_error_for_n_cluster = {}
    for i in cluster_range:
        print("Calculating SSE for {} clusters...".format(str(i)))
        temp_time = time.time()
        clusterer = KMeansClusters(k=i)
        cluster_list = clusterer.transform(tfidf_matrix)
        # A.1 Initialize dict for centroid indexing
        cluster_index = {}
        unique_cluster = list(set(cluster_list))
        for centroid in unique_cluster:
            cluster_index[centroid] = []
        # A.2 Insert positions of centroid into the dict
        for n, x in enumerate(cluster_list):
            cluster_index[x].append(n)
        # B.1 Get all centroid coordinates
        centroids = clusterer.get_centroids()
        # B.2 Calculate sum squared error (SSE) for each centroid
        temp_list = []
        for cluster_x in unique_cluster:
            temp_array = pairwise_distances([centroids[cluster_x]], \
                tfidf_matrix.iloc[cluster_index[cluster_x]].to_numpy(), metric=metric)
            temp_list.append(np.sum(temp_array))
            
        sum_squared_error_for_n_cluster[i] = sum(temp_list) / 2
        # Divided by 2 because the elbow matrix is reflected at its diagonal.
        print("**Cluster:")
        timer(temp_time, time.time())
        timer(start_time, time.time())
    print("Generating elbow method graph...")
    plt.style.use("fivethirtyeight")
    plt.plot(sum_squared_error_for_n_cluster.keys(), \
        sum_squared_error_for_n_cluster.values())
    plt.xticks(list(sum_squared_error_for_n_cluster.keys()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    plt.savefig('other/elbow_methode_{}_{}_clusters.png'.format(year, max_clusters),\
        bbox_inches='tight', dpi=100)
    plt.show()
    
    return

if __name__ == "__main__":
    elbow_method(PATH, LEXICON_PATH, 2022, max_clusters=50)