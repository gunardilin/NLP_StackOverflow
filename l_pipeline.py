from sklearn.pipeline import Pipeline
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import TextNormalizer, GensimVectorizer
from k_clustering import KMeansClusters, HierarhicalClusters
import pandas as pd
import xlwings as xw
import time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Runtime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def pipeline_normalizer_kmeans(path:str, year:int):
    corpus_reader = SqliteCorpusReader(path=path)
    docs = corpus_reader.docs(year)
    
    model = Pipeline([
        ("norm", TextNormalizer()),
        ("vect", GensimVectorizer("other/lexicon.pkl", False, True)),
        ("clusters", KMeansClusters(k=7))
    ])
    
    clusters = model.fit_transform(docs)
    ids = corpus_reader.ids(year)
    
    ### For testing purpose:
    # for idx, cluster in enumerate(clusters):
    #     print("Post {} assigned to cluster {}.".format(ids[idx], cluster))
    print("Total: {} posts.".format(len(ids)))
    ###
    return pd.DataFrame(clusters, index=ids, columns=['cluster'])

# if __name__ == "__main__":
#     start_time = time.time()
#     PATH =  "DB/StackOverflow.sqlite"
    
#     cluster = pipeline_normalizer_kmeans(PATH, 2021)
#     timer(start_time, time.time())
#     xw.view(cluster)
#     timer(start_time, time.time())
#     print("Finish")

def pipeline_normalizer_agglomerative(path:str, year:int):
    corpus_reader = SqliteCorpusReader(path=path)
    docs = corpus_reader.docs(year)
    
    clusterer = HierarhicalClusters()
    model = Pipeline([
        ("norm", TextNormalizer()),
        ("vect", GensimVectorizer("other/lexicon.pkl", False, True)),
        ("clusters", clusterer)
    ])
    
    model.fit_transform(docs)
    ids = corpus_reader.ids(year)
    clusterer.plot_dendrogram(ids)

if __name__ == "__main__":
    start_time = time.time()
    PATH =  "DB/StackOverflow.sqlite"
    
    cluster = pipeline_normalizer_agglomerative(PATH, 2022)
    timer(start_time, time.time())
    print("Finish")