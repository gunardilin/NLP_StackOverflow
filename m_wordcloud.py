import matplotlib.pyplot as plt
from wordcloud import WordCloud

from l_pipeline import pipeline_normalizer_kmeans, timer
import time
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import TextNormalizer, GensimVectorizer
from k_clustering import KMeansClusters
from sklearn.pipeline import Pipeline
import pandas as pd

import gensim

WIDTH = 800
HEIGHT = 400
def wordcloud(word_matrix):
    wordcloud = WordCloud(background_color="white", max_words=50, width=WIDTH, \
        height=HEIGHT)
    wc = wordcloud.generate_from_frequencies(word_matrix)
    plt.figure(figsize=(8,6))
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.tight_layout(pad=0)
    plt.show()

def wordcloud_with_cluster(cluster_list, word_matrix):
    cluster_set = sorted(set(cluster_list))
    for i in cluster_set:
        index_ = [a for a, x in enumerate(cluster_list) if x == i]
        temp_wordmatrix = word_matrix.iloc[index_] * 100
        temp_wordmatrix = temp_wordmatrix.sum(axis=0)/len(temp_wordmatrix)
        wc = WordCloud(background_color="white", max_words=50, width=800, \
            height=400).generate_from_frequencies(temp_wordmatrix)
        plt.figure(figsize=(8,6))
        plt.axis("off")
        plt.imshow(wc, interpolation="bilinear")
        plt.tight_layout(pad=0)
    plt.show()
    return
    

def pipeline_normalizer(path:str, year:int):
    corpus_reader = SqliteCorpusReader(path=path)
    docs = corpus_reader.docs(year)
    
    model = Pipeline([
        ("norm", TextNormalizer()),
        ("vect", GensimVectorizer("/Users/GunardiLin/Desktop/Project/ProjectStackOverflow/Python/other/lexicon.pkl", False, True))
    ])
    
    tfidf = model.fit_transform(docs)
    ids = corpus_reader.ids(year)
    return pd.DataFrame(tfidf, index=ids)

if __name__ == "__main__":
    start_time = time.time()
    PATH =  "DB/StackOverflow.sqlite"
    LEXICON_PATH = "/Users/GunardiLin/Desktop/Project/ProjectStackOverflow/Python/other/lexicon.pkl"
    YEAR = 2022
    tfidf_matrix = pipeline_normalizer(PATH, YEAR)
    lexicon = gensim.corpora.Dictionary.load(LEXICON_PATH)
    tfidf_matrix.columns = list(lexicon.token2id.keys())
    clusterer = KMeansClusters()
    cluster_list = clusterer.transform(tfidf_matrix)
    wordcloud_with_cluster(cluster_list, tfidf_matrix)
    # timer(start_time, time.time())
    # xw.view(cluster)
    # timer(start_time, time.time())
    print("Finish")