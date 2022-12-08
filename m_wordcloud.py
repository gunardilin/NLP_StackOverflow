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
import os
import csv

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

def wordcloud_with_cluster(cluster_list, word_matrix, year):
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
        wc.to_file("other/{}_{}_cluster.png".format(year, i))
    # plt.show()
    return

def pipeline_normalizer(path:str, lexicon_path:str, year:int) -> pd.DataFrame:
    """ Perform Lemmatization and Vectorization.

    Args:
        path (str): SQL path
        lexicon_path (str): Gensim Lexicon path
        year (int): year 

    Returns:
        pd.DataFrame: tfidf dataframe with document id as index
    """
    corpus_reader = SqliteCorpusReader(path=path)
    docs = corpus_reader.docs(year)
    model = Pipeline([
        ("norm", TextNormalizer()),
        ("vect", GensimVectorizer(lexicon_path, False, True))
    ])
    tfidf = model.fit_transform(docs)
    ids = corpus_reader.ids(year)
    return pd.DataFrame(tfidf, index=ids)

def convert_to_list_of_list(list_:list):
    """Convert a list to list of list.
    Input = [1, 2, 3]
    Output = [[1], [2], [3]]
    """
    output_list = []
    for i in list_:
        output_list.append([i])
    return output_list

def write_to_csv(year, data:list):
    file = open('other/cluster_{}.csv'.format(year), 'w+', newline ='')
    # writing the data into the file
    converted_data = convert_to_list_of_list(data)
    with file:   
        write = csv.writer(file)
        write.writerows(converted_data)
    return

def pipeline_normalizer_wordcloud(path, lexicon_path, year):
    # Delete existing lexicon to keep the matrix as small as possible
    if os.path.exists(lexicon_path):
        os.remove(lexicon_path)
        print("Existing lexicon deleted...")
    else:
        print("No existing lexicon.")
    tfidf_matrix = pipeline_normalizer(path, lexicon_path, year)
    lexicon = gensim.corpora.Dictionary.load(lexicon_path)
    tfidf_matrix.columns = list(lexicon.token2id.keys())
    clusterer = KMeansClusters()
    cluster_list = clusterer.transform(tfidf_matrix)
    write_to_csv(year, cluster_list)
    wordcloud_with_cluster(cluster_list, tfidf_matrix, year)
    print("Finish executing pipeline_normalizer_workcloud")
    return
    
if __name__ == "__main__":
    start_time = time.time()
    PATH =  "DB/StackOverflow.sqlite"
    LEXICON_PATH = "other/lexicon.pkl"
    YEAR = 2021
    pipeline_normalizer_wordcloud(PATH, LEXICON_PATH, YEAR)
    timer(start_time, time.time())
    