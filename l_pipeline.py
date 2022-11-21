from sklearn.pipeline import Pipeline
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import TextNormalizer, GensimVectorizer
from k_clustering import KMeansClusters

def pipeline_normalizer_kmeans(path:str, year:int):
    corpus_reader = SqliteCorpusReader(path=path)
    docs = corpus_reader.docs(year)
    
    model = Pipeline([
        ("norm", TextNormalizer()),
        ("vect", GensimVectorizer("other/lexicon.pkl", True)),
        ("clusters", KMeansClusters(k=7))
    ])
    
    clusters = model.fit_transform(docs)
    ids = corpus_reader.ids(year)
    
    for idx, cluster in enumerate(clusters):
        print("Post {} assigned to cluster {}.".format(ids[idx][0], cluster))

if __name__ == "__main__":
    PATH =  "DB/StackOverflow.sqlite"
    
    pipeline_normalizer_kmeans(PATH, 2022)