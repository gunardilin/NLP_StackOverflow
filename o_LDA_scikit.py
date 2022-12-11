from sklearn.pipeline import Pipeline
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import TextNormalizer, GensimVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF

import time
from l_pipeline import timer

class Estimator(object):
    def __init__(self, n_topics=50, estimator="LDA"):
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_topics)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_components=self.n_topics)
    
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        self.estimator.fit_transform(list(documents))
        
class SklearnTopicModels(object):
    def __init__(self, n_topics=50, estimator="LDA", \
        db_path:str="DB/StackOverflow.sqlite", \
        gensim_lexicon:str="other/lexicon.pkl"):
        self.n_topics = n_topics
        self.corpus_reader = SqliteCorpusReader(path=db_path)
        self.model = Pipeline([
            ("norm", TextNormalizer()),
            ("vect", GensimVectorizer(gensim_lexicon, False, True)),
            ("model", Estimator(self.n_topics, estimator))
        ])
    
    def fit_transform(self, year:int):
        docs = self.corpus_reader.docs(year)
        self.model.fit_transform(docs)
        return self.model
    
    def get_topics(self, n_words=25):
        vectorizer = self.model.named_steps['vect']
        # Get object Estimator
        model = self.model.steps[-1][1]
        # Get a dict with id as index and token as value: 
        names = vectorizer.id2word.id2token
        topics = dict()
        
        for idx, topic in enumerate(model.estimator.components_):
            # topic is an array of token weights
            # In next line we get the indices that would sort an array.
            # The indices would be starting from token with most weight only
            # as much as n_words.
            features = topic.argsort()[:-(n_words-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
            
        return topics

if __name__ == "__main__":
    start_time = time.time()
    skmodel = SklearnTopicModels(n_topics=50, estimator="NMF")
    skmodel.fit_transform(2022)
    topics = skmodel.get_topics()
    for topic, term in topics.items():
        print("Topic #{}:".format(topic+1))
        print(term)  
    timer(start_time, time.time())
