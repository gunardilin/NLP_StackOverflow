from sklearn.pipeline import Pipeline
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import TextNormalizer, GensimVectorizer, GensimVectorizer_Topic_Discovery
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF

from gensim.models import LsiModel, LdaModel
from o_ldamodel import LdaTransformer
from o_lsimodel import LsiTransformer
from gensim.corpora.dictionary import Dictionary
from gensim import corpora

import time
from l_pipeline import timer

import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models

import os
import warnings
import sys
import pickle
import itertools

class SklearnEstimator(object):
    def __init__(self, n_topics=50, estimator="LDA"):
        self.n_topics = n_topics
        self.documents = None

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_topics)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_components=self.n_topics)
    
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        self.documents = list(documents)
        self.estimator.fit_transform(self.documents)
        
class SklearnTopicModels(object):
    def __init__(self, n_topics=20, estimator="LDA", \
        db_path:str="DB/StackOverflow.sqlite", \
        gensim_lexicon:str="other/lexicon.pkl"):
        self.n_topics = n_topics
        self.corpus_reader = SqliteCorpusReader(path=db_path)
        self.model = Pipeline([
            ("norm", TextNormalizer()),
            ("vect", GensimVectorizer(gensim_lexicon, False, True)),
            ("model", SklearnEstimator(self.n_topics, estimator)),
        ])
        self.estimator_model = None
    
    def fit_transform(self, year:int):
        docs = self.corpus_reader.docs(year)
        self.model.fit_transform(docs)
        return self.model
    
    def get_topics(self, n_words=25):
        vectorizer = self.model.named_steps['vect']
        # Get object Estimator
        self.estimator_model = self.model.steps[-1][1]
        # Get a dict with id as index and token as value: 
        names = vectorizer.id2word.id2token
        topics = dict()
        
        for idx, topic in enumerate(self.estimator_model.estimator.components_):
            # topic is an array of token weights
            # In next line we get the indices that would sort an array.
            # The indices would be starting from token with most weight only
            # as much as n_words.
            features = topic.argsort()[:-(n_words-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
        return topics
    
    def visualize_topic(self):
        # lda_tf -> self.estimator_model
        # dtm_tf -> self.model.steps[-1][1].documents
        # tf_vectorizer -> self.model.steps[-2][1]
        pyLDAvis.sklearn.prepare(self.estimator_model.estimator,\
            self.model.steps[-1][1].documents, \
            self.model.named_steps['vect'])
        # pyLDAvis.gensim_models.prepare()
        return

class GensimTopicModels(object):
    def __init__(self, n_topics=20, estimator="LDA", \
        db_path:str="DB/StackOverflow.sqlite", \
        gensim_lexicon:str="other/model/lexicon.pkl"):
        self.lexicon_path = gensim_lexicon
        self.estimator_str = estimator
        self.corpus_reader = SqliteCorpusReader(path=db_path)     
        self.n_topics = n_topics
        self.docs = []
        self.doc_matrix = None
        self.id2word = None
        self.model_folderpath = "other/model"
        self.model_filepath = "{}/{}_model".format(self.model_folderpath, estimator)
        self.tempFolderPath = "other/temp"
        self.doc_matrix_pickle_path = "{}/temp.pickle".format(self.tempFolderPath)
        self.remove_temp([self.doc_matrix_pickle_path, gensim_lexicon])

        if estimator == 'LSA':
            self.estimator = LsiTransformer(num_topics=self.n_topics)
        else:
            self.estimator = LdaTransformer(num_topics=self.n_topics)
        # self.load_model()
        
        self.model = Pipeline([
            ("norm", TextNormalizer()),
            ("vect", GensimVectorizer_Topic_Discovery(gensim_lexicon, False, True)),
        ])
    
    def save_model(self):
        if not os.path.exists(self.model_folderpath):
            os.makedirs(self.model_folderpath)
        self.estimator.gensim_model.save(self.model_filepath)
        return
    
    def load_model(self):
        if os.path.exists(self.model_filepath):
            if self.estimator == "LDA":
                self.estimator.gensim_model = LdaModel.load(self.model_filepath)
            elif self.estimator == "LSI":
                self.estimator.gensim_model = LsiModel.load(self.model_filepath)
        return
    
    def fit(self, year:int):
        docs = self.corpus_reader.docs(year)
        self.docs = list(docs)
        self.doc_matrix = self.model.fit_transform(self.docs)
        self.docs = None                # Free up memory
        vectorizer = self.model.named_steps['vect']
        vectorizer.documents = None     # Free up memory
        self.estimator.id2word = vectorizer.id2word.id2token
        self.estimator.partial_fit(self.doc_matrix)
        self.save_model()
        return self.model
    
    def fit_multi_years(self, start_year:int, end_year:int):
        for year in range(start_year, end_year+1):
            print("Reading corpus from Y{}".format(year))
            docs = self.corpus_reader.docs(year)
            self.docs += list(docs)
        print("Variable size: {}".format(sys.getsizeof(self.docs)))
        del docs                        # Free up memory
        self.doc_matrix = self.model.fit_transform(self.docs)
        self.docs = None                # Free up memory
        vectorizer = self.model.named_steps['vect']
        vectorizer.documents = None     # Free up memory
        self.estimator.id2word = vectorizer.id2word.id2token
        self.estimator.partial_fit(self.doc_matrix)
        self.save_doc_matrix_to_pickle(self.doc_matrix)
        self.doc_matrix = None
        self.save_model()
        return self.model
    
    def save_doc_matrix_to_pickle(self, data):
        if not os.path.exists(self.tempFolderPath):
            os.makedirs(self.tempFolderPath)
        with open(self.doc_matrix_pickle_path,"ab") as f:
            pickle.dump(data, f)
        print("Data is saved into pickle.")
        return
    
    def load_from_pickle(self, path):
        def loader(fp):
            # a load iter
            while True:
                try:
                    yield pickle.load(fp)
                except EOFError:
                    print("Generator is exhausted")
                    break
        with open(path, "rb") as f:
            for i in loader(f):
                yield(i)
        return
    
    def remove_temp(self, path_list:list):
        for filePath in path_list:
            if os.path.exists(filePath):
                os.remove(filePath)
        return
    
    def get_topics(self, n_words=25):
        names = self.estimator.id2word
        topics = dict()
        for idx, topic in enumerate (self.estimator.gensim_model.get_topics()):
            features = topic.argsort()[:-(n_words-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
        return topics
    
    # def visualize_topics(self):
    #     if self.estimator_str != "LDA":
    #         print("** The pyLDAvis is only compatible for LDA not the currently used model.")
    #         return
    #     lda_model = self.estimator.gensim_model
    #     temp_list = self.load_from_pickle(self.doc_matrix_pickle_path)
        
    #     corpus = []
    #     for i in temp_list:
    #         corpus += i
    #     del temp_list   # Free up memory
    #     lexicon = Dictionary()
    #     lexicon.token2id = self.estimator.id2word
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         data = pyLDAvis.gensim_models.prepare(lda_model, corpus, lexicon)
    #         pyLDAvis.save_html(data, 'other/lda.html')
    #     return data
    
    def visualize_topics(self):
        if self.estimator_str != "LDA":
            print("** The pyLDAvis is only compatible for LDA not the currently used model.")
            return
        lda_model = self.estimator.gensim_model
        temp_list = self.load_from_pickle(self.doc_matrix_pickle_path)
        corpus = []
        for i in temp_list:
            corpus += i
        #lexicon = self.model.named_steps['vect'].id2word
        #lexicon = lda_model.id2word
        lexicon = Dictionary.load(self.lexicon_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = pyLDAvis.gensim_models.prepare(lda_model, corpus, lexicon)
            pyLDAvis.save_html(data, 'other/lda.html')
        return data
        
        

if __name__ == "__main__":
    
    ## With Sklearn
    # start_time = time.time()
    # model = SklearnTopicModels(n_topics=50, estimator="LDA")
    # model.fit_transform(2022)
    # topics = model.get_topics()
    # for topic, term in topics.items():
    #     print("Topic #{}:".format(topic+1))
    #     print(term)  
    # # model.visualize_topic()
    # timer(start_time, time.time())
    
    
    ## With Gensim for single year
    # start_time = time.time()
    # model = GensimTopicModels(n_topics=50, estimator="LDA")
    # model.fit(2022)
    # # print(model.estimator.gensim_model.print_topics(10))
    # topics = model.get_topics()
    # n = 0
    # for topic in topics.values():
    #     n += 1
    #     print("Topic #{}:".format(n))
    #     print(topic)
    # model.visualize_topics()
    # timer(start_time, time.time())
    

    ## With Gensim for multi years
    start_time = time.time()
    model = GensimTopicModels(n_topics=50, estimator="LDA")
    model.fit_multi_years(start_year=2011, end_year=2021)
    # print(model.estimator.gensim_model.print_topics(10))
    topics = model.get_topics()
    n = 0
    for topic in topics.values():
        n += 1
        print("Topic #{}:".format(n))
        print(topic)
    model.visualize_topics()
    timer(start_time, time.time())
