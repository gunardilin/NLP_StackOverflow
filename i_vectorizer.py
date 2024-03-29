import nltk
import string
from collections.abc import Generator
from sklearn.base import BaseEstimator, TransformerMixin
import os
import gensim
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import types

from sklearn.feature_extraction.text import TfidfVectorizer

NO_BELOW = 20
LANGUAGE = "english"
THRESHOLD_NGRAM = 1

def tokenize(text:str, language:str=LANGUAGE)-> Generator[str]:
    """Remove affixes, e.g. plurality, -"ing", -"tion", etc.

    Args:
        text (str): Input string

    Yields:
        Generator[str]: Generator that contain list of strings. This generator \
            can be unpacked either with 1) converting to list or 2) unpack with\
            '*' operator.
    """
    stem = nltk.stem.SnowballStemmer(language)
    text = text.lower()
    
    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue
        yield stem.stem(token)
class TextNormalizer(BaseEstimator, TransformerMixin):
    """This class performs lemmatization."""
    
    def __init__(self, language=LANGUAGE):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
    def is_punct(self, token):
        """ Detecting punctuations: .,!/? and more..."""
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )
    
    def is_stopword(self, token):
        """ Detecting stopwords: a, in, of and more..."""
        return token.lower() in self.stopwords
    
    def is_one_character(self, token):
        """ Detecting one character token: a, b, 1, 2, ..."""
        return len(token)==1
    
    def lemmatize(self, token:str, pos_tag:str) -> str:
        """ Perform lemmatization.
        Examples of lemmatization:
        -> rocks : rock
        -> corpora : corpus
        -> better : good

        Args:
            token (str): word token
            pos_tag (str): Penn Treebank tag

        Returns:
            str: lemmatized word
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)
    
    def normalize(self, document):
        normalized_list = []
        temp_result = None
        for paragraph in document:
            for sentence in paragraph:
                for (token, tag) in sentence:
                    if not self.is_punct(token) and not self.is_stopword(token) \
                        and not self.is_one_character(token):
                            temp_result = self.lemmatize(token, tag).lower()
                            if len(temp_result) > 1:
                                normalized_list.append(temp_result)
        return normalized_list
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, documents):
        """This function lowers all capital letter, lemmatizers all words.
        Output: 
        ['spring', 'security', '+', 'jwt', 'authenticate', 'via', 'header', 'send', 'jwt', 'trouble', 'try', 'get', 'spring', 'back', ...]
        """
        for document in documents:
            yield self.normalize(document)

class GensimVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizing each document when self.transform is executed.
    Reason for using Gensim: Gensim Dictionary can be saved to disk.
    It allows reloading the Dictionary without requiring a refit.
    
    It could perform either 1) Frequency vectorization or 2) One Hot Encoding
    or 3) TF-IDF
    1) Frequency vectorization: binary=False, tfidf=True/False
    2) One Hot Encoding: binary=True, tfidf=True/False
    3) TF-IDF: binary=False, tfidf=True
    By initializing self.binary=True, One Hot Encoding will be used.
    If self.binary=False, Frequency Vectorization is active. 
    """
    def __init__(self, path:str=None, binary=False, tfidf=False, no_below=NO_BELOW):
        "path: save location for Dictionary after performing self.fit(...)"
        "Change binary to True to activate One Hot Encoding."
        self.path = path
        self.id2word = None
        self.load()
        self.binary = binary
        self.tfidf = tfidf
        self.tfidf_model = None
        self.documents = None
        self.model = None
        self.no_below = no_below
        
    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)
            
    def save(self):
        self.id2word.save(self.path)
        
    def fit(self, documents):
        """
        This method constructs the Dictionary obj by passing already 
        tokenized & normalized documents to the Dictionary constructor.
        Then the Dictionary is saved to disk, so that the transformer
        can be loaded without requiring a refit.
        """
        if isinstance(documents, types.GeneratorType):
            self.documents = list(documents)
        else:
            self.documents = documents
        self.id2word = Dictionary(self.documents)
        self.id2word.filter_extremes(no_below=self.no_below)
        self.save()
        return self
    
    def transform(self, documents:list) -> Generator[np.ndarray]:
        """
        Transforming document vectors so that every document contains exactly 
        "len(self.id2word)" length.
        """
        if self.tfidf:
            self.model = gensim.models.TfidfModel(dictionary=self.id2word, \
                normalize=True)
        for document in self.documents:
            # Applying frequency vectorization:
            docvec = self.id2word.doc2bow(document) # Mode1: Frequency Vectorization
            if self.binary: # Mode2: If True: One Hot Encoding.
                temp = []
                for index_, frequency in docvec:
                    if frequency > 0:
                        onehot_freq = 1
                    else:
                        onehot_freq = 0
                    temp.append((index_, onehot_freq))
                docvec = temp
            elif self.tfidf: # Mode3: If True: TF-IDF
                docvec_temp = docvec
                docvec = self.model[docvec_temp]

            yield sparse2full(docvec, len(self.id2word))

class NgramVectorizer(BaseEstimator, TransformerMixin):
    """Create ngram phrases from corpus.
    """
    def __init__(self, min_count=NO_BELOW, threshold=10, bi_trigram="b"):
        self.min_count = min_count
        self.threshold = threshold
        self.bi_trigram = bi_trigram
        self.documents = None
        if LANGUAGE == "english":
            self.connector_words = ENGLISH_CONNECTOR_WORDS
        else:
            raise Exception("Not compatible with non english language.")
    
    def fit(self, documents):
        return self
    
    def transform(self, documents):
        if isinstance(documents, types.GeneratorType):
            self.documents = list(documents)
        else:
            self.documents = documents
        bigram = Phrases(self.documents, min_count=20, \
            threshold=THRESHOLD_NGRAM)
        if self.bi_trigram == "b":
            # For Bigram:
            for idx in range(len(self.documents)):
                for token in bigram[self.documents[idx]]:
                    if '_' in token:
                        # Token is a bigram+trigram, add to document.
                        self.documents[idx].append(token)
                yield self.documents[idx]
                
        elif self.bi_trigram == "t":
            # For Trigram:
            trigram = Phrases(bigram[self.documents], min_count=20, \
                threshold=THRESHOLD_NGRAM)
            for idx in range(len(self.documents)):
                for token in trigram[bigram[self.documents[idx]]]:
                    if '_' in token:
                        # Token is a bigram+trigram, add to document.
                        self.documents[idx].append(token)
                yield self.documents[idx]
    
class GensimVectorizer_Topic_Discovery(BaseEstimator, TransformerMixin):
    """Vectorizing each document when self.transform is executed.
    Reason for using Gensim: Gensim Dictionary can be saved to disk.
    It allows reloading the Dictionary without requiring a refit.
    
    It could perform either 1) Frequency vectorization or 2) One Hot Encoding
    or 3) TF-IDF
    1) Frequency vectorization: binary=False, tfidf=True/False
    2) One Hot Encoding: binary=True, tfidf=True/False
    3) TF-IDF: binary=False, tfidf=True
    By initializing self.binary=True, One Hot Encoding will be used.
    If self.binary=False, Frequency Vectorization is active. 
    """
    def __init__(self, path:str=None, binary=False, tfidf=False, no_below=NO_BELOW):
        "path: save location for Dictionary after performing self.fit(...)"
        "Change binary to True to activate One Hot Encoding."
        self.path = path
        self.id2word = None
        self.load()
        self.binary = binary
        self.tfidf = tfidf
        self.tfidf_model = None
        self.documents = None
        self.model = None
        self.no_below = no_below
        
    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)
            
    def save(self):
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        self.id2word.save(self.path)
    
    # def get_feature_names(self):
    #     return list(self.id2word.values())
        
    def fit(self, documents):
        """
        This method constructs the Dictionary obj by passing already 
        tokenized & normalized documents to the Dictionary constructor.
        Then the Dictionary is saved to disk, so that the transformer
        can be loaded without requiring a refit.
        """
        if isinstance(documents, types.GeneratorType):
            self.documents = list(documents)
        else:
            self.documents = documents
        if self.id2word == None:
            self.id2word = Dictionary(self.documents)
        else:
            self.id2word.add_documents(self.documents)
        self.id2word.filter_extremes(no_below=self.no_below)
        self.save()
        return self
    
    def transform(self, documents:list) -> list:
        """
        Transforming document vectors so that every document contains exactly 
        "len(self.id2word)" length.
        """
        if self.tfidf:
            self.model = gensim.models.TfidfModel(dictionary=self.id2word, \
                normalize=True)
        def generator():
            for document in self.documents:
                # Applying frequency vectorization:
                docvec = self.id2word.doc2bow(document) # Mode1: Frequency Vectorization
                if self.binary: # Mode2: If True: One Hot Encoding.
                    temp = []
                    for index_, frequency in docvec:
                        if frequency > 0:
                            onehot_freq = 1
                        else:
                            onehot_freq = 0
                        temp.append((index_, onehot_freq))
                    docvec = temp
                elif self.tfidf: # Mode3: If True: TF-IDF
                    docvec_temp = docvec
                    docvec = self.model[docvec_temp]

                yield docvec
        return list(generator())