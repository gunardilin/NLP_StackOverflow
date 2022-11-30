import nltk
import string
from collections.abc import Generator
from sklearn.base import BaseEstimator, TransformerMixin
import os
import gensim
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
import unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import types

def tokenize(text:str, language:str="english")-> Generator[str]:
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
    
    def __init__(self, language='english'):
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
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]
    
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
    def __init__(self, path:str=None, binary=False, tfidf=False):
        "path: save location for Dictionary after performing self.fit(...)"
        "Change binary to True to activate One Hot Encoding."
        self.path = path
        self.id2word = None
        self.load()
        self.binary = binary
        self.tfidf = tfidf
        self.tfidf_model = None
        self.documents = None
        
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
        self.id2word.filter_extremes(no_below=20)
        self.save()
        return self
    
    def transform(self, documents:list) -> Generator[np.ndarray]:
        """
        Transforming document vectors so that every document contains exactly 
        "len(self.id2word)" length.
        """
        if self.tfidf:
            tfidf_model = gensim.models.TfidfModel(dictionary=self.id2word, \
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
                docvec = tfidf_model[docvec_temp]

            yield sparse2full(docvec, len(self.id2word))