#!/usr/bin/env python3

from b_sqlite_operation import SqliteOperation
import logging

from readability.readability import Unparseable
from readability.readability import Document as Paper

import bs4

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']

class SQLiteHTML_Connector(SqliteOperation):
    """Connector between SqliteOperation and HTMLCorpusReader.
    Case specific class.

    Args:
        SqliteOperation (Class): Class that contains SQLite basic operations.
    """
    def __init__(self, path: str = "DB/StackOverflow.sqlite", batchsize: int = 50, \
        tablename: str = "preprocessed_datas", sql_from_value: str = "content"):
        SqliteOperation.__init__(self, path, batchsize)
        self.tablename = tablename  # # SELECT ... FROM (...)
        self.sql_from_value = sql_from_value # SELECT (...) FROM ...
    
    def get_preprocessed_datas(self, limit: int = None):
        iterable_row = self.read_from_db(self.tablename, self.sql_from_value, \
            limit=limit)
        return iterable_row
    
    def html_connector(self, limit: int = None):
        return self.html(self.get_preprocessed_datas(10))

class HTMLCorpusReader(SQLiteHTML_Connector):
    def __init__(self, limits:int=None, tags:list = TAGS):
        SQLiteHTML_Connector.__init__(self)
        self.preprocessed_datas = self.get_preprocessed_datas(limits)
        # Save the tags that we specifically want to extract.
        self.tags = tags
    
    def html(self):
        """
        Returns the HTML content of each document, cleaning it using
        the readability-lxml library.
        """
        for i in self.preprocessed_datas:
            try:
                # i is a tuple therefore to get its content, i[0] will be used 
                # in next line.
                yield Paper(i[0]).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue
    
    def paras(self):
        """
        Uses BeautifulSoup to parse the paragraphs from the HTML.
        """
        for html in self.html():
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(self.tags):
                yield element.text
            soup.decompose()
    
    
    def sents(self):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras():
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents():
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras():
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]
    
    def process(self):
        return

if __name__ == "__main__":
    sqlite_handler = HTMLCorpusReader(limits=10)
    # for i in sqlite_handler.preprocessd_datas:
    #     print(i[0])
    #     print(sqlite_handler.html_single(i[0]))
    
    iter_result = sqlite_handler.tokenize()
    for i in iter_result:
        print(i)
    print("finished")