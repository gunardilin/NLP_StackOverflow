#!/usr/bin/env python3

from b_sqlite_operation import SqliteOperation
import logging

from readability.readability import Unparseable
from readability.readability import Document as Paper

import bs4

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

import json

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']

class SQLiteHtmlJson_Connector(SqliteOperation):
    """Connector among SqliteOperation,  HTMLCorpusReader and Json operations.
    Case specific class.

    Args:
        SqliteOperation (Class): Class that contains SQLite basic operations.
    """
    def __init__(self, path: str = "/Users/GunardiLin/Desktop/Project/ProjectStackOverflow/Python/DB/StackOverflow.sqlite", batchsize: int = 50, \
        tablename: str = "preprocessed_datas", sql_from_value: str = "id, content"):
        SqliteOperation.__init__(self, path, batchsize)
        self.tablename = tablename  # # SELECT ... FROM (...)
        self.sql_from_value = sql_from_value # SELECT (...) FROM ...
    
    def get_preprocessed_datas(self, limit: int = None):
        # Use get_preprocessed_datas_batchwise to read DB in memory friendly way.
        iterable_row = self.read_from_db(self.tablename, self.sql_from_value, \
            limit=limit)
        return iterable_row
    
    def generate_json_string(self, content:str):
        return json.dumps(content)
    
    def load_json(self, content:str):
        return json.loads(content)
    
class HTMLCorpusReader():
    def __init__(self, tags:list = TAGS):
        # Save the tags that we specifically want to extract.
        self.tags = tags
    
    def html(self, html_content:str):
        """
        Returns the HTML content of each document, cleaning it using
        the readability-lxml library.
        """
        try:
            yield Paper(html_content).summary()
        except Unparseable as e:
            print("Could not parse HTML: {}".format(e))
    
    def paras(self, html_content:str):
        """
        Uses BeautifulSoup to parse the paragraphs from the HTML.
        """
        for html in self.html(html_content):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(self.tags):
                yield element.text
            soup.decompose()
    
    
    def sents(self, html_content:str):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras(html_content):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, html_content:str):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents(html_content):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, html_content:str):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(html_content):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]
    
    def process(self, htmls:list):
        # htmls contains a list of html-strings.
        for html in htmls:
            yield [content for content in self.tokenize(html)]

if __name__ == "__main__":
    sqlite_handler = SQLiteHtmlJson_Connector()
    htmls = sqlite_handler.get_preprocessed_datas(limit=10)
    html_handler = HTMLCorpusReader(htmls)
    # for i in sqlite_handler.preprocessd_htmls:
    #     print(i[0])
    #     print(sqlite_handler.html_single(i[0]))
    
    # for i in sqlite_handler.preprocessed_htmls:
    #     html_content_list = []
    #     for x in sqlite_handler.tokenize(i[0]):
    #         html_content_list.append(x)
    #     print(html_content_list)
    #     print("Finish with one html.")
    # print("finished")
    
    for i in html_handler.process():
        print(sqlite_handler.generate_json_string(i))
    