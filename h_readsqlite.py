# Read SQLite and parse Json string into list.

# from f_HTMLCorpusReader import SQLiteHtmlJson_Connector
from b_sqlite_operation import SqliteOperation
# from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader

import json

# sqlite_handler = SQLiteHtmlJson_Connector()
# iterable_row = sqlite_handler.read_from_db("preprocessed_datas", "preprocessed_html", \
#     where_attr="id", like_attr="71367783")
# for i in iterable_row:
#     print(sqlite_handler.load_json(i[0].replace("""'""", '"')))

import time

class SqliteCorpusReader(SqliteOperation):
    def __init__(self, path: str, batchsize: int = 50, \
        tablename: str = "preprocessed_datas", sql_from_value: str = "preprocessed_html"):
        SqliteOperation.__init__(self, path, batchsize)
        self.tablename = tablename
        self.sql_from_value = sql_from_value
        self.error_counter = 0
    
    def ids(self, timestamp:str=None):
        """
        Returns all document ids with the selected timestamp.
        """
        base = """
        SELECT id
        FROM raw_datas
        WHERE creation_date LIKE "{}%"
        """
        base = base.format(timestamp)
        self.execute_query(base)
        return list(self.last_cursor)
    
    def docs(self, timestamp:str=None):
        """
        Returns the document loaded from Sqlite for every row.
        This uses a generator to acheive memory safe iteration.
        """
        base = """
        SELECT preprocessed_html
        FROM preprocessed_datas
        WHERE id IN (
        SELECT id
        FROM raw_datas
        WHERE creation_date LIKE "{}%"
        )
        """
        base = base.format(timestamp)
        self.execute_query(base)
        self.error_counter = 0
        for i in self.last_cursor:
            try:
                json_str = i[0].replace("['", '["').replace("', '", '", "').\
                    replace("']", '"]').replace("""", \'""", '", "').\
                    replace("\\", "").replace('"[""', '"["').replace('[""]"', '["]"')
                yield json.loads(json_str)
            except:
                self.error_counter += 1

    def paras(self, timestamp:str=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for doc in self.docs(timestamp):
            for paragraph in doc:
                yield paragraph

    def sents(self, timestamp:str=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for paragraph in self.paras(timestamp):
            for sentence in paragraph:
                yield sentence

    def words(self, timestamp:str=None):
        """
        Returns a generator of (token, tag) tuples.
        """
        for sentence in self.sents(timestamp):
            for token in sentence:
                yield token
                

if __name__ == "__main__":
    start_time = time.time()
    PATH =  "DB/StackOverflow.sqlite"
    corpus_reader = SqliteCorpusReader(path=PATH)
    counter = 0
    for year in range(2022,2023):
        print("Start reading datas for year {}.".format(year))
        for i in corpus_reader.words(year):
            counter += 1
        print("Year {}".format(str(year)))
        print("List counter: {}".format(counter))
        print("Error counter: {}".format(str(corpus_reader.error_counter)))
        print("Percentage: {}%".format(round(corpus_reader.error_counter/counter, 4)))
    end_time = time.time()
    print("Duration: {} seconds".format(round(end_time-start_time, 2)))
        

# class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):

#     def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
#         """
#         Initialize the corpus reader.  Categorization arguments
#         (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
#         the ``CategorizedCorpusReader`` constructor.  The remaining arguments
#         are passed to the ``CorpusReader`` constructor.
#         """
#         # Add the default category pattern if not passed into the class.
#         if not any(key.startswith('cat_') for key in kwargs.keys()):
#             kwargs['cat_pattern'] = CAT_PATTERN

#         CategorizedCorpusReader.__init__(self, kwargs)
#         CorpusReader.__init__(self, root, fileids)

#     def resolve(self, fileids, categories):
#         """
#         Returns a list of fileids or categories depending on what is passed
#         to each internal corpus reader function. This primarily bubbles up to
#         the high level ``docs`` method, but is implemented here similar to
#         the nltk ``CategorizedPlaintextCorpusReader``.
#         """
#         if fileids is not None and categories is not None:
#             raise ValueError("Specify fileids or categories, not both")

#         if categories is not None:
#             return self.fileids(categories)
#         return fileids

#     def docs(self, fileids=None, categories=None):
#         """
#         Returns the document loaded from a pickled object for every file in
#         the corpus. Similar to the BaleenCorpusReader, this uses a generator
#         to acheive memory safe iteration.
#         """
#         # Resolve the fileids and the categories
#         fileids = self.resolve(fileids, categories)

#         # Create a generator, loading one document into memory at a time.
#         for path, enc, fileid in self.abspaths(fileids, True, True):
#             with open(path, 'rb') as f:
#                 yield pickle.load(f)

#     def paras(self, fileids=None, categories=None):
#         """
#         Returns a generator of paragraphs where each paragraph is a list of
#         sentences, which is in turn a list of (token, tag) tuples.
#         """
#         for doc in self.docs(fileids, categories):
#             for paragraph in doc:
#                 yield paragraph

#     def sents(self, fileids=None, categories=None):
#         """
#         Returns a generator of sentences where each sentence is a list of
#         (token, tag) tuples.
#         """
#         for paragraph in self.paras(fileids, categories):
#             for sentence in paragraph:
#                 yield sentence

#     def tagged(self, fileids=None, categories=None):
#         for sent in self.sents(fileids, categories):
#             for token in sent:
#                 yield token

#     def words(self, fileids=None, categories=None):
#         """
#         Returns a generator of (token, tag) tuples.
#         """
#         for token in self.tagged(fileids, categories):
#             yield token[0]