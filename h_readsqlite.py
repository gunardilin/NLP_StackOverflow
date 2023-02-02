# Read SQLite and parse Json string into list.

# from f_HTMLCorpusReader import SQLiteHtmlJson_Connector
from b_sqlite_operation import SqliteOperation
# from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader

import json
import os
import pickle

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
        self.faulty_index = []  # Necessary to store faulty index because of 
                                # fail parsing during: get_ids_docs
    
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
        id_list = []
        for i in self.last_cursor:
            id_list.append(i[0])
        for i in reversed(self.faulty_index):
            id_list.pop(i)
        return id_list
    
    def docs(self, timestamp:str=None, limit:str=None):
        """
        Returns the document loaded from Sqlite for every row.
        This uses a generator to acheive memory safe iteration.
        """
        base = """
        SELECT id, preprocessed_html
        FROM preprocessed_datas
        WHERE id IN (
        SELECT id
        FROM raw_datas
        WHERE creation_date LIKE "{}%"
        )
        """
        base = base.format(timestamp)
        if limit != None:
            limit_str = """
            LIMIT {}
            """.format(limit)
            base = base + limit_str
        self.execute_query(base)
        self.error_counter = 0
        for i in self.last_cursor:
            try:
                json_str = i[1].replace("['", '["').replace("', '", '", "').\
                    replace("']", '"]').replace("""", \'""", '", "').\
                    replace("\\", "").replace('"[""', '"["').replace('[""]"', '["]"')
                yield json.loads(json_str)
            except:
                self.error_counter += 1
                self.faulty_index.append(i[0])

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

def save_to_pickle(folder_path=None, file_name=None, content=None, file_path=None):
    if file_path == None:
        file_path = "{}/{}.pickle".format(folder_path, file_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    with open(file_path,"wb") as f:
        pickle.dump(content, f)
    print("Data are saved into pickle: {}.".format(file_path))
    return

def load_from_pickle(file_path):
    with open(file_path, "rb") as r:
        content = pickle.load(r)
    print("Content from pickle: {}".format(content))
    return content
    

def create_faulty_dict(DB_PATH, FOLDER_PATH, FILE_NAME, start_year=2008, \
    end_year=2022):
    # It is necessary to generate and save in pickle all faulty document ids.
    # These ids will be used in p_analysis.py
    
    corpus_reader = SqliteCorpusReader(path=DB_PATH)
    counter = 0
    faulty_index_dict = {}
    for year in range(start_year, end_year+1):
        print("Start reading datas for year {}.".format(year))
        for i in corpus_reader.words(year):
            counter += 1
        print("Year {}".format(str(year)))
        print("List counter: {}".format(counter))
        print("Error counter: {}".format(str(corpus_reader.error_counter)))
        print("Percentage: {}%".format(round(corpus_reader.error_counter/counter, 4)))
        print("Faulty index: {}".format(corpus_reader.faulty_index))
        faulty_index_dict[year] = corpus_reader.faulty_index
        corpus_reader.faulty_index = []
        print("\n")
    
    save_to_pickle(FOLDER_PATH, FILE_NAME, faulty_index_dict)
    load_from_pickle("{}/{}.pickle".format(FOLDER_PATH, FILE_NAME))

def create_total_index(DB_PATH, FOLDER_PATH, FILE_NAME, start_year=2008, \
    end_year=2022):
    # This function will read all document ids, group it based on years and 
    # save into pickle. These ids will be used in p_analysis.py
    db_handle = SqliteOperation(DB_PATH)
    base = """
        SELECT id
        FROM raw_datas
        WHERE creation_date LIKE "{}%"
        """
    id_dictionary = {}
    for year in range(start_year, end_year+1):
        print("Processing datas for {}".format(year))
        query = base.format(year)
        db_handle.execute_query(query)
        temp_list = list(db_handle.last_cursor)
        temp_list2 = []
        for i in temp_list:
            temp_list2.append(i[0])
        id_dictionary[year] = temp_list2
    print("Saving to pickle")
    save_to_pickle(FOLDER_PATH, FILE_NAME, id_dictionary)
    print("\n")
    load_from_pickle("{}/{}.pickle".format(FOLDER_PATH, FILE_NAME))
    return

def create_existing_data_index(total_path, faulty_path, save_path):
    total_index = load_from_pickle(total_path)
    faulty_index = load_from_pickle(faulty_path)
    for year in list(faulty_index.keys()):
        for i in faulty_index[year]:
            total_index[year].remove(i)
    save_to_pickle(file_path=save_path, content=total_index)
    return

if __name__ == "__main__":
    
    start_time = time.time()
    # PATH =  "DB/StackOverflow.sqlite"
    # FOLDER_PATH = "other/model_2012-2021_1/index"
    # FILE_NAME = "faulty_data_index"
    # create_faulty_dict(PATH, FOLDER_PATH, FILE_NAME)
    
    # FILE_NAME = "total_data_index"
    # create_total_index(PATH, FOLDER_PATH, FILE_NAME)
    
    total_path = "other/model_2012-2021_1/index/total_data_index.pickle"
    faulty_path = "other/model_2012-2021_1/index/faulty_data_index.pickle"
    save_path = "other/model_2012-2021_1/index/existing_data_index.pickle"
    # create_existing_data_index(total_path, faulty_path, save_path)
    a = load_from_pickle(total_path)
    b = load_from_pickle(faulty_path)
    c = load_from_pickle(save_path)
    
    for i in list(a.keys()):
        print("Year {}: {}".format(i, len(a[i])-len(b[i])==len(c[i])))
        print(len(a[i]), len(b[i]), len(c[i]))
    end_time = time.time()
    print("Duration: {} seconds".format(round(end_time-start_time, 2)))