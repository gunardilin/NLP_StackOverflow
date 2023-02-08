from i_vectorizer import GensimVectorizer_Topic_Discovery
import pickle
from gensim.models import LdaModel
import pandas as pd
import bs4
from b_sqlite_operation import SqliteOperation
import xlwings
import time
from h_readsqlite import load_from_pickle, save_to_pickle
from h_readsqlite import create_faulty_dict, create_total_index, create_existing_data_index

# The purpose of this script is to get the data from existing LDA model.  
# The retreived data will be used for further analysis.

# The index for existing data is necessary before executing this script.
# To get the index, go back to h_readsqlite.py and execute:
# 1. create_total_index 2. create_faulty_dict 3. create_existing_data_index
# 4. Save it in pickle 5. Show in Table.

#!!! This code needs to be executed once, because the result as Dataframe 
# & pickle is saved under DF_DOMINANT_TOPIC_PATH and can be called
# multiple times by:
# a = load_from_pickle(DF_DOMINANT_TOPIC_PATH)
# b = xlwings.view(a.head(100))


DB_PATH = "DB/StackOverflow.sqlite"
LEXICON_PATH = "other/model_2012-2021_1/lexicon.pkl"
DOCMATRIX_PATH = "other/model_2012-2021_1/doc_matrix.pickle"
MODEL_PATH = "other/model_2012-2021_1/LDA_model_6_topics"
START_YEAR = 2012
END_YEAR = 2021
LIMIT = None

FOLDER_PATH = "other/model_2012-2021_1/index"
FAULTY_FILE_NAME = "faulty_data_index"
TOTAL_FILE_NAME = "total_data_index"
EXISTING_INDEX_PATH = "existing_data_index"

DF_DOMINANT_TOPIC_PATH = "other/model_2012-2021_1/index/df_dominant_topic.pickle"

total_path = "{}/{}.pickle".format(FOLDER_PATH, TOTAL_FILE_NAME)
# "other/model_2012-2021_1/index/total_data_index.pickle"
faulty_path = "{}/{}.pickle".format(FOLDER_PATH, FAULTY_FILE_NAME)
# "other/model_2012-2021_1/index/faulty_data_index.pickle"
existing_index_path = "{}/{}.pickle".format(FOLDER_PATH, EXISTING_INDEX_PATH)
# "other/model_2012-2021_1/index/existing_data_index.pickle"

### 0. Initialization:
def initialization():
    # a. create faulty_dict b. create total_index 
    # c. remove faulty_index from total_index (using result from a and b)
    # This step is necessary before executing further codes.
    # Necessary because not all documents from DB can be parsed into string, 
    # therefore some documents with errors were skipped. Therefore the 
    # id sequence is messed up, due to some missing docs (0.02%)
    start_time = time.time()
    # Create dict containing docs that can't be parsed:
    create_faulty_dict(DB_PATH, FOLDER_PATH, FAULTY_FILE_NAME)
    # Get the id of all docs:
    create_total_index(DB_PATH, FOLDER_PATH, TOTAL_FILE_NAME)
    # Remove the faulty index (stored in faulty_dict) from total_index and saved
    # in existing_data_index.pickle
    create_existing_data_index(total_path, faulty_path, existing_index_path)
    a = load_from_pickle(total_path)
    b = load_from_pickle(faulty_path)
    c = load_from_pickle(existing_index_path)

    for i in list(a.keys()):
        print("Year {}: {}".format(i, len(a[i])-len(b[i])==len(c[i])))
        print(len(a[i]), len(b[i]), len(c[i]))
    end_time = time.time()
    print("Duration: {} seconds".format(round(end_time-start_time, 2)))
    return

# initialization()

### 1. Loading Lexicon
print("Read Gensim Dictionary / Lexicon")
vectorizer = GensimVectorizer_Topic_Discovery(LEXICON_PATH)
print(vectorizer.id2word.token2id)
print("Finish reading Gensim Dictionary")

### 2. Loading Doc Matrix
print("Start reading Doc Matrix / Corpus")
docmatrix = None
with open(DOCMATRIX_PATH, "rb") as f:
    docmatrix = pickle.load(f)
print("Finish reading DocMatrix.")

### 3. Loading LDA model
print("Loading LDA model")
ldamodel = LdaModel.load(MODEL_PATH)
print("Finish reading LDA transformer.")

### 4. Getting corpus texts from DB
print("Getting corpus texts")
base_query = """
    SELECT preprocessed_datas.id, preprocessed_datas.content
    from preprocessed_datas
    JOIN raw_datas
    ON raw_datas.id = preprocessed_datas.id
    WHERE substr(raw_datas.creation_date, 1, 4) = "{}"
"""
# db_handle = SqliteCorpusReader(DB_PATH)
db_obj = SqliteOperation(DB_PATH)
id_doc_list = []
for year in range(START_YEAR, END_YEAR+1):
    print("Reading corpus from Y{}".format(year))
    current_query = base_query.format(str(year))
    # docs = db_handle.docs(year, limit=LIMIT)
    db_obj.execute_query(current_query)
    docs = list(db_obj.last_cursor)
    id_doc_list += list(docs)
print("Removing tags")
id_list, doc_list = [], []
for id, doc in id_doc_list:
    id_list.append(id)
    doc_str = bs4.BeautifulSoup(doc, 'lxml').text
    doc_list.append(doc_str)
id_doc_list = None      # Erase unnecessary datas from memory
print("Finish reading DB")

### 5. Creating humanreadable table
print("Getting topics")
def format_topics_sentences(ldamodel, corpus, docs, limit:int=None):
    # Init output
    sent_topics_df = pd.DataFrame()
    if limit is None:
        targetCorpus = corpus
    else:
        targetCorpus = corpus[:limit]
    # Get main topic in each document
    for i, row in enumerate(ldamodel[targetCorpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(doc_list)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.reset_index(inplace=True)
    sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return(sent_topics_df)

# 6. To get the correct Document ID under column Document_No execute:
existing_index = load_from_pickle(existing_index_path)
relevant_index = []
for year in range(START_YEAR, END_YEAR+1):
    relevant_index.extend(existing_index[year])

# 7. Generate relevant doc_list and exclude the 841 unparseable data:
a = load_from_pickle(total_path)
b = load_from_pickle(faulty_path)
list_a, list_b, list_index = [], [], []
for year in range(START_YEAR, END_YEAR+1):
    list_a.extend(a[year])
    list_b.extend(b[year])
for i in list_b:
    list_index.append(list_a.index(i))
list_index.sort(reverse=True)
for i in list_index:
    del doc_list[i]

# 7. Show the table
print("Creating DF")
df_dominant_topic = format_topics_sentences(ldamodel, docmatrix, doc_list, LIMIT)
if LIMIT == None:
    df_dominant_topic['Document_No'] = relevant_index
else:
    df_dominant_topic['Document_No'] = relevant_index[:LIMIT]
print(df_dominant_topic.head(10))
save_to_pickle(content=df_dominant_topic, file_path=DF_DOMINANT_TOPIC_PATH)
print("Open table in Excel")
xlwings.view(df_dominant_topic.head(100))
print("Finish reading topics.")