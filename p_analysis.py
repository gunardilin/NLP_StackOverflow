from i_vectorizer import GensimVectorizer_Topic_Discovery
import pickle
from gensim.models import LdaModel
import pandas as pd
import bs4
from b_sqlite_operation import SqliteOperation
import xlwings

# The index for existing data is necessary before executing this script.
# To get the index, go back to h_readsqlite.py and execute:
# 1. create_total_index 2. create_faulty_dict 3. create_existing_data_index

DB_PATH = "DB/StackOverflow.sqlite"
LEXICON_PATH = "other/model_2012-2021_1/lexicon.pkl"
DOCMATRIX_PATH = "other/model_2012-2021_1/doc_matrix.pickle"
MODEL_PATH = "other/model_2012-2021_1/LDA_model_6_topics"
START_YEAR = 2012
END_YEAR = 2021
LIMIT = None

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

# 6. Show the table
print("Creating DF")
df_dominant_topic = format_topics_sentences(ldamodel, docmatrix, doc_list, 100)
print(df_dominant_topic.head(10))
print("Open table in Excel")
xlwings.view(df_dominant_topic.head(100))
print("Finish reading topics.")