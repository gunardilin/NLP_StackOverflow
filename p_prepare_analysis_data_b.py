# The purpose of this script is to make the cluster labels match the label from PyLDAvis.
# Background: In PyLDAvis, the labels are sorted by its frequency.

from h_readsqlite import load_from_pickle, save_to_pickle
import xlwings
import webbrowser
import platform
import os

from b_sqlite_operation import SqliteOperation
import pandas as pd

DF_DOMINANT_TOPIC_PATH = "other/model_2012-2021_1/index/df_dominant_topic.pickle"
CLEANED_DF_PATH = "other/model_2012-2021_1/index/cleaned_df.pickle"
LDA_VIS_PATH = "other/model_2012-2021_1/lda_2012-2021_6_topics.html"

def fixing_index():
    # This function reindex the df_dominant_topic from p_prepare_analysis_data_a.py
    # so it matches the PyLDAVis cluster labels.
    # Read df_dominant_topic from an existing pickle
    df_dominant_topic = load_from_pickle(DF_DOMINANT_TOPIC_PATH)

    if platform.system().lower() == "darwin":  # check if on Mac OSX
        webbrowser.get('macosx')
        file_prefix = "file:///"
    else:   # all other non Mac
        file_prefix = ""

    # Open PyLDAVis
    webbrowser.open(file_prefix+os.path.abspath(LDA_VIS_PATH),new=2)

    # Open df_dominant_topic in Excel
    xlwings.view(df_dominant_topic.head(100))

    # Get new_index by comparing both Excel and PyLDAVis manually:
    new_index = {0: 2,
                1: 6,
                2: 1,
                3: 5,
                4: 3,
                5: 4,
                }

    # Reindexing with new_index
    df_dominant_topic.replace({"Dominant_Topic": new_index}, inplace=True)
    # xlwings.view(df_dominant_topic.head(100))

    # Save to original pickle
    save_to_pickle(file_path=DF_DOMINANT_TOPIC_PATH, content=df_dominant_topic)

    # Testing saved pickle
    new_df = load_from_pickle(DF_DOMINANT_TOPIC_PATH)
    xlwings.view(new_df.head(100))
    
    return

# fixing_index()

def add_datetime_to_df():
    DB_PATH = "DB/StackOverflow.sqlite"
    START_YEAR = 2012
    END_YEAR = 2021
    # END_YEAR = 2012
    LIMIT = None
    #, substr(raw_datas.creation_date, 1, 4), substr(raw_datas.creation_date, 6, 2)
    base_query = """
        SELECT preprocessed_datas.id, raw_datas.creation_date
        from preprocessed_datas
        JOIN raw_datas
        ON raw_datas.id = preprocessed_datas.id
        WHERE substr(raw_datas.creation_date, 1, 4) = "{}"
    """

    db_obj = SqliteOperation(DB_PATH)
    id_doc_list = []
    for year in range(START_YEAR, END_YEAR+1):
        print("Reading corpus from Y{}".format(year))
        current_query = base_query.format(str(year))
        # docs = db_handle.docs(year, limit=LIMIT)
        db_obj.execute_query(current_query)
        docs = list(db_obj.last_cursor)
        id_doc_list += list(docs)
    print("Finish reading DB")
    new_df = pd.DataFrame(id_doc_list, columns=['Document_No', 'Datetime'])
    new_df['Datetime'] = pd.DatetimeIndex(new_df['Datetime'])
    # By converting to DatetimeIndex, the Year can be called by: 
    # new_df.Datetime.dt.year
    existing_df = load_from_pickle(DF_DOMINANT_TOPIC_PATH)
    
    combined_df = existing_df.join(new_df.set_index('Document_No'), on='Document_No')
    save_to_pickle(file_path=CLEANED_DF_PATH, content=combined_df)
    
    ### Test new saved pickle
    # test_df = load_from_pickle(CLEANED_DF_PATH)
    # xlwings.view(test_df.head(100))
    return
add_datetime_to_df()