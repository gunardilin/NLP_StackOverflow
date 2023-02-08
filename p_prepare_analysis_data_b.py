# The purpose of this script is to make the cluster labels match the label from PyLDAvis.
# Background: In PyLDAvis, the labels are sorted by its frequency.

from h_readsqlite import load_from_pickle, save_to_pickle
import xlwings
import webbrowser
import platform
import os

DF_DOMINANT_TOPIC_PATH = "other/model_2012-2021_1/index/df_dominant_topic.pickle"
LDA_VIS_PATH = "other/model_2012-2021_1/lda_2012-2021_6_topics.html"


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
xlwings.view(df_dominant_topic.head(100))

# Save to original pickle
save_to_pickle(file_path=DF_DOMINANT_TOPIC_PATH, content=df_dominant_topic)

# Testing saved pickle
new_df = load_from_pickle(DF_DOMINANT_TOPIC_PATH)
xlwings.view(new_df.head(100))