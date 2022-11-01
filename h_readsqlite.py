# Read SQLite and parse Json string into list.

from f_SqliteCorpusReader import SQLiteHtmlJson_Connector

sqlite_handler = SQLiteHtmlJson_Connector()
iterable_row = sqlite_handler.read_from_db("preprocessed_datas", "preprocessed_html", \
    where_attr="id", like_attr="71367783")
for i in iterable_row:
    print(sqlite_handler.load_json(i[0].replace("""'""", '"')))