# Generate preprocessed_html and store in SQLite DB

from f_HTMLCorpusReader import HTMLCorpusReader, SQLiteHtmlJson_Connector
import datetime

start_time = datetime.datetime.now()
sqlite_handler = SQLiteHtmlJson_Connector()
corpus_handler = HTMLCorpusReader()

query_list = sqlite_handler.generate_query_batchread(sqlite_handler.tablename,\
    sqlite_handler.sql_from_value)
print("Start now ...")
count = 0
for query in query_list:
    sqlite_handler.execute_query(query)
    id_list = []
    content_list = []
    for id, content in sqlite_handler.last_cursor:
        id_list.append(id)
        content_list.append(content)
    html_tokens = corpus_handler.process(content_list)
    counter = 0
    for i in html_tokens:
        json_str = sqlite_handler.generate_json_string(i).replace('"', "'")
        # Above code is necessary, otherwise the next sqlite operation won't work.
        sqlite_handler.update_column(sqlite_handler.tablename, \
            "preprocessed_html", "id", json_str, id_list[counter])
        counter += 1
    sqlite_handler.commit()
    count += 1
    print("Finished with {} commit".format(count))

duration = start_time - datetime.timedelta(microseconds=start_time.microsecond)
print("Runtime: {}".format(duration))
print("Finished.")