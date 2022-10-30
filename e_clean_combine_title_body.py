from b_sqlite_operation import SqliteOperation
from d_add_remove_tag import add_h1_tag, remove_codes, remove_links, remove_pre_tags

PATH_SQLITE = "./DB/StackOverflow.sqlite"
NAME_TABLE = "raw_datas"
NAME_NEW_TABLE = "preprocessed_datas"
SCHEMA_NEW_TABLE = ['id INTEGER NULLABLE', 'content STRING NULLABLE']
LABEL_LIST = ['id', 'content']

sqlite_handler = SqliteOperation(PATH_SQLITE, 50) # For creating and reading query.

# Workflow:
#0 Create new table for processed datas.
#1 Requesting title and body from table raw_datas and get it as iterable.
#2 Execute add_h1_tag + remove_codes for every iteration.
#3 Combine precleaned title and body.
#4 Save inside a new table preprocessed_table with schema: id, content.

#0 Create new table for processed datas.
sqlite_handler.create_table(NAME_NEW_TABLE, SCHEMA_NEW_TABLE)

#1 Requesting title and body from table raw_datas and get it as iterable.
query_list = sqlite_handler.generate_query_batchread(NAME_TABLE, "id, title, body")

#2 Execute add_h1_tag + remove_codes for every iteration.
count = 0
for query in query_list:
    sqlite_handler.execute_query(query)
    batch_iterator = sqlite_handler._cur
    id_list, combined_text_list = [], []
    for i in batch_iterator:
        id, title, body = i
        new_title = add_h1_tag(title)
        new_body = remove_codes(body)
        new_body = remove_links(new_body)
        new_body = remove_pre_tags(new_body)
        #3 Combine precleaned title and body.
        combined_text = "\n".join([new_title, new_body])
        id_list.append(id)
        combined_text_list.append(combined_text)
        # For debuging purpose, activate:
        # print(id)
        # print(combined_text)

    #4 Save inside a new table preprocessed_table with schema: id, content.
    for n in range(len(id_list)):
        id = id_list[n]
        combined_text = combined_text_list[n]
        # Insert the datas from id_list + combined_text_list into the new table.
        sqlite_handler.insert_table(NAME_NEW_TABLE, LABEL_LIST, \
            [id, combined_text], False)
    # Commiting insertion.
    sqlite_handler.commit()
    print("{}# Commiting...".format(count))
    count += 1
sqlite_handler.close_conn()
print("Finished")