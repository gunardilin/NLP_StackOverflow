from a_requesting_bigquery import BigqueryOperation
from b_sqlite_operation import SqliteOperation
import time

PATH_GOOGLE_CREDENTIAL = "../Credentials/service-account-file.json"
QUERY = """
    SELECT *
    FROM `projectstackoverflow.Query_StackOverflow.03_Post_Questions_with_full_family_keywords`
    ORDER BY creation_date DESC
"""
PATH_SQLITE = "./DB/StackOverflow.sqlite"
NAME_TABLE = "raw_datas"

bigquery_handler = BigqueryOperation(PATH_GOOGLE_CREDENTIAL)
df_iterable = bigquery_handler.query_request(QUERY)
df_schema = bigquery_handler.last_request_schema

sqlite_handler = SqliteOperation(PATH_SQLITE)
sqlite_handler.create_table(NAME_TABLE, df_schema)

count = 0
total_row = 0
for i in df_iterable:
    total_row += len(i)
    print("{}. Total row so far: {}.".format(str(count), str(total_row)))
    sqlite_handler.insert_table_from_df(NAME_TABLE, i)
    del i
    count += 1
    time.sleep(3)
print("Finished")