from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd

class BigqueryOperation():
    def __init__(self, PATH_GOOGLE_CREDENTIAL:str) -> None:
        # Read credential.
        self._credentials = service_account.Credentials.from_service_account_file(\
            PATH_GOOGLE_CREDENTIAL)
        # Construct a BigQuery client object.
        self._client = bigquery.Client(credentials=self._credentials)
        self.last_request_schema = []
    def query_request(self, QUERY:str) -> pd.DataFrame:
        # Make an API request.
        query_job = self._client.query(QUERY)
        query_result = query_job.result()
        # Get the schema and save as self.last_request_schema
        self.last_request_schema = []
        for i in range(len(query_job.result().schema)):
            column_dict = query_job.result().schema[i].to_api_repr()
            column_str = " ".join(column_dict.values())
            self.last_request_schema.append(column_str)
        # Create an iterable of pandas DataFrames, to process the table as a 
        # stream by using: .to_dataframe_iterable().
        df_iterable = query_result.to_dataframe_iterable()
        return df_iterable

if __name__ == "__main__":
    PATH_GOOGLE_CREDENTIAL = "../Credentials/service-account-file.json"
    QUERY = """
        SELECT *
        FROM `projectstackoverflow.Query_StackOverflow.03_Post_Questions_with_full_family_keywords`
        ORDER BY creation_date DESC
        LIMIT 100
    """
    count = 0
    total_row = 0
    bigquery_object = BigqueryOperation(PATH_GOOGLE_CREDENTIAL)
    df_iterable = bigquery_object.query_request(QUERY)
    for i in df_iterable:
        total_row += len(i)
        print("{}. Total row so far: {}.".format(str(count), str(total_row)))
        count += 1
    # print(bigquery_object.last_request_schema)
    print("Finished")
    