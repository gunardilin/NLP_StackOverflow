import sqlite3
from xmlrpc.client import Boolean
import pandas as pd
class SqliteOperation(object):
    """Generic class for basic SQLite Operations."""
    def __init__(self, path:str, batchsize:int=50):
        self.path = path
        self._connect = sqlite3.connect(path)
        self._cur = self._connect.cursor()
        self._connect.row_factory = sqlite3.Row
        self.batchsize = batchsize
        self.last_cursor = self._cur # Needed to query DB batchwise
        
    def execute_query(self, query:str, values:list=None, commit_=True) -> None:
        try:
            if values == None:
                self._cur.execute(query)
            else:
                self._cur.execute(query, values)
            if commit_:
                self._connect.commit()
                # print("*** Query commited successfully.")
        except sqlite3.Error as e:
            print(f"*** The error '{e}' occurred.")
        return
    
    def commit(self):
        self._connect.commit()
        return
    
    def close_conn(self):
        self._cur.close()
        self._connect.close()
        return
    
    def reopen_cursor(self):
        self.close_conn()
        self._connect = sqlite3.connect(self.path)
        self._cur = self._connect.cursor()
        return
    
    def create_table(self, name:str, schema:list) -> None:
        combined_schema = ", ".join(schema)
        create_table_query = """
        CREATE TABLE IF NOT EXISTS {}({})
        """.format(name, combined_schema)
        self.execute_query(create_table_query)
        return
    
    def convert_schema_to_label(self, schema:list) -> list:
        label_list = []
        for i in schema:
            # Extract only column name & insert to a list
            label_list.append((i.split()[0]))
        return label_list
    
    def insert_table(self, name:str, label:list, values_to_insert:list, commit_=True) -> None:
        total_columns = len(label)
        # Create necessary strings for SQLite query
        combined_label = ", ".join(label)
        string_for_values = ",".join(['?' for i in range(total_columns)])
        insert_table_query = """
        INSERT INTO {} ({})
        VALUES ({})
        """.format(name, combined_label, string_for_values)
        self.execute_query(insert_table_query, values_to_insert, commit_)
        return
    
    def repair_df_format(self, df_to_insert:pd.DataFrame) -> pd.DataFrame:
        label = list(df_to_insert.columns)
        #1 Repair date columns have correct format:
        # Necessary because SQLite doesn't have a datetime datatype
        # The next line looks for any label with date string
        datelabel_index = [i for i, s in enumerate(list(df_to_insert.columns)) if 'date' in s]
        for i in datelabel_index:
            current_label = label[i]
            df_to_insert[current_label] = df_to_insert[current_label].astype('str')
        return df_to_insert
        
    def insert_table_from_df(self, name, df_to_insert:pd.DataFrame) -> None:
        # Wraper for insert_table method for dataframe input.
        count = 0
        commit_ = False
        df_to_insert = self.repair_df_format(df_to_insert)
        label = list(df_to_insert.columns)
        for i in df_to_insert.itertuples(index=False):
            self.insert_table(name, label, list(i), commit_)
            # Divide and conquer commiting:
            count += 1
            if count == 100:
                commit_, count = True, 0
            else:
                commit_ = False
        self._connect.commit()
        # print("*** Query commited successfully.")
        return
    
    def add_new_column(self, table_name:str, new_column_name:str, \
        new_column_format:str) -> None:
        base = "ALTER TABLE {} ADD {} {}"
        return base.format(table_name, new_column_name, new_column_format)
    
    def update_column(self, table_name, column_name, column_value, \
        where_column, where_value):
        #UPDATE preprocessed_datas SET preprocessed_html = NULL WHERE id = 71367783
        base = "UPDATE {} SET {} = {} WHERE {} = {}"
        query = base.format(table_name, column_name, column_value, where_column, \
            where_value)
        self.execute_query(query, commit_=False)
        return
    
    def generate_query_to_read_db(self, name:str, select_attr:str="*", \
        where_attr:str=None, like_attr:str=None, order_attr:str=None, \
        descending:Boolean=True, limit:int=None, offset:int=None) -> str:
        """Generate query to read from SQLite."""
        base = r'SELECT {} FROM {}'
        base = base.format(select_attr, name)
        if where_attr != None:
            where_base = r'WHERE {} LIKE {}'
            where_base.format(where_attr, like_attr)
            base = " ".join([base, where_base])
        if order_attr != None:
            order_base = r'ORDER BY {} {}'
            if descending:
                direction_str = "DESC"
            else:
                direction_str = "ASC"
            order_base.format(order_attr, direction_str)
            base = " ".join([base, order_base])
        if limit != None:
            limit_str = r'LIMIT {}'.format(str(limit))
            base = " ".join([base, limit_str])
        if offset != None:
            offset_str = r'OFFSET {}'.format(str(offset))
            base = " ".join([base, offset_str])
        return base
    
    def read_from_db(self, name:str, select_attr:str="*", \
        where_attr:str=None, like_attr:str=None, order_attr:str=None, \
        descending:Boolean=True, limit:int=None, offset:int=None) -> sqlite3.Cursor:
        """This function reads table from a DB and returns an itterable.
        It uses "generate_query_to_read_db" to generate the necessary query.
        """
        query_str = self.generate_query_to_read_db(name, select_attr, \
            where_attr, like_attr, order_attr, descending, limit, offset)
        self.execute_query(query_str)
        for result in iter(self._cur.fetchone, None):
            yield result
    
    def get_rowcount_from_db(self, name:str) -> int:
        """Get total row count of a table. Useful when querying using offset."""
        rowcount_itter = self.read_from_db(name, "COUNT(*)")
        rowcount = next(rowcount_itter)[0]
        return rowcount
    
    def generate_query_batchread(self, name:str, select_attr:str="*", \
        where_attr:str=None, like_attr:str=None, order_attr:str=None, \
        descending:Boolean=True, limit:int=None, offset:int=None) -> list:
        """This function will generate a list consisting query string for 
        multiple batch sizes with OFFSET and save it in a list.
        Output: list of query string.
        Goal: querying the DB in a memory friendly way.
        How to use? Use self.execute_query(self.generate_query_batchread(...)).
        It becomes iterator and with for loop, each iteration can be called.
        """
        max_rowcount = self.get_rowcount_from_db(name)
        if offset == None:
            start_rowcount = 0
        else:
            start_rowcount = offset
            
        if limit == None:
            total_requested_rowcount = max_rowcount
        else:
            total_requested_rowcount = limit
        
        # Generate the size for different batches for LIMIT variable
        available_rowcount = total_requested_rowcount - start_rowcount
        fullrange_count = available_rowcount//self.batchsize
        lastrange_count = available_rowcount%self.batchsize
        batchsize_list = []
        for i in range(fullrange_count):
            batchsize_list.append(self.batchsize)
        batchsize_list.append(lastrange_count)

        # Generate the start position for different batches
        start_position_list = []
        temp_position = start_rowcount
        for i in range(fullrange_count+1):
            temp_position = start_rowcount + i*self.batchsize
            start_position_list.append(temp_position)
        
        # Generate query strings with different start_position, batch size and
        # save it in a list.
        batch_query_list = []
        for i in range(fullrange_count+1):
            start_position = start_position_list[i]
            batch_size = batchsize_list[i]
            query_str = self.generate_query_to_read_db(name, select_attr, \
                where_attr, like_attr, order_attr, descending, \
                batch_size, start_position)
            batch_query_list.append(query_str)
        return batch_query_list
        
if __name__ == "__main__":
    PATH_SQLITE = "./DB/StackOverflow.sqlite"
    NAME_TABLE = "raw_datas"
    
    sqlite_handler = SqliteOperation(PATH_SQLITE, 3)
    itterable_row = sqlite_handler.read_from_db(NAME_TABLE, "id", limit=10)
    
    for i in itterable_row:
        print(i)
    
    # batch_query_list = sqlite_handler.generate_query_batchread(NAME_TABLE, "id", limit=12)
    # batch_count = 0
    # for batch in batch_query_list:
    #     print("{}. itterator:".format(batch_count+1))
    #     print(batch)
    #     batch_count += 1
    #     sqlite_handler.execute_query(batch)
    #     batch_iterator = sqlite_handler.last_cursor
    #     for i in batch_iterator:
    #         print(i)