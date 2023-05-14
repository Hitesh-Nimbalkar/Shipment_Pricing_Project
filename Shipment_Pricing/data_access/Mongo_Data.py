import sys
from typing import Optional

import numpy as np
import pandas as pd
import json
from Shipment_Pricing.config.mongo_db_connection import MongoDBClient
from Shipment_Pricing.constant.database import DATABASE_NAME
from Shipment_Pricing.exception import ApplicationException


class mongodata:
    """
    This class help to export entire mongo db record csv 
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise ApplicationException(e, sys)


    def save_csv_file(self,file_path ,collection_name: str, database_name: Optional[str] = None):
        try:
            data_frame=pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise ApplicationException(e, sys)


    def export_collection_as_csv(self, collection_name: str, file_path: str, database_name: Optional[str] = None) -> None:
        try:
            """
            Export an entire MongoDB collection as a CSV file.
            :param collection_name: Name of the MongoDB collection
            :param file_path: File path to save the CSV file
            :param database_name: Name of the MongoDB database (optional, defaults to None)
            :return: None
            """
            if database_name is None:
                collection = self.mongo_client.client[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            csv_string = df.to_csv(index=False)
            

            # Write the CSV string to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_string)

        except Exception as e:
            raise ApplicationException(e, sys)
