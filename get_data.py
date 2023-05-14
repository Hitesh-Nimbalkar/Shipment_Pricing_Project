
from Shipment_Pricing.data_access.Mongo_Data import mongodata
from Shipment_Pricing.constant.database import COLLECTION_NAME
from main import set_env_variable
import os
if __name__=='__main__':
    data_file_path="SCMS_Delivery_History_Dataset"
    env_file_path='/config/workspace/env.yaml'
    set_env_variable(env_file_path)
    print( os.environ['MONGO_DB_URL'])
    sd = mongodata()
    if COLLECTION_NAME in sd.mongo_client.database.list_collection_names():
        sd.mongo_client.database[COLLECTION_NAME].drop()
    sd.save_csv_file(file_path=data_file_path,collection_name=COLLECTION_NAME)

