import pymongo
import pandas as pd
import json

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/")

DATA_FILE_PATH="SCMS_Delivery_History_Dataset.csv"
DATABASE_NAME = 'Shipment_Pricing'
COLLECTION_NAME = "Data"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    # Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)
    if "_id" in df.columns.to_list():
        df = df.drop(columns=["_id"], axis=1)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    
    print("Data Uploaded")

    # Check if the database exists
    if DATABASE_NAME in client.list_database_names():
        print(f"The database {DATABASE_NAME} already exists")
        # Check if the collection exists
        if COLLECTION_NAME in client[DATABASE_NAME].list_collection_names():
            print(f"The collection {COLLECTION_NAME} already exists")
            # Drop the existing collection
            client[DATABASE_NAME][COLLECTION_NAME].drop()
            print(f"The collection {COLLECTION_NAME} is dropped and will be replaced with new data")
        else:
            print(f"The collection {COLLECTION_NAME} does not exist and will be created")
    else:
        # Create the database and collection
        print(f"The database {DATABASE_NAME} does not exist and will be created")
        db = client[DATABASE_NAME]
        col = db[COLLECTION_NAME]
        print(f"The collection {COLLECTION_NAME} is created")

    # Insert converted json record to mongo db
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    
    print("Data Dump Completed")