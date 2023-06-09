import os
import logging
from Shipment_Pricing.logger import logging
from Shipment_Pricing.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Shipment_Pricing.utils.utils import read_yaml_file
from Shipment_Pricing.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
import sys 
import pymongo
import json
from Shipment_Pricing.constant.database import *
from Shipment_Pricing.constant import *
import urllib
import yaml

env_file_path = os.path.join(ROOT_DIR, 'env.yaml')

# Load environment variables from env.yaml
with open(env_file_path) as file:
    env_vars = yaml.safe_load(file)
username = env_vars.get('USER_NAME')
password = env_vars.get('PASS_WORD')

# Use the escaped username and password in the MongoDB connection string
mongo_db_url = f"mongodb+srv://{username}:{password}@rentalbike.5fi8zs7.mongodb.net/"

client = pymongo.MongoClient(mongo_db_url)




class batch_prediction:
    def __init__(self,input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path
        
        
    def data_dump(self,filepath):
        df = pd.read_csv(filepath)
        print(f"Rows and columns: {df.shape}")

        # Convert dataframe to json so that we can dump these record in mongo db
        df.reset_index(drop=True,inplace=True)
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        json_record = list(json.loads(df.T.to_json()).values())
        print(json_record[0])
        
        print("Data Uploaded")

        # Check if the database exists
        if DATABASE_NAME1 in client.list_database_names():
            print(f"The database {DATABASE_NAME1} already exists")
            # Check if the collection exists
            if COLLECTION_NAME1 in client[DATABASE_NAME1].list_collection_names():
                print(f"The collection {COLLECTION_NAME1} already exists")
                # Drop the existing collection
                client[DATABASE_NAME1][COLLECTION_NAME1].drop()
                print(f"The collection {COLLECTION_NAME1} is dropped and will be replaced with new data")
            else:
                print(f"The collection {COLLECTION_NAME1} does not exist and will be created")
        else:
            # Create the database and collection
            print(f"The database {DATABASE_NAME1} does not exist and will be created")
            db = client[DATABASE_NAME1]
            col = db[COLLECTION_NAME]
            print(f"The collection {COLLECTION_NAME1} is created")

        # Insert converted json record to mongo db
        client[DATABASE_NAME1][COLLECTION_NAME1].insert_many(json_record)
        
        logging.info("Prediction Data Updated to MongoDB")
        
    
    def start_batch_prediction(self):
        try:
            os.makedirs(BATCH_PREDICTION, exist_ok=True)
            logging.info(f"Loading the saved pipeline")

            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            # Load the data transformation pipeline
            with open(self.transformer_file_path , 'rb') as f:
                transformer_pipeline = pickle.load(f)

            # Load the model separately
            with open(self.model_file_path, 'rb') as f:
                model = pickle.load(f)

            # loading Pipeline
            pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])

            # Feature Labels 
            schema = read_yaml_file("config\schema.yaml")
            input_features = schema['numerical_columns']
            categorical_features = schema['categorical_columns']
            target_features = schema['target_column']
            drop_columns = schema['drop_columns']
            all_columns=input_features+categorical_features+target_features
            print("Schema information:")
            print("-" * 20)
            print(f"Input features: {input_features}")
            print(f"Categorical features: {categorical_features}")
            print(f"Target feature: {target_features}")
            print(f"Columns to drop: {drop_columns}")
            print("-" * 20)

            # Read the input file
            df = pd.read_csv(self.input_file_path)

            # Apply feature engineering
            df = feature_pipeline.transform(df)

            # Convert the ndarray to a DataFrame
            df = pd.DataFrame(df,columns=all_columns)
            
            df.to_csv("batch_fea_eng.csv",index=False)

            
            logging.info("Feature Engineering Done")
            
            pipeline = Pipeline([
                ('transformer', transformer_pipeline),
                ('model', model)
            ])

            # Make predictions using the trained model
            predictions = pipeline.predict(df)

            # Save the predictions to a file
            # Check if the prediction CSV file exists

            output_file_path = os.path.join(BATCH_PREDICTION, "predictions.csv")
            if os.path.exists(output_file_path):
                logging.info("Found Files in prediction folder ..")
                logging.info("Deleting the found Files")
                # Delete the existing file
                os.remove(output_file_path)
            pd.DataFrame(predictions, columns=[target_features]).to_csv(output_file_path, index=False)
            
            logging.info("Exporting data to Mongo database")
            
            self.data_dump(filepath=output_file_path)

            logging.info(f"Batch prediction completed successfully. Predictions saved to: {output_file_path}")

        except Exception as e:
            logging.error(f"Batch prediction failed due to an error: {str(e)}")


