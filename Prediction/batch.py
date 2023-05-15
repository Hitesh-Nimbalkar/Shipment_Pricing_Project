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



BATCH_PREDICTION = "batch_prediction"
INSTANCE_PREDICTION="Instance_prediction"
input_file_path="SCMS_Delivery_History_Dataset.csv"
feature_engineering_file_path ="prediction_files/feat_eng.pkl"
transformer_file_path ="prediction_files/preprocessed.pkl"
modmodel_file_pathel ="prediction_files/model.pkl"




def start_batch_prediction(input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path):
    try:
        os.makedirs(BATCH_PREDICTION, exist_ok=True)
        logging.info(f"Loading the saved pipeline")

        # Load the feature engineering pipeline
        with open(feature_engineering_file_path, 'rb') as f:
            feature_pipeline = pickle.load(f)

        # Load the data transformation pipeline
        with open(transformer_file_path, 'rb') as f:
            transformer_pipeline = pickle.load(f)

        # Load the model separately
        with open(model_file_path, 'rb') as f:
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
        df = pd.read_csv(input_file_path)

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
        output_file_path = os.path.join(BATCH_PREDICTION, "predictions.csv")
        pd.DataFrame(predictions, columns=[target_features]).to_csv(output_file_path, index=False)

        logging.info(f"Batch prediction completed successfully. Predictions saved to: {output_file_path}")

    except Exception as e:
        logging.error(f"Batch prediction failed due to an error: {str(e)}")


    
start_batch_prediction(input_file_path=input_file_path,model_file_path=modmodel_file_pathel,transformer_file_path=transformer_file_path,
                       feature_engineering_file_path=feature_engineering_file_path)
