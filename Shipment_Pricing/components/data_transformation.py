from Shipment_Pricing.logger import logging
from Shipment_Pricing.exception import ApplicationException
from Shipment_Pricing.entity.config_entity import DataTransformationConfig
from Shipment_Pricing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from Shipment_Pricing.constant import *
from Shipment_Pricing.utils.utils import read_yaml_file, save_data,save_object
import sys
from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os,sys
import pandas as pd
import numpy as np
import re


class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        """
        This class applies necessary Feature Engneering for Rental Bike Share Data
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")








class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    