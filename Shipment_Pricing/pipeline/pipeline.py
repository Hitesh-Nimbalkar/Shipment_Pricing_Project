from collections import namedtuple
from datetime import datetime
import uuid
from Shipment_Pricing.config.configuration import Configuration
from Shipment_Pricing.logger import logging
from Shipment_Pricing.exception import ApplicationException
from threading import Thread
from typing import List
from Shipment_Pricing.utils.utils import read_yaml_file
from multiprocessing import Process
from Shipment_Pricing.entity.artifact_entity import DataIngestionArtifact
from Shipment_Pricing.components.data_ingestion import DataIngestion
import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd



class Pipeline():

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e