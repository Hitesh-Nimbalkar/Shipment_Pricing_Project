from collections import namedtuple


DataIngestionConfig=namedtuple("DataIngestionConfig",[
    "dataset_download_url",
    "raw_data_dir",
    "ingested_train_dir",
    "ingested_test_dir"
    ])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path"])