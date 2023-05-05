from Shipment_Pricing.logger import logging
from Shipment_Pricing.exception import ApplicationException
from Shipment_Pricing.entity.config_entity import DataTransformationConfig
from Shipment_Pricing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from Shipment_Pricing.constant import *
from Shipment_Pricing.utils.utils import read_yaml_file,save_data,save_object
import sys
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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
        
    def Map_encoding(self,x):
        try:
            Encode_Features=['Country', 'Fulfill_Via', 
                        'Shipment_Mode','Vendor', 'Brand', 'Dosage_Form',
                        'Product_Group',
                        'Sub_Classification',
                        'First_Line_Designation']
            x.to_csv('Before_Encoding.csv', index=False)
            logging.info(f"Columns Before encoding {x.columns}")
            for feature in Encode_Features:
                unique_values = x[feature].unique()
                print(f"Unique values of {feature}: {unique_values}")
                print("\n")

            # Define the mapping for each feature
            mapping = {}
            for feature in Encode_Features:
                unique_values = x[feature].unique()
                mapping[feature] = {value: idx for idx, value in enumerate(unique_values)}

            # Use the mapping to encode each feature and add encoded values to dataframe
            for feature in Encode_Features:
                encoded_values = x[feature].map(mapping[feature])
                x[f"{feature}"] = encoded_values
            x.to_csv("Encoded_data.csv", index=False)    
            # Drop the original columns
            
        

            
            

            return x 
    
        except Exception as e:
            raise ApplicationException(e,sys) from e 
            
      
    def data_modification(self,x):
        try:
            
            # Manufacturing Site - "Others"
          #  counts = x['Manufacturing_Site'].value_counts()
          #  idx = counts[counts.lt(20)].index
          #  x.loc[x['Manufacturing_Site'].isin(idx), 'Manufacturing_Site'] = 'Others'
            
            # Country 
            counts = x['Country'].value_counts()
            idx = counts[counts.lt(30)].index
            x.loc[x['Country'].isin(idx), 'Country'] = 'Others'

            
            # Brand 
            counts = x['Brand'].value_counts()
            idx = counts[counts.lt(50)].index
            x.loc[x['Brand'].isin(idx), 'Brand'] = 'Others'
            
            # Dosage Form modification
            x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("Tablet",case=False)),"Tablet",x["Dosage_Form"])
            x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("Oral",case=False)),"Oral",x["Dosage_Form"])
            x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("Capsule",case=False)),"Capsule",x["Dosage_Form"])
            x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("kit",case=False)),"Test kit",x["Dosage_Form"])
            
            
            
            logging.info(
                    f" >>>>>>>>>>>>  Columns Modififcation Complete  <<<<<<<<<<")
            return x
        
        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
    def drop_columns(self,x):
        try:
            
            columns_to_drop = ['ID', 'Project_Code', 'PQ', 'PO/SO', 'ASN/DN', 'Managed_By', 'Vendor_INCO_Term',
                                'PQ_First_Sent_to_Client_Date', 'PO_Sent_to_Vendor_Date',
                                'Scheduled_Delivery_Date', 'Delivered_to_Client_Date',
                                'Delivery_Recorded_Date','Line_Item_Value',
                                'Item_Description', 'Molecule/Test_Type', 'Dosage',
                                'Weight_(Kilograms)', 'Freight_Cost_(USD)','Line_Item_Insurance_(USD)','Manufacturing_Site']
            
            x.drop(columns=columns_to_drop, inplace=True)
            logging.info('Columns Dropped')
            logging.info(f"Columns After dropping {x.columns}")
            
            data_types = {
                'Country': 'object',
                'Fulfill_Via': 'category',
                'Shipment_Mode': 'category',
                'Product_Group': 'category',
                'Sub_Classification': 'category',
                'Vendor': 'category',
                'Brand': 'object',
                'Dosage_Form': 'object',
                'Unit_of_Measure_(Per_Pack)': 'float64',
                'Line_Item_Quantity': 'float64',
                'Pack_Price': 'float64',
                'Unit_Price': 'float64',
                'First_Line_Designation': 'category',
                'Freight_Cost_USD_Clean': 'float64',
                'Weight_Kilograms_Clean': 'float64'
            }

            # Assume that your dataframe is called "df"
            for col, dtype in data_types.items():
                if col in x.columns:
                    x[col] = x[col].astype(dtype)

            # Logging the datatypes of columns after changing
            logging.info(f'DataTypes of Columns After changing: \n{x.dtypes}')
            
        
            # Assuming 'df' is the name of your dataframe
            logging.info('Columns and DataTypes: \n\n{}'.format(x.dtypes))
            
            
            
            
            
            

            
            return x
        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
            
    def Missing_fills(self,x):
        try:
            # Shipment Mode - Mode 
            mode=x['Shipment_Mode'].mode()
            x['Shipment_Mode']=x['Shipment_Mode'].fillna(mode[0])
            
            # Shipment Mode - Mode 
           # mode=x['Manufacturing_Site'].mode()
           # x['Manufacturing_Site']=x['Manufacturing_Site'].fillna(mode[0])
            
            # Shipment Mode - Mode 
            mode=x['Country'].mode()
            x['Country']=x['Country'].fillna(mode[0])
            
             #BRand 
            mode=x['Brand'].mode()
            x['Brand']=x['Brand'].fillna(mode[0])
            
            
            logging.info(
                    f" >>>>>>>>>>>>   Missing Fills :  Shipment_Mode,Manufacturing_Site,Country,Brand <<<<<<<<<<")
            
            return x
    
            
 
        
        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
    def outlier(self,x):
        try:
            import logging
            import numpy as np
            from sklearn.impute import SimpleImputer

            impute_cols = ['Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity', 'Pack_Price', 'Unit_Price', 'Weight_Kilograms_Clean']

            for i in impute_cols:
                median = np.median(x[i])
                x[i] = np.where(np.isnan(x[i]), median, x[i])
                logging.info(f"Missing values in {i} column have been imputed with median")

            logging.info('Missing value imputation with median completed')

            return x
        except Exception as e:
                raise ApplicationException(e,sys) from e 
    
        
    def Weight_and_freight(self,x):
        try:
            
            regex = {"id_number": ":\d*"
                                    }
            def change_to_number(freight_cost_usd):
                regex = {
                                        "id_number": ":\d*"
                                    }
                match = re.search(regex['id_number'], freight_cost_usd, re.IGNORECASE)
                if match:
                    id = match.group(0).replace(':','')
                    filtered = x.query("ID == "+id)
                    if not filtered.empty:
                        return filtered['Freight_Cost_(USD)'].iloc[0]
                return freight_cost_usd
            


            def convert_to_number(weight):
                regex = {
                                        "id_number": ":\d*"
                                    }
                match = re.search(regex['id_number'], weight, re.IGNORECASE)
                if match:
                    id = match.group(0).replace(':','')
                    filtered = x.query("ID == "+id)
                    if not filtered.empty:
                        return filtered['Weight_(Kilograms)'].iloc[0]
                return weight   


                
                
            x['Freight_Cost_USD_Clean'] = x['Freight_Cost_(USD)'].apply(change_to_number)
            
            x['Weight_Kilograms_Clean'] = x['Weight_(Kilograms)'].apply(convert_to_number)
            
            print("Weight and freight completed")
                        
            freight_cost_indexes = x.index[(x['Freight_Cost_USD_Clean'] == 'Freight Included in Commodity Cost') | (x['Freight_Cost_USD_Clean'] == 'Invoiced Separately')].tolist()
            weight_indexes = x.index[x['Weight_Kilograms_Clean'] == 'Weight Captured Separately'].tolist()
            shipment_indexes = x.index[x['Shipment_Mode'] == 'no_value'].tolist()
            print("Freight_Cost_USD_Clean_indexes:",len(freight_cost_indexes))
            print("Weight_Kilograms_Clean_indexes:",len(weight_indexes))
            print("Shipment_Mode indexes:         ",len(shipment_indexes))

            indexes = list(set(freight_cost_indexes + weight_indexes + shipment_indexes))
            print("Indexes:",len(indexes))
            df_clean = x.drop(indexes)
            
            df_clean = df_clean[~df_clean['Freight_Cost_USD_Clean'].str.contains('See')]
            df_clean = df_clean[~df_clean['Weight_Kilograms_Clean'].str.contains('See')]
            

            
            df_clean["Freight_Cost_USD_Clean"]=df_clean["Freight_Cost_USD_Clean"].astype("float")
            df_clean["Weight_Kilograms_Clean"]=df_clean["Weight_Kilograms_Clean"].astype("float")
            
            
            print(df_clean.shape)
            
            logging.info('Weight and Freight Cost Data Clean Completed')
        
            
            return df_clean
            
        except Exception as e:
            raise ApplicationException(e,sys) from e
        

    
    def data_wrangling(self,x):
        try:
            # Weight_(kilograms) and Freight_Cost Data Clean 
            data=self.Weight_and_freight(x)
            
            # Dropping Columns 
            data = self.drop_columns(data)
            
            

            # Filling Missing Data 
            data= self.Missing_fills(data)
          
            
            # Data Modification 
            data = self.data_modification(data)
            
            # Outlier Detection 
            data = self.outlier(data)
            
            # Data Encoding 
            data = self.Map_encoding(data)
         
            # Drop Columns 
            
            
            return data
    
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
            
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        try:
            X = self.data_wrangling(X)
            
            

            # Reindex DataFrame with new column order
            logging.info('Data Wrangling Completed')
            logging.info('Columns after data wrangling:\n%s', X.columns.to_list())
            
            # reorder columns
            new_order = ['Country', 'Fulfill_Via', 'Shipment_Mode', 'Product_Group', 'Sub_Classification','First_Line_Designation',
                         'Vendor', 'Brand', 'Dosage_Form', 'Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity',
                         'Pack_Price', 'Unit_Price', 'Freight_Cost_USD_Clean','Weight_Kilograms_Clean']
            X = X.reindex(columns=new_order)


            
            X.to_csv('data_wrangling.csv',index=False)
            
        
            arr = X.values

            return arr
        except Exception as e:
            raise ApplicationException(e,sys) from e






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
        
    def get_data_transformer_object(self):
        try:


       
            logging.info('Creating Data Transformer Object')
            
            numerical_columns=['Unit_of_Measure_(Per_Pack)','Line_Item_Quantity','Pack_Price','Unit_Price','Weight_Kilograms_Clean']
            category_columns=['Country','Fulfill_Via','Shipment_Mode','Product_Group', 
                              'Sub_Classification','Vendor','Brand','Dosage_Form','First_Line_Designation']
 
            num_pipeline = Pipeline(steps =[("impute", SimpleImputer(strategy="median")),
                                            ("scaler",StandardScaler())])

            cat_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy="most_frequent"))
                                             ])

            preprocessing = ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),
                                               ('cat_pipeline',cat_pipeline,category_columns)])

            logging.info('Data Transformer Object created successfully')
            return preprocessing
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
        


    def initiate_data_transformation(self):
        try:
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            # Reading schema file for columns details
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(file_path=schema_file_path)
            
            logging.info(f"Extracting train column name {train_df.columns.to_list()}")

            # Extracting target column name
            target_column_name = schema[TARGET_COLUMN_KEY]

            date_columns = schema[DATE_COLUMN_KEY]
            numerical_columns = schema[NUMERICAL_COLUMN_KEY] 
            categorical_columns = schema[CATEGORICAL_COLUMN_KEY]
            string_columns=schema[STRING_COLUMN_KEY]
    

            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            
            
            logging.info(f"Feature Enineering - Train Data ")
            feature_eng_train_arr = fe_obj.fit_transform(train_df)
            
            logging.info(f"Feature Enineering - Test Data ")
            feature_eng_test_arr = fe_obj.transform(test_df)
            
            # Converting featured engineered array into dataframe
            logging.info(f"Converting featured engineered array into dataframe.")
            col=['Country', 'Fulfill_Via', 'Shipment_Mode', 'Product_Group', 'Sub_Classification','First_Line_Designation',
                         'Vendor', 'Brand', 'Dosage_Form', 'Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity',
                         'Pack_Price', 'Unit_Price', 'Freight_Cost_USD_Clean','Weight_Kilograms_Clean']
            
            feature_eng_train_df = pd.DataFrame(feature_eng_train_arr,columns=col)
            
            feature_eng_test_df = pd.DataFrame(feature_eng_test_arr,columns=col)
            
            feature_eng_train_df.to_csv('feature_eng_train_df.csv',index=False)

            target_column_name='Freight_Cost_USD_Clean'
            
            # Train Dataframe 
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            target_feature_train_df = feature_eng_train_df[target_column_name]
            input_feature_train_df = feature_eng_train_df.drop(columns = target_column_name,axis = 1)
            
            Columns=input_feature_train_df.columns
            logging.info(f"Colums before transformer object {Columns}")
            
            
            # Test Dataframe 
            target_feature_test_df = feature_eng_test_df[target_column_name]
            input_feature_test_df = feature_eng_test_df.drop(columns = target_column_name,axis = 1)
            
            input_feature_train_df.to_csv('input_feature_train_df.csv',index=False)
            

            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            
            
            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            column_order=['Unit_of_Measure_(Per_Pack)','Line_Item_Quantity', 'Pack_Price', 'Unit_Price', 'Weight_Kilograms_Clean',
                     'Country', 'Fulfill_Via', 'Shipment_Mode','Product_Group', 'Sub_Classification', 'Vendor', 'Brand', 'Dosage_Form',
                     'First_Line_Designation']
            transformed_train_df = pd.DataFrame(np.c_[train_arr,np.array(target_feature_train_df)],columns=column_order+[target_column_name])
            transformed_test_df = pd.DataFrame(np.c_[test_arr,np.array(target_feature_test_df)],columns=column_order+[target_column_name])

            
        

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            
            transformed_test_dir = self.data_transformation_config.transformed_test_dir           

            
            transformed_train_file_path = os.path.join(transformed_train_dir,"transformed_train.csv")
            transformed_test_file_path = os.path.join(transformed_test_dir,"transformed_test.csv")
            

            
            
            save_data(file_path = transformed_train_file_path, data = transformed_train_df)
            save_data(file_path = transformed_test_file_path, data = transformed_test_df)
            logging.info("Transformed Train and Transformed test file saved")


            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(preprocessing_object_file_path)),obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path = transformed_train_file_path,
            transformed_test_file_path = transformed_test_file_path,
            preprocessed_object_file_path = preprocessing_object_file_path,
            feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")
    
   