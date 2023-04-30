from Shipment_Pricing.logger import logging
from Shipment_Pricing.exception import ApplicationException
from Shipment_Pricing.entity.config_entity import DataTransformationConfig
from Shipment_Pricing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from Shipment_Pricing.constant import *
from Shipment_Pricing.utils.utils import read_yaml_file,save_data,save_object
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import asarray
from sklearn.preprocessing import StandardScaler
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
        
    def encoding(self,x):
        try:
                # Shipment Mode 
            x['shipment_mode_encoded'] = x['Shipment_Mode'].map({ 
                    'Air': 1,
                    'Truck': 2,
                    'Air Charter': 3,
                    'Ocean': 4 }).astype(int)  
            
            
            
            
            # Manufacturing Site 
            mapping = { 
                'Aurobindo Unit III, India': 1, 
                'Mylan (formerly Matrix) Nashik': 2,
                'Hetero Unit III Hyderabad IN': 3, 
                'Cipla, Goa, India': 4,
                'Strides, Bangalore, India.': 5, 
                'Alere Medical Co., Ltd.': 6,
                'Trinity Biotech, Plc': 7, 
                'ABBVIE Ludwigshafen Germany': 8,
                'Inverness Japan':9,
                'Others': 10, 
                'ABBVIE (Abbott) Logis. UK': 11,
                'Chembio Diagnostics Sys. Inc.': 12, 
                'Standard Diagnostics, Korea': 13,
                'Aurobindo Unit VII, IN': 14, 
                'Aspen-OSD, Port Elizabeth, SA': 15,
                'MSD, Haarlem, NL': 16, 
                'KHB Test Kit Facility, Shanghai China':17,
                'Micro labs, Verna, Goa, India' : 18, 
                'Cipla, Kurkumbh, India': 19,
                'Emcure Plot No.P-2, I.T-B.T. Park, Phase II, MIDC, Hinjwadi, Pune, India': 20,
                'BMS Meymac, France' : 21, 
                'Ranbaxy, Paonta Shahib, India': 22,
                'Hetero, Jadcherla, unit 5, IN': 23, 
                'Bio-Rad Laboratories': 24,
                'ABBVIE GmbH & Co.KG Wiesbaden': 25, 
                'Cipla, Patalganga, India': 26,
                'Pacific Biotech, Thailand': 27, 
                'Roche Basel': 28,
                'Gilead(Nycomed) Oranienburg DE': 29, 
                'Mylan,  H-12 & H-13, India': 30}

            x['manufacturing_site_encoded'] = x['Manufacturing_Site'].replace(mapping).astype(int)
            
           
         
            
            
             
        
                
                
               
            # Country 
            mapping = {country: i for i, country in enumerate(["Côte d'Ivoire",'Zambia','Others','Nigeria','Tanzania','Mozambique','Zimbabwe','South Africa','Rwanda','Haiti','Vietnam','Uganda','Congo, DRC','Ghana','Ethiopia','Kenya','South Sudan','Sudan','Guyana','Cameroon','Dominican Republic','Burundi','Namibia','Botswana'], start=1)}
            default_code = len(mapping) + 1
            x['country_encoded'] = x['Country'].replace(mapping).fillna(default_code).astype('Int64')
          

           
           # Dosage Form 
           
            dosage_form_encoding = {"Tablet":1, "Test kid":2, "Oral":3, "Capsule":4, "Test kit": 5, "Injection": 6}

            x["Dosage_Form_encoded"] = x["Dosage_Form"].map(dosage_form_encoding).astype("Int64")
        
           
            x['First_Line_Designation_encoded'] = x['First_Line_Designation'].map({'Yes': 1, 'No': 0}).astype(int)
    
           
           
           # Ful Fill Via 
            x['fulfill_via_encoded'] = x['Fulfill_Via'].map({'From RDC': 0, 'Direct Drop': 1}).astype(int)
         
            

            # Sub_classification 
            x['Sub_Classification_encoded'] = x['Sub_Classification'].map({
                'Adult': 1,
               'Pediatric': 2,
                'HIV test': 3,
                'HIV test - Ancillary': 4,
               'Malaria': 5,
               'ACT': 6
            }).astype('Int64')


            
            x['brand_encoded'] = x['Brand'].replace({
                                                                        'Others': 1,
                                                                        'Generic': 2,
                                                                        'Determine': 3,
                                                                        'Uni-Gold': 4,
                                                                        'Stat-Pak': 5,
                                                                        'Aluvia': 6,
                                                                        'Bioline': 7,
                                                                        'Kaletra': 8,
                                                                        'Norvir': 9,
                                                                        'Colloidal Gold': 10,
                                                                        'Truvada': 11
                                                                    }).astype(int)

   
            # Dropping Columns after encoding 
            
            columns_to_drop = ['Brand', 'Fulfill_Via', 'Sub_Classification', 'First_Line_Designation', 'Country', 'Shipment_Mode','Manufacturing_Site','Dosage_Form']
            x.drop(columns=columns_to_drop, inplace=True)
            
            

            return x 
    
        except Exception as e:
            raise ApplicationException(e,sys) from e 
            
        
    def data_modification(self,x):
        try:
            
            # Manufacturing Site - "Others"
            counts = x['Manufacturing_Site'].value_counts()
            idx = counts[counts.lt(20)].index
            x.loc[x['Manufacturing_Site'].isin(idx), 'Manufacturing_Site'] = 'Others'
            
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
                                'Delivery_Recorded_Date', 'Product_Group',
                                'Vendor', 'Item_Description', 'Molecule/Test_Type', 'Dosage',
                                'Weight_(Kilograms)', 'Freight_Cost_(USD)','Line_Item_Insurance_(USD)']
            

            x.drop(columns=columns_to_drop, inplace=True)
            
            

            
            return x
        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
            
    def Missing_fills(self,x):
        try:
            # Shipment Mode - Mode 
            mode=x['Shipment_Mode'].mode()
            x['Shipment_Mode']=x['Shipment_Mode'].fillna(mode[0])
            
            # Shipment Mode - Mode 
            mode=x['Manufacturing_Site'].mode()
            x['Manufacturing_Site']=x['Manufacturing_Site'].fillna(mode[0])
            
            # Shipment Mode - Mode 
            mode=x['Country'].mode()
            x['Country']=x['Country'].fillna(mode[0])
            
            # BRand 
            mode=x['Brand'].mode()
            x['Brand']=x['Brand'].fillna(mode[0])
            
            
            logging.info(
                    f" >>>>>>>>>>>>   Missing Fills :  Shipment_Mode,Manufacturing_Site,Country,Brand <<<<<<<<<<")
            
            return x
    
            
 
        
        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
    def outlier(self,x):
        try:
            Outlier_Detection = ['Unit_of_Measure_(Per_Pack)','Line_Item_Quantity','Line_Item_Value','Pack_Price','Unit_Price','Weight_Kilograms_Clean']
            for i in Outlier_Detection:
        
                upper_limit = x[i].mean() + 3*x[i].std()
                lower_limit = x[i].mean() - 3*x[i].std()
                print(i)
                x[i] = np.where(x[i]>upper_limit,upper_limit,np.where(x[i]<lower_limit,lower_limit,x[i])
                )
            
            
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
            data = self.encoding(data)

            
            
            # Drop Columns 
            
            
            return data
            
            
            
    
        
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
            
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        try:
            X = self.data_wrangling(X)
            
            column_order=['Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity', 'Line_Item_Value',
                            'Pack_Price', 'Unit_Price', 'Freight_Cost_USD_Clean',
                            'Weight_Kilograms_Clean', 'shipment_mode_encoded',
                            'manufacturing_site_encoded', 'country_encoded', 'Dosage_Form_encoded',
                            'First_Line_Designation_encoded', 'fulfill_via_encoded',
                            'Sub_Classification_encoded', 'brand_encoded']
            
            X = X.reindex(columns=column_order)
            arr = X[column_order].values

            
                        
            

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
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY] 
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            num_pipeline = Pipeline(steps =[("impute", SimpleImputer(strategy="median")),
                                            ("scaler",StandardScaler())])

            cat_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy="most_frequent")),
                                            ])

            preprocessing = ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),('cat_pipeline',cat_pipeline,categorical_columns),
                                            ])



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

            # Extracting target column name
            target_column_name = schema[TARGET_COLUMN_KEY]

            date_columns = schema[DATE_COLUMN_KEY]
            numerical_columns = schema[NUMERICAL_COLUMN_KEY] 
            categorical_columns = schema[CATEGORICAL_COLUMN_KEY]
            string_columns=schema[STRING_COLUMN_KEY]
            
            all_columns= ['ID', 'Project Code', 'PQ', 'PO/SO', 'ASN/DN', 'Country', 'Managed By',
                        'Fulfill Via', 'Vendor INCO Term', 'Shipment Mode',
                        'PQ First Sent to Client Date', 'PO Sent to Vendor Date',
                        'Scheduled Delivery Date', 'Delivered to Client Date',
                        'Delivery Recorded Date', 'Product Group', 'Sub Classification',
                        'Vendor', 'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage',
                        'Dosage Form', 'Unit of Measure (Per Pack)', 'Line Item Quantity',
                        'Line Item Value', 'Pack Price', 'Unit Price', 'Manufacturing Site',
                        'First Line Designation', 'Weight (Kilograms)', 'Freight Cost (USD)',
                        'Line Item Insurance (USD)']
            
            
            

            
            

            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            train_df
            logging.info(f"Feature Enineering - Train Data ")
            
            
            
            feature_eng_train_arr = fe_obj.fit_transform(train_df)
            
            logging.info(f"Feature Enineering - Test Data ")
            feature_eng_test_arr = fe_obj.transform(test_df)

            
            
            col=['Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity', 'Line_Item_Value',
                'Pack_Price', 'Unit_Price', 'Freight_Cost_USD_Clean',
                'Weight_Kilograms_Clean', 'shipment_mode_encoded',
                'manufacturing_site_encoded', 'country_encoded', 'Dosage_Form_encoded',
                'First_Line_Designation_encoded', 'fulfill_via_encoded',
                'Sub_Classification_encoded', 'brand_encoded']
            
            
            
            # Converting featured engineered array into dataframe
            feature_eng_train_df = pd.DataFrame(feature_eng_train_arr,columns=col)
            
            feature_eng_test_df = pd.DataFrame(feature_eng_test_arr,columns=col)
            


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
            

            
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            column_order = ['Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity', 'Line_Item_Value',
                'Pack_Price', 'Unit_Price',
                'Weight_Kilograms_Clean', 'shipment_mode_encoded',
                'manufacturing_site_encoded', 'country_encoded', 'Dosage_Form_encoded',
                'First_Line_Designation_encoded', 'fulfill_via_encoded',
                'Sub_Classification_encoded', 'brand_encoded']
            
            
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
    
   