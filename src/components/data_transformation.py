import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import statsmodels.api as sm

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        # new code
        # self.selected_features = None

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            outliers = ['MS_SubClass', 'Lot_Frontage', 'Lot_Area', 'Overall_Qual',
                        'Overall_Cond', 'Year_Built', 'Mas_Vnr_Area', 'BsmtFin_SF_One',
                        'BsmtFin_SF_Two', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF',
                        'Second_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath',
                        'Bsmt_Half_Bath', 'Full_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr',
                        'TotRms_AbvGrd', 'Fireplaces', 'Garage_Yr_Blt', 'Garage_Cars',
                        'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch',
                        'Ssn_Porch', 'Screen_Porch', 'Pool_Area', 'Misc_Val']
            
            no_outliers_num = ['Year_Remod', 'Half_Bath', 'Mo_Sold', 'Yr_Sold']

            cat = ['MS_Zoning', 'Street', 'Lot_Shape', 'Land_Contour',
                   'Lot_Config', 'Land_Slope', 'Neighborhood', 'Conition_One',
                   'Condition_Two', 'Bldg_Type', 'House_Style', 'Roof_Style', 'Roof_Matl',
                   'Exterior_First', 'Exterior_Second', 'Mas_Vnr_Type', 'Exter_Qual',
                   'Exter_Cond', 'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure',
                   'BsmtFin_Type_One', 'BsmtFin_Type_Two', 'Heating', 'Heating_QC',
                   'Central_Air', 'Electrical', 'Kitchen_Qual', 'Functional',
                   'Fireplace_Qu', 'Garage_Type', 'Garage_Finish', 'Garage_Qual',
                   'Garage_Cond', 'Paved_Drive', 'Sale_Type', 'Sale_Condition']
            
            outliers_pipeline= Pipeline( steps=
                                        [("imputer",SimpleImputer(missing_values = np.nan, strategy="median")),
                                         ("rs", RobustScaler())] )
            
            no_outliers_num_pipeline = Pipeline( steps=
                                        [("imputer",SimpleImputer(missing_values = np.nan, strategy="mean")),
                                         ("ss", StandardScaler())] )

            cat_pipeline = Pipeline( steps=
                                  [ ('imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
                                   ('ohe', OneHotEncoder())
                                   ])
            
            preprocessor = ColumnTransformer(
                [
                    ("outliers_pipeline", outliers_pipeline, outliers),
                    ("no_outliers_num_pipeline", no_outliers_num_pipeline, no_outliers_num),
                    ("cat_pipeline", cat_pipeline, cat)
                ]
            )


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train = pd.read_csv(train_path)

            logging.info("Read train data")
            
            test = pd.read_csv(test_path)

            logging.info("Read test data")

            x_train_transf = train.drop('SalePrice',axis=1)

            logging.info("Dropped target column from the train set to make the input data frame for model training")

            constant_columns_x_train_transf = x_train_transf.columns[x_train_transf.nunique() == 1]

            if len(constant_columns_x_train_transf) > 0:
                 print("Constant columns found in x_train_transf:", constant_columns_x_train_transf)
            else:
                print("No constant columns found in x_train_transf.")

            logging.info("Checked for constant columns in x_train_transf")

            y_train_transf = train['SalePrice']

            logging.info("Target feature obtained for model training")

            x_test_transf = test.drop('SalePrice', axis=1)

            logging.info("Dropped target column from the test set to make the input data frame for model testing")

            constant_columns_x_test_transf = x_test_transf.columns[x_test_transf.nunique() == 1]

            if len(constant_columns_x_test_transf) > 0:
                 print("Constant columns found in x_test_transf:", constant_columns_x_test_transf)
            else:
                print("No constant columns found in x_test_transf.")

            logging.info("Checked for constant columns in x_test_transf")      
        
            y_test_transf = test['SalePrice']

            logging.info("Target feature obtained for model testing")

            preprocessor = self.get_data_transformer_object()
            
            logging.info("Preprocessing object obtained")

            x_train_transf_preprocessed = preprocessor.fit_transform(x_train_transf)

            logging.info("Preprocessor applied on x_train_transf")

            x_train_transf_preprocessed_df = pd.DataFrame(x_train_transf_preprocessed)

            logging.info("x_train_transf dataframe formed for backwards elimination")

            for i in range(len(x_train_transf_preprocessed_df.columns)):
                
                x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.rename(columns={x_train_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info("x_train_transf dataframe columns renamed")

            print ("x_train_transf_preprocessed_df shape before be:", x_train_transf_preprocessed_df.shape)

            # print ("x_train_transf_preprocessed_df columns before be:", x_train_transf_preprocessed_df.columns)

            constant_columns_x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.columns[x_train_transf_preprocessed_df.nunique() == 1]

            if len(constant_columns_x_train_transf_preprocessed_df) > 0:
                 print("Constant columns found in x_train_transf_preprocessed_df:", constant_columns_x_train_transf_preprocessed_df)
            else:
                print("No constant columns found in x_train_transf_preprocessed_df.")

            x_train_transf_be = sm.add_constant(x_train_transf_preprocessed_df)  # Add a constant column for intercept

            # print ("x_train_transf_be columns after adding constant:", x_train_transf_be.columns)

            logging.info("Constant column added in x_train_transf for initiating backwards elimination")
            
            model = sm.OLS(y_train_transf, x_train_transf_be)

            logging.info("x_train_transf_be and y_train_transf fitted on the backwards elimintaion model")
            
            results = model.fit()

            logging.info("Backwards elimination results obtained")

            while True:
                 p_values = results.pvalues[1:]  # Exclude the constant column
                 max_p_value = p_values.max()
                 
                 if max_p_value > 0.05:
                      max_p_index = p_values.idxmax()
                      x_train_transf_be = x_train_transf_be.drop(columns=max_p_index)
                      model = sm.OLS(y_train_transf, x_train_transf_be)
                      results = model.fit()
                 else:
                      break
            
            logging.info("Backwards elimination performed")

            x_train_transf_be = x_train_transf_be.drop('const', axis =1)

            # print ("x_train columns after performing be and dropping the constant added for be:", x_train_transf_be.columns)

            logging.info("Features seleted for preprocessor after performing backwards elimination")

            # Print the final selected features
            selected_features = x_train_transf_be.columns

            print ("selected fetures:", selected_features)

            print ("selected fetures len:", len(selected_features))

            logging.info("Features selceted after performing backwards elimination")
             
            all_features = x_train_transf_preprocessed_df.columns
            
            not_selected_features = set(all_features) - set(selected_features)

            logging.info("Not selected features obtained")

            x_test_transf_preprocessed = preprocessor.transform(x_test_transf)

            logging.info("Preprocessor applied on x_test_transf")

            x_test_transf_preprocessed_df = pd.DataFrame(x_test_transf_preprocessed)

            logging.info("x_test_transf dataframe formed for backwards elimination")

            for i in range(len(x_test_transf_preprocessed_df.columns)):
                
                x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.rename(columns={x_test_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info("x_test_transf dataframe columns renamed")

            constant_columns_x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.columns[x_test_transf_preprocessed_df.nunique() == 1]

            if len(constant_columns_x_test_transf_preprocessed_df) > 0:
                 print("Constant columns found in x_test_transf_preprocessed_df:", constant_columns_x_test_transf_preprocessed_df)
            else:
                print("No constant columns found in x_test_transf_preprocessed_df.")
            
            x_test_transf_be = x_test_transf_preprocessed_df.drop(not_selected_features, axis=1)

            logging.info("not selected features dropped from x_test_transf")

            # print ("x_train_transf_be columns:", x_train_transf_be.columns)

            # print ("x_train_transf_be shape:", x_train_transf_be.shape)

            # print ("x_test_transf_be columns:", x_test_transf_be.columns)

            # print ("x_test_transf_be shape:", x_test_transf_be.shape)

            train_arr = np.c_[np.array(x_train_transf_be), np.array(y_train_transf)]
            
            logging.info("Combined the input features and target feature of the train set as an array.")
            
            test_arr = np.c_[np.array(x_test_transf_be), np.array(y_test_transf)]
            
            logging.info("Combined the input features and target feature of the test set as an array.")
            
            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor)
            
            logging.info("Saved preprocessing object.")
            
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,)
        
        except Exception as e:
            raise CustomException(e, sys)