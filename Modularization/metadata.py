
import pandas as pd


class DataDictionary:
    """
    Class designed to generate a data dictionary for a given DataFrame.
    
    General overviews of the data dictionary are:
    - Raw_dtype: The dtype of the column as read in by pandas
    - True_dtype: The dtype of the column after transformation
    - Length: The number of rows in the column
    - Null_Count: The number of null values in the column
    - Memory (MB): The amount of memory the column takes up in MB
    - Definition: The definition of the column
    """    

    def __init__(self, 
                 df:pd.DataFrame
                 ):
        """
        Initialize the class with a DataFrame

        Args:
            df (pd.DataFrame): DataFrame to generate a data dictionary for
        """        
        
        self.df = df
   

    def make_my_data_dictionary(self,
                                raw_dtype:dict,
                                true_dtype:dict,
                                dd_descriptions:dict,
                                ):
        """
        Generate a data dictionary for a given DataFrame

        Args:
            raw_dtype (dict): dtypes given with the raw data
            true_dtype (dict): Corrected dtypes
            dd_descriptions (dict): Descriptions of each column

        Returns:
            df_DD (pd.DataFrame): Data dictionary for the given DataFrame 
        """        

        df_cols = self.df.columns
        df_DataDict = {}

        for col in df_cols:
                df_DataDict[col] = {
                               'Raw_dtype': str(self.df.dtypes[col]),
                               'True_dtype': true_dtype[col],
                               'Length': len(self.df[col]),
                               'Null_Count': sum(self.df[col].isna()),
                               'Memory (MB)': round(self.df.memory_usage(deep=True)/1000000, 3)[col],
                               'Definition': dd_descriptions[col],
                                }

        df_DD = pd.DataFrame(df_DataDict)
        return df_DD
    

class BasicMetaData:
    def __init__(self, 
                 df:pd.DataFrame
                 ):
        self.df = df
        """
        Class to generate basic metadata for a given DataFrame

        Args:
            df (pd.DataFrame): DataFrame to generate metadata for
        """
        
    def generate_basic_metadata(self,
                                df:pd.DataFrame
                                ):
        nan_count = self.df.isna().sum()
        nan_percent = round(nan_count / len(self.df) * 100, 2)
        
        target_vcs = self.df.target.value_counts()
        target_percent = round(target_vcs / len(self.df) * 100, 2)
        
        duplicate_count = self.df.duplicated().sum()
        duplicate_percent = round(duplicate_count / len(self.df) * 100, 2)
        
        metadata = {
            'nan_count': nan_count,
            'nan_percent': nan_percent,
            'target_count': target_vcs,
            'target_percent': target_percent,
            'duplicate_count': duplicate_count,
            'duplicate_percent': duplicate_percent
        }
        
        return metadata