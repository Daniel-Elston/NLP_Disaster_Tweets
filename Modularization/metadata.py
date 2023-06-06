
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