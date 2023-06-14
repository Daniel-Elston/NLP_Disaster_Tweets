import sys
import os

import pandas as pd
import numpy as np

import string
import re
import nltk

import contractions
from nltk.tokenize import TweetTokenizer


class InitialDataProcessing:
    def __init__(self, 
                 df:pd.DataFrame
                 ):
        """
        Processing class for minor processing steps.

        Args:
            df (pd.DataFrame): Dataframe to be processed.
        """             
        self.df = df
        
    def transform_dtypes(self,
                         to_dtype:dict,
                        ):
        """
        Transform dtypes of columns in dataframe.

        Args:
            to_dtype (dict): What to change dtypes to

        Returns:
            pd.DataFrame: Corrected dytpe dataframe
        """        
        
        for col, dtype in to_dtype.items():
            self.df[col] = self.df[col].astype(dtype)
        return self.df
    
    def dup_nan_drop(self,
                     drop_cols:list
                     ):
        """
        Drop useless data from dataframe.

        Args:
            drop_cols (list): Columns to drop
        """        
        self.df = self.df.drop(drop_cols, axis=1)
        self.df = self.df.dropna()
        return self.df
    

class InitialTextProcessing:
    def __init__(self, 
                 df:pd.DataFrame,
                 text_col:str,
                 token_col:str,
                 ):
        """
        Initial text processing class.

        Args:
            df (pd.DataFrame): Dataframe to be processed.
        """        
        self.df = df
        self.text_col = text_col
        self.token_col = token_col
    
    def transform_to_lowercase(self, 
                               ):
        """
        Make all characters in a column lowercase.

        Args:
            lower_c_cols (str): Column to make lowercase

        Returns:
            pd.DataFrame: Corrected lowercase dataframe 
        """        
        
        self.df[self.text_col] = self.df[self.text_col].str.lower()
        return self.df
    
    def remove_chars(self,
                     text:str
                     ):
        """
        Remove desired characters from text.

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """    
        cleaned_text = re.sub(r'[^\w\s!?#@\\\\]|[\n]', '', text)
        return cleaned_text
    
    def apply_remove_chars(self, 
                           ):
        self.df[self.text_col] = self.df[self.text_col].apply(self.remove_chars)
        return self.df

    def remove_urls(self,
                    ):
        self.df[self.text_col] = self.df[self.text_col].str.replace(r'\s*http?://\S+(\s+|$)', ' ').str.strip()
        return self.df

    def fix_contractions(self,
                         ):
        self.df[self.text_col] = self.df[self.text_col].apply(lambda x: contractions.fix(x))
        # self.df[self.text_col] = self.df[self.text_col].apply(contractions.fix)
        return self.df
    
    def tokenize_text(self, tokenizer):
        self.df[self.token_col] = self.df[self.text_col].apply(tokenizer.tokenize)
        return self.df
    
    