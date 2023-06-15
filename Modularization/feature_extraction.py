import sys
import os
import yaml
import pickle

import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(
        self,
        df:pd.DataFrame,
        token_col:str,
        ):
        """ Initial text processing class."""
        self.df = df
        self.token_col = token_col
    
    def extract_hashtags(
        self,
        tokens:str
        ):
        """ Extract hashtags from tokens."""
        return [token for token in tokens if token.startswith('#')]

    def extract_mentions(
        self,
        tokens:str
        ):
        """ Extract mentions from tokens."""
        return [token for token in tokens if token.startswith('@')]

    def extract_questions(
        self,
        tokens:str
        ):
        """ Extract questions from tokens."""
        return [token for token in tokens if token == '?']

    def extract_exclamations(
        self,
        tokens:str
        ):
        """ Extract exclamations from tokens."""
        return [token for token in tokens if token == '!']
    
    def extract(
        self
        ):
        """ Apply extractions to dataframe."""
        self.df['hashtags'] = self.df[self.token_col].apply(self.extract_hashtags)
        self.df['mentions'] = self.df[self.token_col].apply(self.extract_mentions)
        self.df['questions'] = self.df[self.token_col].apply(self.extract_questions)
        self.df['exclamations'] = self.df[self.token_col].apply(self.extract_exclamations)
        return self.df
        
    