import sys
import os

import pandas as pd


class InitialProcessing:
    def __init__(self, 
                 df
                 ):
        
        self.df = df
    
    def transform_dtypes(self,
                        to_dtype:dict,
                        ):
    
        
        for col, dtype in to_dtype.items():
            self.df[col] = self.df[col].astype(dtype)
        return self.df