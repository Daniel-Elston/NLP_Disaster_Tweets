import sys
import os
import yaml
import pickle


class FeatureExtractor:
    def __init__(self, df, token_col):
        self.df = df
        self.token_col = token_col
    
    def extract_hashtags(self, tokens):
        return [token for token in tokens if token.startswith('#')]

    def extract_mentions(self, tokens):
        return [token for token in tokens if token.startswith('@')]

    def extract_questions(self, tokens):
        return [token for token in tokens if token == '?']

    def extract_exclamations(self, tokens):
        return [token for token in tokens if token == '!']
    
    def extract(self):
        self.df['hashtags'] = self.df[self.token_col].apply(self.extract_hashtags)
        self.df['mentions'] = self.df[self.token_col].apply(self.extract_mentions)
        self.df['questions'] = self.df[self.token_col].apply(self.extract_questions)
        self.df['exclamations'] = self.df[self.token_col].apply(self.extract_exclamations)
        return self.df
        
    