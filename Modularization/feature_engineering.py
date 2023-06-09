import sys
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('axes', grid=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

class FeatureEngineer:
    def __init__(
        self, 
        df: pd.DataFrame, 
        token_col: str,
        text_col: str, 
        sia: SentimentIntensityAnalyzer
        ):
        """ Initialises FeatureEngineer class. """
        self.df = df
        self.token_col = 'tokens'
        self.text_col = 'text'
        self.sia = SentimentIntensityAnalyzer()
    
    def count_occurrences(
        self,
        character:str,
        token_list:list
        ):
        """ Counts occurrences of a character in a list of tokens. """
        count = sum(1 for token in token_list if token.startswith(character))
        return count
    
    def generate_counts(
        self
        ):
        """ Generates counts for hashtags, mentions, questions and exclamations. """
        self.df['num_hashtags'] = self.df[self.token_col].apply(lambda x: self.count_occurrences('#', x))
        self.df['num_mentions'] = self.df[self.token_col].apply(lambda x: self.count_occurrences('@', x))
        self.df['num_questions'] = self.df[self.token_col].apply(lambda x: self.count_occurrences('?', x))
        self.df['num_exclamations'] = self.df[self.token_col].apply(lambda x: self.count_occurrences('!', x))

        self.df['num_words'] = self.df[self.text_col].apply(lambda x: len(x.split()))
        self.df['num_letters'] = self.df[self.text_col].apply(lambda x: len(x))
        return self.df
    
    
    def get_sentiment_score(
        self,
        text:str
        ):
        """ 
        Gets sentiment score for a tweet.
        VADER sentiment analysis tool is used to calculate the sentiment score.
        """
        sentiment = self.sia.polarity_scores(text)
        return sentiment['compound']
    
    def generate_sentiment_score(
        self
        ):
        """ Generates sentiment score for each tweet. """
        self.df['sentiment_score'] = self.df[self.text_col].apply(lambda x: self.get_sentiment_score(x))
        return self.df