import sys
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('axes', grid=True)

from nltk.probability import FreqDist
import contractions

from nltk.tokenize import TweetTokenizer

# from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

sdo_pkl = 'C:/Users/delst/OneDrive/Desktop/Code/Workspace/NLP_Disaster_Tweets/Data/Serialised_Data_Objects/Pkl'

class CorpusBowCreator:
    def __init__(
        self,
        df,
        token_col
        ):
        """ Initialises CorpusBowCreator class. """
        self.df = df
        self.token_col = token_col
        self.stop_words = set(stopwords.words('english'))
        
        self.corpus_stopwords = None
        self.corpus_doc_raw = None
        self.corpus_word_raw = None
        self.corpus_doc = None
        self.corpus_word = None

    def remove_stop_words(
        self,
        tokens
        ):
        """ Removes stopwords from tokens. """
        remove_stopwords = [token for token in tokens if token not in self.stop_words]
        return remove_stopwords
    
    def generate_corpus(
        self
        ):        
        """ Generates corpus by document, word or stopword."""
        self.corpus_doc_raw = self.df[self.token_col].tolist()
        self.corpus_word_raw = self.df[self.token_col].explode().tolist()
        self.corpus_stopwords = [word for word in self.corpus_word_raw if word in self.stop_words]
        
        self.df[self.token_col] = self.df[self.token_col].apply(self.remove_stop_words)

        self.corpus_doc = self.df[self.token_col].tolist()
        self.corpus_word = self.df[self.token_col].explode().tolist()
    
    def generate_bow(
        self,
        corpus:list
        ):
        """ Generates bag of words from corpus. """
        self.bow = set(corpus)
        self.bow_fd = FreqDist(corpus)
        
    
    @staticmethod
    def create_corpus(
        df:pd.DataFrame,
        target_val:int,
        token_col:str
        ):
        """ Creates corpus from dataframe. """
        df_target = df[df['target'] == target_val]
        processor = CorpusBowCreator(df_target, token_col)
        processor.generate_corpus()
        return processor


class CorpusBowCreatorSingle:
    def __init__(
        self,
        df:pd.DataFrame,
        token_col:str
        ):
        """ Initialises CorpusBowCreator class (disregard target). """
        self.df = df
        self.token_col = token_col
        self.stop_words = set(stopwords.words('english'))
        
        self.corpus_stopwords = None
        self.corpus_doc_raw = None
        self.corpus_word_raw = None
        self.corpus_doc = None
        self.corpus_word = None

    def remove_stop_words(
        self, 
        tokens:list
        ):
        remove_stopwords = [token for token in tokens if token not in self.stop_words]
        return remove_stopwords
    
    def generate_corpus(
        self
        ):        
        
        self.corpus_doc_raw = self.df[self.token_col].tolist()
        self.corpus_word_raw = self.df[self.token_col].explode().tolist()
        self.corpus_stopwords = [word for word in self.corpus_word_raw if word in self.stop_words]
        
        self.df[self.token_col] = self.df[self.token_col].apply(self.remove_stop_words)

        self.corpus_doc = self.df[self.token_col].tolist()
        self.corpus_word = self.df[self.token_col].explode().tolist()
    
    def generate_bow(
        self,
        corpus:list
        ):
        self.bow = set(corpus)
        self.bow_fd = FreqDist(corpus)

    @staticmethod
    def create_corpus(
        df,
        token_col:str
        ):
        processor = CorpusBowCreatorSingle(df, token_col)
        processor.generate_corpus()
        return processor

def load_corpus_bow(
    file_index:str
    ):
    filename = f'corpus_bow_save_{file_index}'
    path_to_pkl_store = os.path.join(sdo_pkl, f'{filename}.pkl')

    with open(path_to_pkl_store, 'rb') as file:
        if file_index == 'sw':
            corpus_stopwords, bow_stopwords, bow_fd_stopwords = pickle.load(file)
            return corpus_stopwords, bow_stopwords, bow_fd_stopwords
        elif file_index == '1':
            corpus_doc_1, corpus_word_1, bow_1, bow_fd_1 = pickle.load(file)
            return corpus_doc_1, corpus_word_1, bow_1, bow_fd_1
        elif file_index == '0':
            corpus_doc_0, corpus_word_0, bow_0, bow_fd_0 = pickle.load(file)
            return corpus_doc_0, corpus_word_0, bow_0, bow_fd_0
        else:
            raise ValueError("Invalid file index.")
        
        