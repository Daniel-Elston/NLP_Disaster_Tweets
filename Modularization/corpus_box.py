import sys
import os
import yaml
import pickle

from nltk.corpus import stopwords
from nltk.probability import FreqDist

stop_words = set(stopwords.words('english'))
project_dir = 'C:/Users/delst/OneDrive/Desktop/Code/Workspace/NLP_Disaster_Tweets'

class TextProcessor:
    def __init__(self, df, token_col):
        self.df = df
        self.token_col = token_col
        self.stop_words = set(stopwords.words('english'))
        
        self.corpus_stopwords = None
        self.corpus_doc_raw = None
        self.corpus_word_raw = None
        self.corpus_doc = None
        self.corpus_word = None

    def remove_stop_words(self, tokens):
        remove_stopwords = [token for token in tokens if token not in self.stop_words]
        return remove_stopwords
    
    def generate_corpus(self):        
        
        self.corpus_doc_raw = self.df[self.token_col].tolist()
        self.corpus_word_raw = self.df[self.token_col].explode().tolist()
        self.corpus_stopwords = [word for word in self.corpus_word_raw if word in self.stop_words]
        
        self.df[self.token_col] = self.df[self.token_col].apply(self.remove_stop_words)

        self.corpus_doc = self.df[self.token_col].tolist()
        self.corpus_word = self.df[self.token_col].explode().tolist()
    
    def generate_bow(self, corpus):
        self.bow = set(corpus)
        self.bow_fd = FreqDist(corpus)
        
    
    @staticmethod
    def create_processor(df, target_val, token_col):
        df_target = df[df['target'] == target_val]
        processor = TextProcessor(df_target, token_col)
        processor.generate_corpus()
        return processor



def load_corpus_bow(file_index):
    filename = f'corpus_bow_save_{file_index}'
    path_to_pkl_store = os.path.join(project_dir, 'Data/Serialised_Data_Objects/Pkl', f'{filename}.pkl')

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