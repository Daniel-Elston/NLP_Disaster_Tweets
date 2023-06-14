import sys
import os
import pickle

import pandas as pd
import numpy as np


from nltk.stem import WordNetLemmatizer

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# class Preprocessor:
#     def __init__(self, df, tokens):
#         self.df = df
#         self.tokens = tokens
#         self.lemmatizer = WordNetLemmatizer()
        
#     def lemmatize_tokens(self):
#         lemm_tokens = [self.lemmatizer.lemmatize(word) for word in self.tokens]
#         return lemm_tokens
    
#     def apply_lemmatizer(self):
#         self.df['lemmatized_tokens'] = self.df[self.tokens].apply(self.lemmatize_tokens)
#         return self.df
    
    

class Preprocessor:
    def __init__(self, df, tokens_col):
        self.df = df
        self.tokens_col = tokens_col
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_tokens(self, tokens):
        lemm_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemm_tokens

    def apply_lemmatizer(self):
        self.df['lemmatized_tokens'] = self.df[self.tokens_col].apply(self.lemmatize_tokens)
        return self.df



# class Embeddings:
#     def __init__(self, corpus, embed_model, num_features):
#         self.corpus = corpus
#         self.embed_model = embed_model
#         self.num_features = num_features
        
#     # Calculate the average word vector for a list of words
#     def average_word_vectors(self, words):
#         feature_vector = np.zeros((self.num_features,), dtype="float64")
#         nwords = 0.
#         for word in words:
#             if word in self.embed_model:
#                 nwords = nwords + 1.
#                 feature_vector = np.add(feature_vector, self.embed_model[word])
#         if nwords:
#             feature_vector = np.divide(feature_vector, nwords)
#         return feature_vector

#     # Apply average word vector calculation to a list of tokenized documents (corpus)
#     def averaged_word_vectorizer(self):
#         vocabulary = set(self.embed_model.index_to_key)
#         features = [self.average_word_vectors(tokenized_sentence) for tokenized_sentence in self.corpus]
#         return np.array(features)



class Embeddings:
    def __init__(self, embed_model, num_features):
        self.embed_model = embed_model
        self.num_features = num_features
        
    # Calculate the average word vector for a list of words
    def average_word_vectors(self, words):
        feature_vector = np.zeros((self.num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            if word in self.embed_model:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, self.embed_model[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    # Apply average word vector calculation to a list of tokenized documents (corpus)
    def averaged_word_vectorizer(self, corpus):
        vocabulary = set(self.embed_model.index_to_key)
        features = [self.average_word_vectors(tokenized_sentence) for tokenized_sentence in corpus]
        return np.array(features)
