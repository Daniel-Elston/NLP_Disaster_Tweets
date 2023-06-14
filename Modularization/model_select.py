import sys
import os
import pickle

import pandas as pd
import numpy as np

import time
from sklearn.model_selection import cross_validate


class ModelSelection:
    def __init__(self, models, X, y, cv):
        self.models = models
        self.X = X
        self.y = y
        self.cv = cv
        self.report_df = None
    
    def evaluate_models(self):
        rows = {}
        for name, model in self.models.items():
            
            start_time = time.time()
            cv_results = cross_validate(model, self.X, self.y, cv=self.cv, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])            
            end_time = time.time()
            
            train_time = end_time - start_time
            
            rows[name] = {
                'accuracy': cv_results['test_accuracy'].mean(),
                'precision': cv_results['test_precision_weighted'].mean(),
                'recall': cv_results['test_recall_weighted'].mean(),
                'f1_score': cv_results['test_f1_weighted'].mean(),
                'training_time': train_time 

            }
        self.report_df = rows
    
    def print_report(self):
        for name, row in self.report_df.items():
            print(f"Model: {name}")
            print(f"accuracy: {row['accuracy']:.4f}")
            print(f"precision: {row['precision']:.4f}")
            print(f"recall: {row['recall']:.4f}")
            print(f"f1_score: {row['f1_score']:.4f}")
            print(f"training_time: {row['training_time']:.4f} seconds")
            print()
        
    def get_best_model(self, metric='accuracy'):
        best_model_name = max(self.report_df, key=lambda x: self.report_df[x][metric])
        return best_model_name
    
