import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', grid=True)
import seaborn as sns



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


class BinaryClassifierEvaluator:
    def __init__(self, y_true, y_pred, y_scores):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        
    def create_classification_report(self):
        cr = classification_report(self.y_true, self.y_pred, output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()
        cr_df = cr_df.rename({
            '0' : 'No Disaster',
            '1' : 'Disaster'
            }, axis=0)
        cr_df = cr_df.round(2)
        
        print("Classification Report:")
        return cr_df
    
    def visualise_confusion_matrix(self):
        labels = ['No Disaster','Disaster']
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)

        plt.rc('axes', grid=False)
        plt.figure(figsize=(12,8))

        heatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', annot_kws={"fontsize": 16})
        heatmap.set_xlabel('Predicted Values')
        heatmap.set_ylabel('True Values ')
        plt.show()
    
    def visualise_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12,8))
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Line of No Discrimination (Random Guessing)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate (1 - specificity)')
        plt.ylabel('True Positive Rate (sensitivity)')
        plt.legend(loc="lower right")
        plt.show()
