''' 
    Author: Vu Quoc Hien
    Date created: 2022-03-14
'''
import matplotlib.pyplot as plt
from matplotlib.style import available
import torch as T
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import os

from models.SharedNet1C import SharedNet1C

class Evaluator(object):
    
    def __init__(self, model=None) -> None:
        self.model = model
        pass
    
    def precision(self, y_preds, y_trues):
        """
        Calculates the precision
        """
        return metrics.precision_score(y_trues, y_preds, average='macro')

    def recall(self, y_preds, y_trues):
        """
        Calculates the recall
        """
        return metrics.recall_score(y_trues, y_preds, average='macro')

    def accuracy(self, y_preds, y_trues):
        """
        Calculates the accuracy
        """
        return metrics.accuracy_score(y_trues, y_preds)

    def roc_auc(self, y_preds, y_trues):
        """
        Calculates the area under the ROC curve for multi class classification
        """
        return metrics.roc_auc_score(y_trues, y_preds, average='macro')

    def confusion_matrix(self, y_preds, y_trues):
        return metrics.confusion_matrix(y_trues, y_preds)

    def get_all_metrics(self, y_preds, y_trues, metric_names='all'):
        metrics = {}
        if metric_names == 'all':
            metrics['precision'] = self.precision(y_preds, y_trues)
            metrics['recall'] = self.recall(y_preds, y_trues)
            metrics['accuracy'] = self.accuracy(y_preds, y_trues)
        else:
            for metric in metric_names:
                if metric == 'precision':
                    metrics[metric] = self.precision(y_preds, y_trues)
                elif metric == 'recall':
                    metrics[metric] = self.recall(y_preds, y_trues)
                elif metric == 'accuracy':
                    metrics[metric] = self.accuracy(y_preds, y_trues)
                elif metric == 'roc_auc':
                    metrics[metric] = self.roc_auc(y_preds, y_trues)
                elif metric == 'confusion_matrix':
                    metrics[metric] = self.confusion_matrix(y_preds,y_trues)
                    sns.heatmap(metrics[metric],cmap='coolwarm',annot=True,)
                    if not os.path.exists('output/evaluation'):
                        os.makedirs('output/evaluation')
                    plt.savefig('output/evaluation/confusion_matrix.png')
                    plt.close()
        return metrics
    
    def evaluate(self, dataset: dict,  metric_names='all') -> dict:
        y_trues = {}
        y_preds = {}
        self.model._eval()
        metrics = {}
        
        for name, output_size in self.model.outputs_size:
            y_trues[name] = []
            y_preds[name] = []
            metrics[name] = {}
        
        with T.no_grad():		# Disable gradient calculation
            for _iter, (input, target) in enumerate(zip(dataset[0], dataset[1])):		# L???y s??? v??ng l???p, data v?? target l???n l?????t trong tqdm
                output = self.model(input).flatten(0)
                presize = 0
                for name, output_size in self.model.outputs_size:
                    allele_out = output[presize:presize + output_size].cpu().numpy().argsort()[-1]
                    allele_target = target[presize:presize + output_size].cpu().numpy().argsort()[-1]
                   # y_true = np.zeros(output_size)
                    #y_true[allele_target[0]] = 1
                    #if target[presize + allele_target[1]] == 1:
                     #   y_true[allele_target[1]] = 1
                        
                    #y_pred = np.zeros(output_size)
                    #y_pred[allele_out] = 1
                    y_trues[name].append(allele_target)
                    y_preds[name].append(allele_out)
                    presize += output_size
        for name, output_size in self.model.outputs_size:
            metrics[name] = self.get_all_metrics(y_preds[name], y_trues[name], metric_names=metric_names)
        
        return metrics
        