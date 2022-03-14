''' 
    Author: Vu Quoc Hien
    Date created: 2022-03-14
'''
import numpy as np
import sklearn.metrics as metrics

class Evaluator(object):
    
    def __init__(self) -> None:
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
        Calculates the area under the ROC curve
        """
        return metrics.roc_auc_score(y_trues, y_preds, average='macro')

    def get_all_metrics(self, outputs, metrics='all'):
        y_preds = outputs[:, 0]
        y_trues = outputs[:, 1]
        metrics = {}
        if metrics == 'all':
            metrics['precision'] = self.precision(y_preds, y_trues)
            metrics['recall'] = self.recall(y_preds, y_trues)
            metrics['accuracy'] = self.accuracy(y_preds, y_trues)
            metrics['roc_auc'] = self.roc_auc(y_preds, y_trues)
        else:
            for metric in metrics:
                if metric == 'precision':
                    metrics[metric] = self.precision(y_preds, y_trues)
                elif metric == 'recall':
                    metrics[metric] = self.recall(y_preds, y_trues)
                elif metric == 'accuracy':
                    metrics[metric] = self.accuracy(y_preds, y_trues)
                elif metric == 'roc_auc':
                    metrics[metric] = self.roc_auc(y_preds, y_trues)
        return metrics