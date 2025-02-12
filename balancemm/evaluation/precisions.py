import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict

class BatchMetricsCalculator:
    def __init__(self, num_classes: int, modalitys: list):
        self.num_classes = num_classes
        self.confusion_matrix = {}
        self.modalitys = modalitys
        for modality in modalitys:
            self.confusion_matrix[modality] = np.zeros((num_classes, num_classes))
        self.total_samples = 0

    def update(self, y_true, y_pred):
        for modality in self.confusion_matrix.keys():
            batch_cm = confusion_matrix(y_true, y_pred[modality].cpu(), labels=range(self.num_classes))
            self.confusion_matrix[modality] += batch_cm
        self.total_samples += len(y_true)

    def compute_metrics(self):
        # calculate accuracy
        Metrics_res = defaultdict(dict)
        for modality in self.confusion_matrix.keys():
            accuracy = np.sum(np.diag(self.confusion_matrix[modality])) / self.total_samples

            # calculate f1 score of each class
            fps = self.confusion_matrix[modality].sum(axis=0) - np.diag(self.confusion_matrix[modality])
            fns = self.confusion_matrix[modality].sum(axis=1) - np.diag(self.confusion_matrix[modality])
            tps = np.diag(self.confusion_matrix[modality])
            precisions = np.divide(tps, tps + fps, out=np.zeros_like(tps, dtype=float), where=(tps + fps) != 0)
            recalls = np.divide(tps, tps + fns, out=np.zeros_like(tps, dtype=float), where=(tps + fns) != 0)
            f1_scores = np.divide(2 * (precisions * recalls), precisions + recalls, 
                              out=np.zeros_like(precisions, dtype=float), 
                              where=(precisions + recalls) != 0)
            Metrics_res['f1'][modality] = np.mean(f1_scores)
            Metrics_res['acc'][modality] = accuracy
        return Metrics_res
    def ClearAll(self):
        for modality in self.modalitys:
            self.confusion_matrix[modality] = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
        