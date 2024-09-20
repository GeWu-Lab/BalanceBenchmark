import numpy as np
from sklearn.metrics import confusion_matrix

class BatchMetricsCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.total_samples = 0

    def update(self, y_true, y_pred):
        batch_cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        self.confusion_matrix += batch_cm
        self.total_samples += len(y_true)

    def compute_metrics(self):
        # 计算准确率
        accuracy = np.sum(np.diag(self.confusion_matrix)) / self.total_samples

        # 计算每个类别的F1分数
        fps = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fns = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tps = np.diag(self.confusion_matrix)
        
        precisions = tps / (tps + fps)
        recalls = tps / (tps + fns)
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        macro_f1 = np.mean(f1_scores)

        return accuracy, f1_scores, macro_f1