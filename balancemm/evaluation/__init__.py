from .precisions import BatchMetricsCalculator
def Evaluation(trainer, model, temp_model,train_dataloader, val_dataloader, optimizer, scheduler, logger):
    if temp_model is None:
        trainer(model, train_dataloader, val_dataloader, optimizer, scheduler, logger)
class ComprehensiveModelEvaluator:
    def __init__(self, args):
        self.Metrics = BatchMetricsCalculator(args['Metrics'])
        self.Complex = {}
        self.Modalitys = {}
        
