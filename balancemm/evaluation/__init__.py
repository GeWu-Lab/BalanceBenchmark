from complex import get_model_complexity
def Evaluation(trainer, model, temp_model,train_dataloader, val_dataloader, optimizer, scheduler, logger):
    if temp_model is None:
        trainer(model, train_dataloader, val_dataloader, optimizer, scheduler, logger)