import torch.optim as optim

def find_module(optimizer_type):
    if hasattr(optim, optimizer_type):
        return getattr(optim, optimizer_type)
    # elif globals().get(optimizer_type + 'Optimizer'):
    #     return globals()[optimizer_type + 'Optimizer']
    else:
        raise ValueError(f'Optimizer {optimizer_type} not found in torch.optim or current module.')

def create_optimizer(model, args: dict):
    if 'type' not in args:
        raise ValueError('Optimizer type is required.')
    optimizer_type = args['type']
    optimizer_cls = find_module(optimizer_type)
    optimizer_args = {k: v for k, v in args.items() if k != 'type'}
    optimizer = optimizer_cls(model.parameters(), **optimizer_args)
    print (f'Optimizer {optimizer.__class__.__name__} - {optimizer_type} is created.')
    return optimizer