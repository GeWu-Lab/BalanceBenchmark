def find_module(encoder):
    if hasattr(optim, optimizer_type):
        return getattr(optim, optimizer_type)
    # elif globals().get(optimizer_type + 'Optimizer'):
    #     return globals()[optimizer_type + 'Optimizer']
    else:
        raise ValueError(f'Optimizer {optimizer_type} not found in torch.optim or current module.')