import torch.optim.lr_scheduler as lr_scheduler

def find_scheduler(scheduler_type):
    if hasattr(lr_scheduler, scheduler_type):
        return getattr(lr_scheduler, scheduler_type)
    else:
        raise ValueError(f'Scheduler {scheduler_type} not found.')

def create_scheduler(optimizer, args: dict):
    if 'type' not in args:
        raise ValueError('Scheduler type is required.')
    scheduler_type = args['type']
    scheduler_cls = find_scheduler(scheduler_type)
    scheduler_args = {k: v for k, v in args.items() if k != 'type'}
    scheduler = scheduler_cls(optimizer, **scheduler_args)
    print (f'Scheduler {scheduler.__class__.__name__} - {scheduler_type} is created.')
    return scheduler
