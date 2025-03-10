import sys, yaml
from os import path as osp
import os

def add_to_pythonpath(path):
    """
    Add a directory to PYTHONPATH environment variable.
    
    Args:
        path (str): The directory path to be added to PYTHONPATH.
    """
    # Get the current value of PYTHONPATH or an empty string if not set
    pythonpath = os.environ.get('PYTHONPATH', '')

    # Add the new directory to PYTHONPATH
    pythonpath += os.pathsep + path
    print(pythonpath)
    # Update the PYTHONPATH environment variable
    os.environ['PYTHONPATH'] = pythonpath
add_to_pythonpath(os.getcwd())
from balancemm.utils.parser_utils import parse_cli_args_to_dict, load_config_dict, ensure_and_get_config_path
from balancemm.train import train_and_test, linear_probe_eval
from balancemm.test import test, t_sne

from lightning import fabric
import datetime
from .utils.train_utils import set_seed
root_path = osp.dirname(osp.dirname(__file__))

def args_split(args: list):
    arg = {}
    for i in range(len(args)):
        if args[i].startswith('--umodel'):
            name = args[i].split('.')[1]
            args['model'][name] = args[i+1]
        if args[i].startswith('--utrainer'):
            name = args[i].split('.')[1]
            args['trainer'][name] = args[i+1]


def create_config(config_dict: dict, args): 
    """Create configuration from cli and yaml."""
    # with open(osp.join(root_path ,"configs", "global_config.yaml"), 'r') as f:
    #     global_settings = yaml.safe_load(f)
    with open(osp.join(root_path ,"configs", "dataset_config.yaml"), 'r') as f:
        dataset_settings = yaml.safe_load(f)
    with open(osp.join(root_path ,"configs", "trainer_config.yaml"), 'r') as f:
        trainer_settings = yaml.safe_load(f)
    with open(osp.join(root_path ,"configs", "model_config.yaml"), 'r') as f:
        model_settings = yaml.safe_load(f)
    with open(osp.join(root_path ,"configs", "encoder_config.yaml"), 'r') as f:
        encoder_settings = yaml.safe_load(f)
    # config_dict = global_settings | config_dict
    if args.device :
        config_dict['Main_config']['device'] = args.device
    # config_dict['fabric']['devices'] = list(map(int,config_dict['Main_config']['device'].split(',')))
    gpu_ids = list(map(int,config_dict['Main_config']['device'].split(',')))
    gpu_counts = len(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    config_dict['fabric']['devices'] = [i for i in range(gpu_counts)]
    if isinstance(config_dict['train']['parameter']['base_lr'], str):
        config_dict['train']['parameter']['base_lr'] = float(config_dict['train']['parameter']['base_lr'])
    print('==================')
    print(config_dict)
    # print('==================')
    # print(global_settings)
    if args.model :
        config_dict['Main_config']['model'] = args.model
    if args.dataset:
        config_dict['Main_config']['dataset'] = args.dataset
    if args.trainer:
        config_dict['Main_config']['trainer'] = args.trainer
    if args.lr:
        config_dict['train']['parameter']['base_lr'] = args.lr
    if args.mode:
        config_dict['mode'] = args.mode
    name = config_dict['Main_config']['trainer'].split("Trainer", 1)[0]
    config_dict['Train']['dataset'] = config_dict['Main_config']['dataset']
    config_dict['Test']['dataset'] = config_dict['Main_config']['dataset']
    if name == "unimodal":
        name = 'baseline'
    if args.mu:
        trainer_settings['trainer_para'][name]['mu'] = args.mu
    if args.alpha: 
        trainer_settings['trainer_para'][name]['alpha'] = args.alpha
    if args.eta: 
        trainer_settings['trainer_para'][name]['eta'] = args.eta
    if args.scaling: 
        trainer_settings['trainer_para'][name]['scaling'] = args.scaling
    if args.lam: 
        trainer_settings['trainer_para'][name]['lam'] = args.lam
    if args.momentum: 
        trainer_settings['trainer_para'][name]['momentum_coef'] = args.momentum
    if args.super_epoch: 
        trainer_settings['trainer_para'][name]['super_epoch'] = args.super_epoch
    if args.sigma:
        trainer_settings['trainer_para'][name]['sigma'] = args.sigma
    if args.U:
        trainer_settings['trainer_para'][name]['U'] = args.U
    if args.eps:
        trainer_settings['trainer_para'][name]['eps'] = args.eps
    if args.T_epochs:
        trainer_settings['trainer_para'][name]['T_epochs'] = args.T_epochs
    if args.weight1:
        trainer_settings['trainer_para'][name]['weight1'] = args.weight1
    if args.weight2:
        trainer_settings['trainer_para'][name]['weight2'] = args.weight2
    if args.p_exe:
        trainer_settings['trainer_para'][name]['p_exe'] = args.p_exe
    if args.q_base:
        trainer_settings['trainer_para'][name]['q_base'] = args.q_base
    if args.part_ratio:
        trainer_settings['trainer_para'][name]['part_ratio'] = args.part_ratio
    if args.move_lambda:
        trainer_settings['trainer_para'][name]['move_lambda'] = args.move_lambda
    if args.reinit_epoch:
        trainer_settings['trainer_para'][name]['reinit_epoch'] = args.reinit_epoch
    if args.lr_alpha:
        trainer_settings['trainer_para'][name]['lr_alpha'] = args.lr_alpha
    try:
        #waiting for support iteration
        config_dict['dataset'] = dataset_settings['dataset'][config_dict['Main_config']['dataset']]
        config_dict['dataset']['name'] = config_dict['Main_config']['dataset']
    except:
        raise ValueError("Wrong Dataset name")
    try:
        Trainer_name = config_dict['Main_config']['trainer']
        name = Trainer_name.split("Trainer", 1)[0]
        if name == 'unimodal':
            config_dict['trainer'] = trainer_settings['trainer_para']['baseline'] 
        else:
            config_dict['trainer'] = trainer_settings['trainer_para'][name] 
        config_dict['trainer']['base_para'] = trainer_settings['trainer_para']['base']
        config_dict['trainer']['name'] = name
        
    except:
        raise ValueError("Wrong Trainer setting")
    ## model settings
    try:
        config_dict['model'] = model_settings['model'][config_dict['Main_config']["model"]]
        config_dict['model']['type'] = config_dict['Main_config']['model']
        config_dict['model']['device'] = config_dict['Main_config']['device']
        config_dict['model']['n_classes'] = config_dict['dataset']['classes']
        config_dict['fusion'] = model_settings['fusion'][config_dict['model']['fusion']]
        modalitys = config_dict['model']['encoders'].keys()
        ##new
        config_dict['model']['modality_size'] = {}
        for modality in modalitys:
            name = config_dict['model']['encoders'][modality] 
            config_dict['model']['encoders'][modality] = encoder_settings[name][modality]
            if config_dict['dataset'][modality]['input_dim'] is not None:
                config_dict['model']['encoders'][modality]['input_dim'] = config_dict['dataset'][modality]['input_dim']
            config_dict['model']['encoders'][modality]['name'] = name
            config_dict['model']['encoders'][modality]['if_pretrain'] = encoder_settings[name]['if_pretrain']
            config_dict['model']['encoders'][modality]['pretrain_path'] = encoder_settings[name]['pretrain_path']

            ###new
            config_dict['model']['modality_size'][modality] = encoder_settings[name]['output_dim']
    except:
        raise ValueError("Wrong model setting")
    
    mode = config_dict.get("mode")
    if config_dict['Train'].get('dataset') is None:
        raise ValueError("Dataset not specified.")
    if config_dict.get("Test") is None:
        raise ValueError("Test set not specified.")

    if mode == "train_and_test" or mode == "linear_probe":
        if config_dict.get("Train") is None:
            raise ValueError("Train set not specified.")
        if config_dict.get("Val") is None:
            config_dict['Val'] = config_dict['Test'].copy()
            print ("No validation set specified. Using test set as validation set.")
        exp_name = f"train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if config_dict['train']['checkpoint']['resume'] != '':
            config_dict["out_dir"] = exp_name['train']['checkpoint']['resume']
        else:
            config_dict['name'] = config_dict['Main_config']['model'] + '_' +config_dict['Main_config']['trainer'] \
                + '_' +config_dict['Train']['dataset']
            config_dict["out_dir"] = osp.join(root_path, 'experiments', config_dict["name"], mode, exp_name)
    elif mode == "test" or mode == 'tsne':
        config_dict['Val'] = config_dict['Test'].copy()
        if args.test_path:
            config_dict['test_path'] = args.test_path
        else:
            config_dict['test_path'] = None
        exp_name = f"test_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        config_dict['name'] = config_dict['Main_config']['model'] + '_' +config_dict['Main_config']['trainer'] \
                + '_' +config_dict['Train']['dataset']
        config_dict["out_dir"] = osp.join(root_path, 'experiments', config_dict["name"], mode, exp_name)
        config_dict['trainer']['base_para']['should_train'] = False
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return config_dict


if __name__ == "__main__":
    # Create args. Priority: cli > user > global_default. High priority args overwrite low priority args.
    # specify user config path in cli, if not specified, use user_default BalanceMM/configs/default.toml.
    # global_default indicates global train settings. Read BalanceMM/balancemm/config.toml.
    # --------------------------------------------

    # get args from cli
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default= None)
    parser.add_argument('--trainer', type=str, default= None)
    parser.add_argument('--mode', type= str, default= None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--lr', type=float, default= None)
    parser.add_argument('--alpha', type= float, default= None)
    parser.add_argument('--eta', type= float, default= None)
    parser.add_argument('--mu', type= float, default= None)
    parser.add_argument('--lam', type= float, default= None)
    parser.add_argument('--scaling', type= float, default= None)
    parser.add_argument('--momentum', type= float, default= None)
    parser.add_argument('--super_epoch', type= int, default= None)
    parser.add_argument('--eps', type= float, default= None)
    parser.add_argument('--sigma', type= float, default= None)
    parser.add_argument('--U', type= float, default= None)
    parser.add_argument('--window_size', type= float, default= None)
    parser.add_argument('--T_epochs', type= float, default= None)
    parser.add_argument('--weight1', type= float, default= None)
    parser.add_argument('--weight2', type= float, default= None)
    parser.add_argument('--p_exe', type= int, default= None)
    parser.add_argument('--q_base', type= float, default= None)
    parser.add_argument('--part_ratio', type= float, default= None)
    parser.add_argument('--move_lambda', type= str, default= None)
    parser.add_argument('--reinit_epoch', type= str, default= None)
    parser.add_argument('--lr_alpha', type= str, default= None)
    parser.add_argument('--test_path', type= str, default= None)
    args = sys.argv[1:]
    print(args)
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    # get yaml config path and load, if not specified, use user_default
    default_config_path = osp.join(root_path, "configs", "user_default.yaml")
    config_path = ensure_and_get_config_path(args, default_config_path)
    custom_args = load_config_dict(config_path)
    with open(osp.join(root_path ,"configs", "global_config.yaml"), 'r') as f:
        global_settings = yaml.safe_load(f)
    custom_args = global_settings | custom_args
    # merge cli_args into yaml and create config.
    # temp_args = {}
    # parse_cli_args_to_dict(args, custom_args)
    # parse_cli_args_to_dict(args, temp_args)
    # print(custom_args)
    
    # merge custom_args into default global config(balancemm/config.toml)
    args = create_config(custom_args, parser.parse_args())
    set_seed(args['seed'])
    # print args in yaml format
    print ("BalanceMM")
    print('------Load Config-----')
    print (yaml.dump(args, sort_keys=False), end='')
    print('----------------------')


    if args['mode'] == "train_and_test":
        train_and_test(args)
    elif args['mode'] == "test":
        test(args)
    elif args['mode'] == "tsne":
        t_sne(args)
    elif args['mode'] == "linear_probe":
        linear_probe_eval(args)
