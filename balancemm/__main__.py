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
from balancemm.train import train_and_test
from balancemm.test import only_test
from lightning import fabric
import datetime
from .utils.train_utils import set_seed
root_path = osp.dirname(osp.dirname(__file__))

def create_config(config_dict: dict): 
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
    print('==================')
    print(config_dict)
    # print('==================')
    # print(global_settings)
    try:
        #waiting for support iteration
        config_dict['dataset'] = dataset_settings['dataset'][config_dict['Main_config']['dataset']]
        config_dict['dataset']['name'] = config_dict['Main_config']['dataset']
    except:
        raise ValueError("Wrong Dataset name")
    try:
        Trainer_name = config_dict['Main_config']['trainer']
        name = Trainer_name.split("Trainer", 1)[0]
        config_dict['trainer'] = trainer_settings['trainer_para'][name] 
        config_dict['trainer']['base_para'] = trainer_settings['trainer_para']['base']
        
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

    if mode == "train_and_test":
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
    elif mode == "test":
        # config_dict.pop('train', None)
        # config_dict.pop('trainer', None)
        # config_dict.pop('Train', None)
        # config_dict.pop('Val', None)
        config_dict['Val'] = config_dict['Test'].copy()
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
    args = sys.argv[1:]

    # get yaml config path and load, if not specified, use user_default
    default_config_path = osp.join(root_path, "configs", "user_default.yaml")
    config_path = ensure_and_get_config_path(args, default_config_path)
    custom_args = load_config_dict(config_path)
    with open(osp.join(root_path ,"configs", "global_config.yaml"), 'r') as f:
        global_settings = yaml.safe_load(f)
    custom_args = global_settings | custom_args
    # merge cli_args into yaml and create config.
    parse_cli_args_to_dict(args, custom_args)
    print(custom_args)
    
    # merge custom_args into default global config(balancemm/config.toml)
    args = create_config(custom_args)
    set_seed(args['seed'])
    # print args in yaml format
    print ("BalanceMM")
    print('------Load Config-----')
    print (yaml.dump(args, sort_keys=False), end='')
    print('----------------------')


    if args['mode'] == "train_and_test":
        train_and_test(args)
    elif args['mode'] == "test":
        only_test(args)