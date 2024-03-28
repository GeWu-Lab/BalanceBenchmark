import sys, yaml

from os import path as osp
from balancemm.utils.parser_utils import parse_cli_args_to_dict, load_config_dict, ensure_and_get_config_path
from balancemm.train import train_and_test
from balancemm.test import only_test
from lightning import fabric
import datetime

root_path = osp.dirname(osp.dirname(__file__))

def create_config(config_dict: dict): 
    """Create configuration from cli and yaml."""
    with open(osp.join(root_path ,"configs", "global_config.yaml"), 'r') as f:
        global_settings = yaml.safe_load(f)
    config_dict = global_settings | config_dict

    mode = config_dict.get("mode")
    if config_dict.get('dataset') is None:
        raise ValueError("Dataset not specified.")
    if config_dict['dataset'].get("test") is None:
        raise ValueError("Test set not specified.")

    if mode == "train_and_test":
        if config_dict['dataset'].get("train") is None:
            raise ValueError("Train set not specified.")
        if config_dict['dataset'].get("val") is None:
            config_dict['dataset']['val'] = config_dict['dataset']['test'].copy()
            print ("No validation set specified. Using test set as validation set.")
        exp_name = f"train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if config_dict['train']['checkpoint']['resume'] != '':
            config_dict["out_dir"] = exp_name['train']['checkpoint']['resume']
        else:
            config_dict["out_dir"] = osp.join(root_path, 'experiments', config_dict["name"], exp_name)
    elif mode == "test":
        config_dict.pop('train', None)
        config_dict.pop('trainer', None)
        config_dict['dataset'].pop('train', None)
        config_dict['dataset'].pop('val', None)
        exp_name = f"test_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        config_dict["out_dir"] = osp.join(root_path, 'experiments', config_dict["name"], exp_name)
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
    
    # merge cli_args into yaml and create config.
    parse_cli_args_to_dict(args, custom_args)
    
    # merge custom_args into default global config(balancemm/config.toml)
    args = create_config(custom_args)
    
    # print args in yaml format
    print ("BalanceMM")
    print('------Load Config-----')
    print (yaml.dump(args, sort_keys=False), end='')
    print('----------------------')

    if args['mode'] == "train_and_test":
        train_and_test(args)
    elif args['mode'] == "test":
        only_test(args)