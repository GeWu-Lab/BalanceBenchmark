import yaml
from dynaconf import Dynaconf

__all__ = ['parse_cli_args_to_dict', 'load_config_dict_from_yaml', 'ensure_and_get_config_path', 'find_module']

def _merge_into_dict(target_dict: dict, key_path: str, value: any) -> None:
    """
    Merge a hierarchically structured key and its value into the target dictionary.

    :param target_dict: The dictionary to be updated.
    :param key_path: A string representing the hierarchical keys, separated by dots.
    :param value: The value to be set at the innermost key.
    """
    keys = key_path.split('.')
    for key in keys[:-1]:
        if key not in target_dict or not isinstance(target_dict[key], dict):
            target_dict[key] = {}
        target_dict = target_dict[key]
    target_dict[keys[-1]] = value

def parse_cli_args_to_dict(cli_args: list[str], args_dict: dict = {}) -> dict:
    """
    Parses a list of command-line arguments into a dictionary, supporting keys prefixed with '-' or '--'.
    Values can be specified either by '=' or as the next argument. Keys without explicit values are assigned
    an empty string. When the same key is specified multiple times, the last value takes precedence.
    Notice that format '-a.b' or '-a .b' is not supported, use "--a.b" or "-a '.b'" instead.

    Args:
        cli_args (list[str]): The list of command-line arguments.
        args_dict (dict, optional): An existing dictionary to update with the command-line arguments.
                                    Defaults to None, in which case a new dictionary is created.

    Returns:
        dict: A dictionary containing the parsed command-line arguments, merged with any existing
              entries in `args_dict`.
    """
    i = 0
    while i < len(cli_args):
        if not cli_args[i].startswith('-'):
            i += 1
            continue
        if not cli_args[i].startswith('--'):
            # check for invalid argument format '-a.b' or '-a .b'
            if i + 1 < len(cli_args) and cli_args[i + 1].startswith('.'):
                raise ValueError(f"Invalid argument: '-{cli_args[i]}{cli_args[i + 1]}'. Use --a.b or -a '.b' instead.")
        arg = cli_args[i].lstrip('-')
        value = ''  # Default value if no explicit value is provided
        # Check for '=' in the current argument
        if '=' in arg:
            arg, value = arg.split('=', 1)
        else:
            # Check if the next argument is a value (not another key)
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith('-'):
                value = cli_args[i + 1]
                i += 1  # Skip the next argument since it's a value
        # Use '.' to replace '-' in key paths for hierarchical structuring
        _merge_into_dict(args_dict, arg, value)
        i += 1

    return args_dict

def load_config_dict(config_path: str) -> dict:
    """
    Load configuration using dynaconf, suppor toml, yaml.

    :param config_path: The file path to the YAML configuration file.
    :return: A dictionary containing the loaded configuration, or None if an error occurs.
    """
    try:
        with open(config_path, "r") as f:
            args = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading file {config_path}: \n{e}")
    print(f"Loaded config from {config_path}")
    return args
    
def ensure_and_get_config_path(args: list[str], default_config_path: str) -> str:
    """
    Ensure that the --config argument is present in args, supporting both '--config=<path>' and '--config <path>' formats,
    as well as the '-c' abbreviation. If not present, extend args with the default config path.
    
    :param args: List of command-line arguments.
    :param default_config_path: The default path to the configuration file if --config or -c is not specified.
    :return: config_path (str): The path to the configuration file.
    """
    config_key_long = '--config'
    config_key_short = '-c'
    config_path = None

    for i, arg in enumerate(args):
        if arg.startswith(config_key_long + '=') or arg.startswith(config_key_short + '='):
            config_path = arg.split('=', 1)[1]
            break
        elif arg == config_key_long or arg == config_key_short:
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                config_path = args[i + 1]
                break

    if config_path is None:
        config_path = default_config_path
        args.extend([f'{config_key_long}={config_path}'])

    return config_path

def find_module(module_list: list, module_name: str, module_type: str = 'Module') -> any:
    module_name = module_name + module_type
    for module in module_list:
        module_cls = getattr(module, module_name, None)
        if module_cls is not None:
            return module_cls
    raise ValueError(f'{module_type} {module_name} is not found.')