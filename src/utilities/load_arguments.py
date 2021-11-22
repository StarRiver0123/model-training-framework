import os
import torch
import yaml
import src
from src.utilities.create_logger import create_logger

# will be called by run.py in the project root, to load all arguments.
def load_arguments(root_config_file):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(src.__file__)))
    file_config = load_file_config(project_root, root_config_file)
    args = load_args_from_yaml(project_root, file_config['file']['src']['arguments'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arguments = {}
    arguments.update(file_config)
    arguments.update(args)
    arguments['general'].update({'device': device, 'project_root': project_root})
    logger = create_logger(arguments)
    arguments['general'].update({'logger': logger})
    return arguments

def load_file_config(project_root, config_file):
    args = {}
    with open(project_root + os.path.sep + config_file, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f.read())
    return args

def load_args_from_yaml(project_root, arg_file_root):
    arguments_root = project_root + os.path.sep + arg_file_root
    arguments_paths = []
    arguments_paths.append(arguments_root)
    arguments = {}
    while len(arguments_paths):
        f_p = arguments_paths.pop()
        if os.path.isfile(f_p):
            with open(f_p, 'r', encoding='utf-8') as f:
                args = yaml.safe_load(f.read())
            arguments.update(args)
        if os.path.isdir(f_p):
            for sub_f_p in os.listdir(f_p):
                arguments_paths.append(f_p + os.path.sep + sub_f_p)
    return arguments
