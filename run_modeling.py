import os,sys
import importlib
from src.utilities.load_config import load_config


if __name__ == '__main__':
    config = load_config('config_project.yaml', 'config_deploy.yaml')
    running_app = config['running_app']
    running_task = config['running_task']
    task_module_root = 'src.applications.' + running_app + '.tasks.' + running_task + '.'
    running_step = config['running_step']
    getattr(importlib.import_module(task_module_root + running_step), running_step)(config)


