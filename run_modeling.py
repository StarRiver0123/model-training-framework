import os,sys
import importlib
from src.utilities.load_arguments import load_arguments


if __name__ == '__main__':
    arguments = load_arguments('config_project.yaml', 'config_deploy.yaml')
    running_app = arguments['running_app']
    running_task = arguments['running_task']
    task_module_root = 'src.applications.' + running_app + '.tasks.' + running_task + '.'
    for step in arguments['running_pipeline']:
        getattr(importlib.import_module(task_module_root + step), step)(arguments)


