import os,sys
import importlib
from src.utilities.load_arguments import load_arguments


if __name__ == '__main__':
    arguments = load_arguments('file_config.yaml')
    running_task = arguments['general']['running_task']

    # getattr(importlib.import_module('src.tasks.' + running_task + '.preprocess'), 'preprocess')(arguments)
    getattr(importlib.import_module('src.tasks.' + running_task + '.train_model'), 'train_model')(arguments)
    # getattr(importlib.import_module('src.tasks.' + running_task + '.test_model'), 'test_model')(arguments)
    # getattr(importlib.import_module('src.tasks.' + running_task + '.apply'), 'apply_model')(arguments)

