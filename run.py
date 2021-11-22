from src.utilities.load_arguments import load_arguments
from src.tasks.translation.train_model import train_model
from src.tasks.translation.test_model import test_model
from src.tasks.translation.preprocess import preprocess
from src.tasks.translation.apply import apply_model
import os,sys


if __name__ == '__main__':
    arguments = load_arguments('file_config.yaml')
    # preprocess(arguments)
    train_model(arguments)
    # test_model(arguments)
    # apply_model(arguments)
