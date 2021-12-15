import os, sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.load_arguments import load_arguments


def preprocess(arguments):
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    corpus_tagging = arguments['tasks'][running_task]['dataset']['corpus_tagging']
    train_tagging = arguments['tasks'][running_task]['dataset']['train_tagging']
    test_tagging = arguments['tasks'][running_task]['dataset']['test_tagging']
    test_size = arguments['training'][running_task]['test_size']
    data_set = get_txt_from_file(project_root + os.path.sep + corpus_tagging)
    print("spliting...")
    train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=False)
    print("saving...")
    put_txt_to_file(train_set, project_root + os.path.sep + train_tagging)
    put_txt_to_file(test_set, project_root + os.path.sep + test_tagging)
    print("preprocess over.")

if __name__ == '__main__':
    arguments = load_arguments('file_config.yaml')
    preprocess(arguments)