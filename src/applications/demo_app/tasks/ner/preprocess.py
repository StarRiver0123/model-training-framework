import os, sys
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.load_arguments import load_arguments


def preprocess(arguments):
    dataset_root = arguments['dataset_root']
    corpus_tagging_file = arguments['net_structure']['dataset']['corpus_tagging_file']
    train_tagging_file = arguments['net_structure']['dataset']['train_tagging_file']
    test_tagging_file = arguments['net_structure']['dataset']['test_tagging_file']
    test_size = arguments['training']['test_size']

    # data_set = get_txt_from_file(project_root + os.path.sep + corpus_tagging)
    # print("spliting...")
    # train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=False)
    # print("saving...")
    # put_txt_to_file(train_set, project_root + os.path.sep + train_tagging)
    # put_txt_to_file(test_set, project_root + os.path.sep + test_tagging)

    train_file = r'G:\AI\projects\mtf_projects\dataset\for_ner\train.json'
    valid_file = r'G:\AI\projects\mtf_projects\dataset\for_ner\dev.json'
    test_file = r'G:\AI\projects\mtf_projects\dataset\for_ner\test.json'
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            text = json_line['text']
            label = json_line['label']

    data_set = get_txt_from_file(dataset_root + os.path.sep + corpus_tagging_file)
    print("spliting...")
    train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=False)
    print("saving...")
    put_txt_to_file(train_set, dataset_root + os.path.sep + train_tagging_file)
    put_txt_to_file(test_set, dataset_root + os.path.sep + test_tagging_file)
    print("preprocess over.")

if __name__ == '__main__':
    arguments = load_arguments('file_config.yaml')
    preprocess(arguments)