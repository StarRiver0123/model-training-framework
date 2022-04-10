import os, sys
from tqdm import tqdm
import json
import pickle as pk
import numpy, copy
import pandas as pd
from src.utilities.clean_data import *
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.load_config import load_config


def task_specified_processing(data_set, p_mask=0.1, p_pos=0.1, p_ng=0.25, min_size=4, gram_size=8, n_iter=10):
    # 数据增强
    # 可以使用多进程，加快数据处理速度
    import jieba
    import jieba.posseg as pseg
    from numpy.random import randint, rand
    from random import choice
    # 统计词性标签
    print("数据增强处理，统计词性标签...")
    pos_dict = {}
    cut_words_list = []
    for t in tqdm(data_set):
        text = t[0].replace(' ', '')
        words = pseg.cut(text)
        word_flag = []
        for word, flag in words:
            if flag not in pos_dict.keys():
                pos_dict[flag] = set()
            pos_dict[flag].add(word)
            word_flag.append((word, flag))
        cut_words_list.append(word_flag)
    # 开始生成增强样本
    print("数据增强处理，开始生成增强样本...")
    augment_set = []
    for s_id, example in tqdm(enumerate(cut_words_list)):  # 每一条样本
        for _ in range(n_iter):                 # 增强采集n_iter次
            augment = []                        # 生成替换后的列表
            p1_replaced = False
            for word, flag in example:
                p1 = rand()
                if p1 < p_mask:
                    augment.append('[mask]')    # masking
                    p1_replaced = True
                elif p1 < (p_mask + p_pos):      # pos guided word replacement
                    augment.append(choice(list(pos_dict[flag])))    # 这里有一个bug：随机选出来的有可能跟原来的一样。
                    p1_replaced = True
                else:
                    augment.append(word)
            p2 = rand()
            if p2 < p_ng:                        # n-gram sampling  n-gram截取
                list_len = len(augment)
                n_size = min(list_len, randint(min_size, gram_size + 1))
                max_start_id = list_len - n_size
                start_id = randint(0, max_start_id + 1)
                end_id = start_id + n_size
            elif p1_replaced:                    # 不进行n-gram截取，但如果曾经进行过替换，则对整句进行分字处理
                start_id = 0
                end_id = len(augment)
            if p1_replaced or p2 < p_ng:
                augment_cut = []
                for idx in range(start_id, end_id):
                    if augment[idx] == '[mask]' or example[idx][1] == 'eng':
                        augment_cut.append(augment[idx])
                    else:
                        augment_cut += list(augment[idx])
                augment_set.append([' '.join(augment_cut), data_set[s_id][1]])
    return augment_set


clean_config = {
    'replace_specials': False,
    'specials_token_replacement': [('\n', '')],
    'replace_english': False,
    'english_token_replacement': 'eng',
    'replace_digits': False,
    'digits_token_replacement': 'num',
    'remove_english_punctuation': False,
    'remove_chinese_punctuation': False,
    'remove_non_hanzi_english_digits': False,
    'lower_case': False
}


def preprocess(config):
    dataset_root = config['dataset_root']
    corpus_file = dataset_root + os.path.sep + config['net_structure']['dataset']['corpus_file']
    train_set_file = dataset_root + os.path.sep + config['net_structure']['dataset']['train_set_file']
    valid_set_file = dataset_root + os.path.sep + config['net_structure']['dataset']['valid_set_file']
    test_set_file = dataset_root + os.path.sep + config['net_structure']['dataset']['test_set_file']
    valid_size = config['net_structure']['dataset']['valid_size']
    test_size = config['net_structure']['dataset']['test_size']
    random_state = config['general']['random_state']

    # step 1: read the raw data
    corpus = load_data_from_file(corpus_file, encoding='utf-8')
    corpus = corpus[:, [1, 0]]

    # step 2: clean the data
    data_cleaner = DataCleaner(clean_config)
    text_list = data_cleaner.clean(corpus[:, 0])
    data_set = list(zip(text_list, corpus[:, 1]))

    # step 3: split the data
    train_set, test_set = train_test_split(corpus, test_size=test_size, shuffle=True, random_state=random_state)
    train_set, valid_set = train_test_split(train_set, test_size=valid_size, shuffle=True, random_state=random_state)

    # step 4: some task-specified processing
    use_data_augmentation = config['net_structure']['dataset']['use_data_augmentation']
    n_iter = 1
    if use_data_augmentation:
        train_augment_set_file = dataset_root + os.path.sep + config['net_structure']['dataset'][
            'train_augment_set_file']
        train_augment_set = task_specified_processing(train_set, n_iter=n_iter)

    # step 5: save the data
    train_dataframe = pd.DataFrame(train_set)
    valid_dataframe = pd.DataFrame(valid_set)
    test_dataframe = pd.DataFrame(test_set)
    train_dataframe.to_csv(train_set_file, index=False)
    valid_dataframe.to_csv(valid_set_file, index=False)
    test_dataframe.to_csv(test_set_file, index=False)
    if use_data_augmentation:
        train_augment_dataframe = pd.DataFrame(train_augment_set)
        train_augment_dataframe.to_csv(train_augment_set_file, index=False)

    print('ok')


if __name__ == '__main__':
    config = load_config('config_project.yaml')
    preprocess(config)