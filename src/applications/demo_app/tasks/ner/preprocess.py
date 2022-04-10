import os, sys
from tqdm import tqdm
import json
import pickle as pk
import numpy, copy
from src.utilities.clean_data import *
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.load_config import load_config


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

def load_teachers_dataset(config, sos_token=None, eos_token=None):
    dataset_root = config['dataset_root']
    data_source = dataset_root + os.path.sep + config['net_structure']['dataset']['ner_parameter_file']
    parameter = pk.load(open(data_source, 'rb'))
    if (sos_token is not None) and (eos_token is not None):
        train_set = [([sos_token] + pair[0] + [eos_token], pair[1]) for pair in parameter['data_set']['train']]
        valid_set = [([sos_token] + pair[0] + [eos_token], pair[1]) for pair in parameter['data_set']['dev']]
    else:
        train_set = parameter['data_set']['train']
        valid_set = parameter['data_set']['dev']
    stoi = parameter['key2ind']
    itos = list(stoi.keys())
    return train_set, valid_set, stoi, itos


def random_sample(data_set, max_len, sos_token=None, eos_token=None, strip_more_O = False):
    # data_set input format: [('机','B-基本概念'),('器','I-基本概念')]
    # data_set output format: (['机','器'],['B-基本概念','I-基本概念'])
    assert (sos_token is None and eos_token is None) or (sos_token is not None and eos_token is not None)
    data_set_len = len(data_set)
    if eos_token is not None:
        max_len -= 2     # sos_token 和eos_token占去了2位
    up_range = data_set_len - max_len
    while True:
        id0 = random.randint(0, up_range)
        id1 = id0 + max_len - 1
        while (id1 >= id0) and (data_set[id1][1][0] not in ['E', 'S', 'O']):
            id1 -= 1
        while (id1 >= id0) and (data_set[id0][1][0] not in ['B', 'S', 'O']):
            id0 += 1
        if (id1 >= id0):
            if strip_more_O:
                len_threshold = random.randint(max_len // 8, max_len // 2)
                could_be = copy.deepcopy(data_set[id0:id1 + 1])
                if (numpy.sum([i[1][0] == 'O' for i in could_be]) >= len(data_set[id0:id1 + 1])):
                    continue
                seq_len = len(could_be)
                o_r = 0
                while o_r < seq_len:            # 减少'O'的数量，连续'O'的数量不能超过阈值，降低标签不平衡的影响。
                    o_l = o_r
                    while (o_l < seq_len) and (could_be[o_l][1][0] != 'O'):
                        o_l += 1
                    if o_l >= seq_len:               #列表右端已经全部是’O‘了
                        break
                    o_r = o_l + 1
                    while (o_r < seq_len) and (could_be[o_r][1][0] == 'O'):
                        o_r += 1
                    if (o_r - o_l) > len_threshold:
                        del could_be[o_l : o_r - len_threshold]
                        seq_len -= o_r - o_l -len_threshold
                        o_r -= o_r - o_l -len_threshold
                if (o_r - o_l) >= len(data_set[id0:id1 + 1]):    # 说明本次得到的全部是’O‘则丢弃不用，重新检索
                    continue
            else:
                could_be = data_set[id0:id1 + 1]
            zipped_list = list(zip(*could_be))
            if sos_token is None:
                return (list(zipped_list[0]), list(zipped_list[1]))
            else:
                return ([sos_token] + list(zipped_list[0]) + [eos_token], list(zipped_list[1]))
                        # source需要加上sos和eos符号，但是target不需要加，因为在iterator生成数据时根据field设置会自动加上。
        else:   # 如果找不到，则重新生成随机起始位去寻找
            continue


def clue_data(config):
    dataset_root = config['dataset_root']
    train_set_file = config['net_structure']['dataset']['train_set_file']
    valid_set_file = config['net_structure']['dataset']['valid_set_file']
    test_set_file = config['net_structure']['dataset']['test_set_file']

    src_sos_token = None  # 预处理阶段不添加特殊字符，留在建造数据时根据配置添加
    src_eos_token = None  # 预处理阶段不添加特殊字符，留在建造数据时根据配置添加
    train_set, valid_set, stoi, itos = load_teachers_dataset(config, src_sos_token, src_eos_token)
    valid_set, test_set = train_test_split(valid_set, test_size=0.5, shuffle=False)
    train_list = []
    for train in train_set:
        train_list.append(' '.join(train[0]) + '\n')
        train_list.append(' '.join(train[1]) + '\n')
    valid_list = []
    for valid in valid_set:
        valid_list.append(' '.join(valid[0]) + '\n')
        valid_list.append(' '.join(valid[1]) + '\n')
    test_list = []
    for test in valid_set:
        test_list.append(' '.join(test[0]) + '\n')
        test_list.append(' '.join(test[1]) + '\n')

    put_txt_to_file(train_list, dataset_root + os.path.sep + train_set_file)
    put_txt_to_file(valid_list, dataset_root + os.path.sep + valid_set_file)
    put_txt_to_file(test_list, dataset_root + os.path.sep + test_set_file)


def mine_data(config):
    dataset_root = config['dataset_root']
    corpus_tagging_file = config['net_structure']['dataset']['corpus_tagging_file']
    train_set_file = config['net_structure']['dataset']['train_set_file']
    valid_set_file = config['net_structure']['dataset']['valid_set_file']
    test_set_file = config['net_structure']['dataset']['test_set_file']
    valid_size = config['net_structure']['dataset']['valid_size']
    test_size = config['net_structure']['dataset']['test_size']
    used_model = config['net_structure']['model']
    max_len = config['model'][used_model]['max_len']
    src_sos_token = None  # 预处理阶段不添加特殊字符，留在建造数据时根据配置添加
    src_eos_token = None  # 预处理阶段不添加特殊字符，留在建造数据时根据配置添加

    # step 1: read the raw data
    data_set = load_data_from_file(dataset_root + os.path.sep + corpus_tagging_file, encoding='utf-8')

    # step 2: clean the data
    # data_cleaner = DataCleaner(clean_config)
    # text_list = data_cleaner.clean(data_set[:, 0])
    # data_set = list(zip(text_list, data_set[:, 1]))

    # step 3: split the data
    print("spliting...")
    train_part, test_part = train_test_split(data_set, test_size=test_size, shuffle=False)
    train_part, valid_part = train_test_split(train_part, test_size=valid_size, shuffle=False)
    train_s = []
    for text in train_part:
        train_s.append(tuple(text.strip().split()))
    valid_s = []
    for text in valid_part:
        valid_s.append(tuple(text.strip().split()))
    test_s = []
    for text in test_part:
        test_s.append(tuple(text.strip().split()))

    # step 4: generate the data
    print("generating...")
    gen_num_total_examples = config['net_structure']['dataset']['gen_num_total_examples']
    test_set = []
    for i in range(int(gen_num_total_examples * test_size)):
        test_set.append(random_sample(test_s, max_len, src_sos_token, src_eos_token))
    valid_set = []
    for i in range(int(gen_num_total_examples * (1 - test_size) * valid_size)):
        valid_set.append(random_sample(valid_s, max_len, src_sos_token, src_eos_token))
    train_set = []
    for i in range(int(gen_num_total_examples * (1 - test_size) * (1 - valid_size))):
        train_set.append(random_sample(train_s, max_len, src_sos_token, src_eos_token))

    train_list = []
    for train in train_set:
        train_list.append(' '.join(train[0]) + '\n')
        train_list.append(' '.join(train[1]) + '\n')
    valid_list = []
    for valid in valid_set:
        valid_list.append(' '.join(valid[0]) + '\n')
        valid_list.append(' '.join(valid[1]) + '\n')
    test_list = []
    for test in valid_set:
        test_list.append(' '.join(test[0]) + '\n')
        test_list.append(' '.join(test[1]) + '\n')

    # step 5: save the data
    put_txt_to_file(train_list, dataset_root + os.path.sep + train_set_file)
    put_txt_to_file(valid_list, dataset_root + os.path.sep + valid_set_file)
    put_txt_to_file(test_list, dataset_root + os.path.sep + test_set_file)


def preprocess(config):
    dataset_class = config['net_structure']['dataset']['dataset_class']
    if dataset_class == 'clue':
        clue_data(config)
    if dataset_class == 'mine':
        mine_data(config)
    print("preprocess over.")

if __name__ == '__main__':
    config = load_config('config_project.yaml')
    preprocess(config)