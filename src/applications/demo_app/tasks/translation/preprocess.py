import os, sys, json, numpy, torch
from tqdm import tqdm
# import spacy
# spacy_tokenizer = spacy.load("en_core_web_sm")
from sklearn.model_selection import train_test_split
from src.utilities.clean_data import *
from src.utilities.load_data import *
from src.utilities.load_config import load_config
from src.modules.tokenizers.tokenizer import *


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
    used_model = config['tasks']['model']
    dataset_root = config['dataset_root']
    train_json_file = config['net_structure']['dataset']['train_json_file']
    valid_json_file = config['net_structure']['dataset']['valid_json_file']
    craw_corpus_en_file = config['net_structure']['dataset']['corpus_en_file']
    craw_corpus_zh_file = config['net_structure']['dataset']['corpus_zh_file']
    file_train_en_file = config['net_structure']['dataset']['train_en_file']
    file_train_zh_file = config['net_structure']['dataset']['train_zh_file']
    file_test_en_file = config['net_structure']['dataset']['test_en_file']
    file_test_zh_file = config['net_structure']['dataset']['test_zh_file']
    test_size = config['net_structure']['dataset']['test_size']
    max_len = config['model'][used_model]['max_len'] - 4   # condidering sos, eos, pad, unk
    random_state = config['general']['random_state']
    module_obj = sys.modules['src.utilities.load_data']

    # step 1: read the raw data
    data_text_en = []
    data_text_zh = []
    train_json_file = dataset_root + os.path.sep + train_json_file
    valid_json_file = dataset_root + os.path.sep + valid_json_file
    with open(train_json_file, 'r', encoding='utf-8') as f:
        json_txt = f.readlines()
    for txt in json_txt:
        t = json.loads(txt)
        data_text_en.append(t['english'] + '\n')
        data_text_zh.append(t['chinese'] + '\n')

    with open(valid_json_file, 'r', encoding='utf-8') as f:
        json_txt = f.readlines()
    for txt in json_txt:
        t = json.loads(txt)
        data_text_en.append(t['english'] + '\n')
        data_text_zh.append(t['chinese'] + '\n')

    # #分析字长。
    # s_len_en = []
    # s_len_zh = []
    # for i, text in tqdm(enumerate(data_text_en)):
    #     s_len_en.append(count_token(text))
    #     s_len_zh.append(count_token(data_text_zh[i]))
    # import matplotlib.pyplot as plt
    # # plt.plot(numpy.arange(len(s_len_en)), [[s_len_en[i], s_len_zh[i]] for i in range(len(s_len_en))])
    # fig, axes = plt.subplots(1,2)
    # ax0 = axes[0]
    # ax1 = axes[1]
    # ax0.plot(numpy.arange(len(s_len_en)), s_len_en)
    # ax0.legend(['en'])
    # ax1.plot(numpy.arange(len(s_len_zh)), s_len_zh)
    # ax1.legend(['zh'])
    # plt.show()
    # return None

    # data_text_en = get_txt_from_file(project_root + os.path.sep + craw_corpus_en)
    # data_text_zh = get_txt_from_file(project_root + os.path.sep + craw_corpus_zh)
    #

    # 按照实际业务处理的分词配置筛选预料，过滤过短和过长的句子。
    fun_name_en = config['net_structure']['word_vector']['tokenizer_en_file']
    fun_name_zh = config['net_structure']['word_vector']['tokenizer_zh_file']
    en_filter = getattr(module_obj, fun_name_en)
    zh_filter = getattr(module_obj, fun_name_zh)
    data_set = []
    for i,text in tqdm(enumerate(data_text_en)):
        en_len = len(en_filter(text))
        zh_len = len(zh_filter(data_text_zh[i]))
        if (en_len > 4) and (en_len < max_len) and (zh_len < max_len):
            data_set.append((data_text_zh[i], text))
    data_set = list(zip(data_text_zh, data_text_en))

    print("spliting...")
    train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=True, random_state=random_state)
    train_zh, train_en = list(zip(*train_set))
    test_zh, test_en = list(zip(*test_set))

    print("saving...")
    put_txt_to_file(train_zh, dataset_root + os.path.sep + file_train_zh_file)
    put_txt_to_file(train_en, dataset_root + os.path.sep + file_train_en_file)
    put_txt_to_file(test_zh, dataset_root + os.path.sep + file_test_zh_file)
    put_txt_to_file(test_en, dataset_root + os.path.sep + file_test_en_file)
    print("preprocess over.")
    return "over"

if __name__ == '__main__':
    config = load_config('file_config.yaml')
    preprocess(config)