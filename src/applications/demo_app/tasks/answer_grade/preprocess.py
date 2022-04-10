import os, sys, json, numpy, torch, pandas
from tqdm import tqdm
# import spacy
# spacy_tokenizer = spacy.load("en_core_web_sm")
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.clean_data import *
from src.utilities.load_config import load_config


def task_specified_processing(text_list_q, text_list_a, gen_num_total_examples):
    # 小样本增强：原始数据中一个问题对应多个答案，先把多个答案组成同一个问题的答案列表，整体数据集生成一个问题词典，然后随机选择一个问题，
    # 再从对应的答案列表中随机选择几个进行合并，合并后的答案和问题构成要给问答组。
    data_set = []
    qa_list = defaultdict(list)
    for i in range(len(text_list_q)):
        qa_list[text_list_q[i]] += [text_list_a[i]]
    len_qa = len(qa_list)
    list_q = list(qa_list.keys())
    for i in range(gen_num_total_examples):
        idx = random.randint(1,len_qa - 1)
        q = list_q[idx]
        num_a = random.randint(1,len(qa_list[q]))
        list_a = random.sample(qa_list[q], num_a)
        a = ' '.join(list_a)
        data_set.append((q, a))
    return data_set


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
    craw_corpus_qa_file = config['net_structure']['dataset']['corpus_qa_file']
    file_train_q_file = config['net_structure']['dataset']['train_q_file']
    file_train_a_file = config['net_structure']['dataset']['train_a_file']
    file_test_q_file = config['net_structure']['dataset']['test_q_file']
    file_test_a_file = config['net_structure']['dataset']['test_a_file']
    test_size = config['net_structure']['dataset']['test_size']
    gen_num_total_examples = config['net_structure']['dataset']['gen_num_total_examples']
    random_state = config['general']['random_state']

    # step 1: read the raw data
    craw_data = load_data_from_file(dataset_root + os.path.sep + craw_corpus_qa_file, encoding='utf-8')
    craw_data = craw_data[craw_data[:, 2] == 'question2answer']

    # step 2: clean the data
    data_cleaner = DataCleaner(clean_config)
    text_list_q = data_cleaner.clean(craw_data[:, 0])
    text_list_a = data_cleaner.clean(craw_data[:, 1])

    # step 3: some task-specified processing
    data_set = task_specified_processing(text_list_q, text_list_a, gen_num_total_examples)

    # step 4: split the data into train, valid and test set
    print("spliting...")
    train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=True, random_state=random_state)
    train_q, train_a = list(zip(*train_set))
    test_q, test_a = list(zip(*test_set))

    # step 5: save the data
    print("saving...")
    put_txt_to_file(train_q, dataset_root + os.path.sep + file_train_q_file)
    put_txt_to_file(train_a, dataset_root + os.path.sep + file_train_a_file)
    put_txt_to_file(test_q, dataset_root + os.path.sep + file_test_q_file)
    put_txt_to_file(test_a, dataset_root + os.path.sep + file_test_a_file)
    print("preprocess over.")

if __name__ == '__main__':
    config = load_config('file_config.yaml')
    preprocess(config)