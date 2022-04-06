import os, sys, json, numpy, torch, pandas
from tqdm import tqdm
# import spacy
# spacy_tokenizer = spacy.load("en_core_web_sm")
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.load_config import load_config


def preprocess(config):
    used_model = config['net_structure']['model']
    dataset_root = config['dataset_root']
    craw_corpus_qa_file = config['net_structure']['dataset']['corpus_qa_file']
    file_train_q_file = config['net_structure']['dataset']['train_q_file']
    file_train_a_file = config['net_structure']['dataset']['train_a_file']
    file_test_q_file = config['net_structure']['dataset']['test_q_file']
    file_test_a_file = config['net_structure']['dataset']['test_a_file']
    test_size = config['net_structure']['dataset']['test_size']
    gen_num_total_examples = config['net_structure']['dataset']['gen_num_total_examples']
    # max_len = config['model'][used_model]['max_len'] - 4   # condidering sos, eos, pad, unk
    random_state = config['general']['random_state']
    # module_obj = sys.modules['src.utilities.load_data']
    # use_bert = config['tasks'][running_task]['use_bert']
    # device = config['general']['device']

    craw_data = pandas.read_csv(dataset_root + os.path.sep + craw_corpus_qa_file, encoding='utf-8')
    craw_data = craw_data[craw_data['关系'] == 'question2answer']
    text_list_q = list(craw_data.实体1)
    text_list_a = list(craw_data.实体2)
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

    print("spliting...")
    train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=True, random_state=random_state)
    train_q, train_a = list(zip(*train_set))
    test_q, test_a = list(zip(*test_set))
    print("saving...")
    put_txt_to_file(train_q, dataset_root + os.path.sep + file_train_q_file)
    put_txt_to_file(train_a, dataset_root + os.path.sep + file_train_a_file)
    put_txt_to_file(test_q, dataset_root + os.path.sep + file_test_q_file)
    put_txt_to_file(test_a, dataset_root + os.path.sep + file_test_a_file)
    print("preprocess over.")

if __name__ == '__main__':
    config = load_config('file_config.yaml')
    preprocess(config)