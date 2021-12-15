import os, sys, json, numpy, torch
from tqdm import tqdm
# import spacy
# spacy_tokenizer = spacy.load("en_core_web_sm")
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.utilities.load_arguments import load_arguments
from src.models.tokenizers.tokenizer import *


def preprocess(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    project_root = arguments['general']['project_root']
    train_json = arguments['tasks'][running_task]['dataset']['train_json']
    valid_json = arguments['tasks'][running_task]['dataset']['valid_json']
    craw_corpus_en = arguments['tasks'][running_task]['dataset']['corpus_en']
    craw_corpus_zh = arguments['tasks'][running_task]['dataset']['corpus_zh']
    file_train_en = arguments['tasks'][running_task]['dataset']['train_en']
    file_train_zh = arguments['tasks'][running_task]['dataset']['train_zh']
    file_test_en = arguments['tasks'][running_task]['dataset']['test_en']
    file_test_zh = arguments['tasks'][running_task]['dataset']['test_zh']
    test_size = arguments['training'][running_task]['test_size']
    max_len = arguments['model'][used_model]['max_len'] - 4   # condidering sos, eos, pad, unk
    random_state = arguments['general']['random_state']
    module_obj = sys.modules['src.utilities.load_data']
    use_bert = arguments['tasks'][running_task]['use_bert']
    device = arguments['general']['device']
    #
    # # max_lines = 2000
    # data_text_en = []
    # data_text_zh = []
    # train_json_file = project_root + os.path.sep + train_json
    # valid_json_file = project_root + os.path.sep + valid_json
    # max_text_len = 0
    # with open(train_json_file, 'r', encoding='utf-8') as f:
    #     json_txt = f.readlines()
    # for txt in json_txt:
    #     # if len(data_text_en) > max_lines:
    #     #     break
    #     t = json.loads(txt)
    #     data_text_en.append(t['english'] + '\n')
    #     data_text_zh.append(t['chinese'] + '\n')
    # with open(valid_json_file, 'r', encoding='utf-8') as f:
    #     json_txt = f.readlines()
    # for txt in json_txt:
    #     # if len(data_text_en) > max_lines:
    #     #     break
    #     t = json.loads(txt)
    #     data_text_en.append(t['english'] + '\n')
    #     data_text_zh.append(t['chinese'] + '\n')

    # data_text_en = get_txt_from_file(project_root + os.path.sep + craw_corpus_en)
    # data_text_zh = get_txt_from_file(project_root + os.path.sep + craw_corpus_zh)
    #
    # data_set = []
    # # 按照实际业务处理的分词配置筛选预料，过滤过短和过长的句子。
    # if use_bert not in ['static', 'dynamic']:
    #     fun_name_en = arguments['tasks'][running_task]['word_vector']['tokenizer_en']
    #     fun_name_zh = arguments['tasks'][running_task]['word_vector']['tokenizer_zh']
    #     en_filter = getattr(module_obj, fun_name_en)
    #     zh_filter = getattr(module_obj, fun_name_zh)
    # else:
    #     # en_filter = get_bert_tokenizer(arguments, language='en').tokenize
    #     # zh_filter = get_bert_tokenizer(arguments, language='zh').tokenize
    #     en_filter = get_bert_tokenizer(arguments, language='en')
    #     zh_filter = get_bert_tokenizer(arguments, language='zh')
    #
    # if (use_bert not in ['static', 'dynamic']) and (fun_name_en == 'tokenize_en_bySpacy'):
    #     for i,text in tqdm(enumerate(data_text_en)):
    #         en_len = count_token(text)
    #         zh_len = len(zh_filter(data_text_zh[i]))
    #         if (en_len > 4) and (en_len < max_len) and (zh_len < max_len):
    #             data_set.append((data_text_zh[i], text))
    # elif use_bert in ['static', 'dynamic']:
    #     token_list_en = en_filter(data_text_en)['input_ids']
    #     token_list_zh = en_filter(data_text_zh)['input_ids']
    #     for i,token_list in tqdm(enumerate(token_list_en)):
    #         en_len = len(token_list)
    #         zh_len = len(token_list_zh[i])
    #         if (en_len > 4) and (en_len < max_len) and (zh_len < max_len):
    #             data_set.append((data_text_zh[i], data_text_en[i]))
    # else:
    #     for i,text in tqdm(enumerate(data_text_en)):
    #         en_len = len(en_filter(text))
    #         zh_len = len(zh_filter(data_text_zh[i]))
    #         if (en_len > 4) and (en_len < max_len) and (zh_len < max_len):
    #             data_set.append((data_text_zh[i], text))

    data_text_en = get_txt_from_file(project_root + os.path.sep + file_train_en)
    data_text_zh = get_txt_from_file(project_root + os.path.sep + file_train_zh)
    data_text_en2 = get_txt_from_file(project_root + os.path.sep + file_test_en)
    data_text_zh2 = get_txt_from_file(project_root + os.path.sep + file_test_zh)
    data_text_en += data_text_en2
    data_text_zh += data_text_zh2

    data_set = list(zip(data_text_zh, data_text_en))

    print("spliting...")
    train_set, test_set = train_test_split(data_set, test_size=test_size, shuffle=True, random_state=random_state)
    train_zh, train_en = list(zip(*train_set))
    test_zh, test_en = list(zip(*test_set))
    print("saving...")
    put_txt_to_file(train_zh, project_root + os.path.sep + file_train_zh)
    put_txt_to_file(train_en, project_root + os.path.sep + file_train_en)
    put_txt_to_file(test_zh, project_root + os.path.sep + file_test_zh)
    put_txt_to_file(test_en, project_root + os.path.sep + file_test_en)
    print("preprocess over.")

if __name__ == '__main__':
    arguments = load_arguments('file_config.yaml')
    preprocess(arguments)