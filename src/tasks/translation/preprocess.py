from src.utilities.load_arguments import load_arguments
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
import os, sys
from tqdm import tqdm
import spacy
spacy_tokenizer = spacy.load("en_core_web_sm")


def preprocess(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    project_root = arguments['general']['project_root']
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
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
    data_text_en = get_txt_from_file(project_root + os.path.sep + craw_corpus_en)
    data_text_zh = get_txt_from_file(project_root + os.path.sep + craw_corpus_zh)
    data_set = []
    # 按照实际业务处理的分词配置筛选预料，过滤过短和过长的句子。
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        fun_name_en = arguments['tasks'][running_task]['word_vector']['tokenizer_en']
        fun_name_zh = arguments['tasks'][running_task]['word_vector']['tokenizer_zh']
        en_filter = getattr(module_obj, fun_name_en)
        zh_filter = getattr(module_obj, fun_name_zh)
    else:
        en_filter = get_bert_tokenizer(arguments, language='en').tokenize
        zh_filter = get_bert_tokenizer(arguments, language='zh').tokenize

    if (use_bert != 'static') and (use_bert != 'dynamic') and (fun_name_en == 'tokenize_en_bySpacy'):
        for i,text in tqdm(enumerate(data_text_en)):
            en_len = count_token(text)
            zh_len = len(zh_filter(data_text_zh[i]))
            if (en_len > 4) and (en_len < max_len) and (zh_len < max_len):
                data_set.append((data_text_zh[i], text))
    else:
        for i,text in tqdm(enumerate(data_text_en)):
            en_len = len(en_filter(text))
            zh_len = len(zh_filter(data_text_zh[i]))
            if (en_len > 4) and (en_len < max_len) and (zh_len < max_len):
                data_set.append((data_text_zh[i], text))

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