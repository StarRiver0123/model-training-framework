import re
import jieba
from tqdm import tqdm
import torchtext.legacy.data as dt
import os
import spacy
spacy_tokenizer = spacy.load("en_core_web_sm")
from transformers import BertTokenizer, BertModel, BertConfig

def get_txt_from_file(file, encoding='utf-8'):
    data_set = []
    with open(file, "r", encoding=encoding) as f:
        data_set = f.readlines()
    return data_set


def put_txt_to_file(txt_list, abspath_file, encoding='utf-8'):
    file_path = os.path.dirname(abspath_file)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    with open(abspath_file, "w", encoding=encoding) as f:
        for txt in txt_list:
            f.write(txt)


def tokenize_en_bySplit(text):
    return [word for word in en_data_clean(text).split()]
    # 英语标点符号集：!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # 其中：单引号'，连接符号-，下划线_等，进行保留，其他的左右两侧加空格，但是.两侧如果都紧跟数字或字母则不加空格（小数和email地址）。

def tokenize_en_bySpacy(text):
    return [word.lemma_ for word in spacy_tokenizer(text)]

def count_token(text):
    text = re.sub(r'(?<=[^a-zA-Z0-9])|(?=[^a-zA-Z0-9])', " ", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\s]', '', text)
    return len(text)

def tokenize_en_byJieba(text):
    return [word for word in jieba.cut(text) if word != ' ']

def en_data_clean(text):
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"\n", " ", text)
    text = text.lower()
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"e-mail", "email", text)
    # text = re.sub(r'((?<=[&,/:;`~\#\<\=\>\!\*\+\.\"\$\^\?\(\)\[\\\]\{\|\}])|(?=[&,/:;`~\#\<\=\>\!\*\+\.\"\$\^\?\(\)\[\\\]\{\|\}]))(?<![0-9a-zA-Z]\.)(?!\.[0-9a-zA-Z])', " ", text)
    text = re.sub(
        r'(?<=[@,&,/:;`~\#\<\=\>\!\*\+\.\"\$\%\^\?\(\)\[\\\]\{\|\}\d])|(?=[@,&,/:;`~\#\<\=\>\!\*\+\.\"\$\%\^\?\(\)\[\\\]\{\|\}\d])',
        " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def zh_data_clean(text):
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r'(?<=[^a-zA-Z0-9\._@])|(?=[^a-zA-Z0-9\._@])', " ", text)
    return text


def tokenize_zh_byStrip(text):
    return [word for word in text.strip() if word != ' ']

def tokenize_zh_bySplit(text):
    return [word for word in zh_data_clean(text).split() if word != ' ']

def tokenize_zh_bySpacy(text):
    tokenizer = spacy.load("zh_core_web_sm")
    return [word.text for word in tokenizer(text)]

def tokenize_zh_byJieba(text):
    return [word for word in jieba.cut(text) if word != ' ']


class DataExamples_withTorchText(dt.Dataset):
    def __init__(self, data_tuple, source_field, target_field, is_train=True):
        fields = [("Source", source_field), ("Target", target_field)]
        examples = []
        if is_train:
            for source, target in tqdm(data_tuple):
                examples.append(dt.Example.fromlist([source, target], fields))
        else:
            for source, target in tqdm(data_tuple):
                examples.append(dt.Example.fromlist([source, None], fields))
        super().__init__(examples, fields)


def get_bert_tokenizer(arguments, language=None):
    # language: 'en', or 'zh'
    project_root = arguments['general']['project_root']
    running_task = arguments['general']['running_task']
    if language == 'en':
        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_en']
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    elif language == 'zh':
        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_zh']
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    else:
        tokenizer = None
    return tokenizer


def get_bert_model(arguments, language=None):
    # language: 'en', or 'zh'
    project_root = arguments['general']['project_root']
    running_task = arguments['general']['running_task']
    if language == 'en':
        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_en']
        bert_model = BertModel.from_pretrained(bert_model_name)
    elif language == 'zh':
        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_zh']
        bert_model = BertModel.from_pretrained(bert_model_name)
    else:
        bert_model = None
    return bert_model


def get_bert_configer(arguments, language=None):
    # language: 'en', or 'zh'
    project_root = arguments['general']['project_root']
    running_task = arguments['general']['running_task']
    if language == 'en':
        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_en']
        configer = BertConfig.from_pretrained(bert_model_name)
    elif language == 'zh':
        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_zh']
        configer = BertConfig.from_pretrained(bert_model_name)
    else:
        configer = None
    return configer


def init_field_vocab_special_tokens_from_model(field, tokenizer):
    field.build_vocab()
    field.vocab.stoi.update(tokenizer.vocab)
    field.vocab.itos.extend(tokenizer.vocab.keys())
    # 四个特殊的token不能直接写到filed对象的定义里，那样会使得build_vocab时vocab增加四个值从而跟vector不匹配。但是又不能不写到field的属性里去。不写的话会给数据集的处理带来问题。所以只能写到这里。
    # 本函数中特殊字符赋值应该在build_vocab之后。特殊字符已经在模型词典中
    field.init_token = tokenizer.cls_token
    field.eos_token = tokenizer.sep_token
    field.unk_token = tokenizer.unk_token
    field.pad_token = tokenizer.pad_token


def init_field_vocab_special_tokens(field, stoi, itos, start_token=None, end_token=None, pad_token=None, unk_token=None):
    field.build_vocab()
    field.vocab.stoi.update(stoi)
    field.vocab.itos.extend(itos)
    # 四个特殊的token不能直接写到filed对象的定义里，在运行build_vocab之前不能存在于那样会使得build_vocab时vocab增加四个值从而跟vector不匹配。但是又不能不写到field的属性里去。不写的话会给数据集的处理带来问题。所以只能写到这里。
    # 本函数中特殊字符赋值应该在build_vocab之后。特殊字符已经在词典中
    if start_token is not None:
        field.init_token = start_token
    if end_token is not None:
        field.eos_token = end_token
    if pad_token is not None:
        field.pad_token = pad_token
    if unk_token is not None:
        field.unk_token = unk_token

