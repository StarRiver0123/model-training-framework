import re, os, random
from tqdm import tqdm
import yaml
import torchtext.legacy.data as dt
from multiprocessing import cpu_count, Process, Pool
from transformers import BertTokenizer, BertModel, BertConfig
from collections import defaultdict
from torch.utils.data import DataLoader

def get_config(config_file):
    args = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f.read())
    return args

def get_config_from_yamls(project_root, arg_file_root):
    config_root = project_root + os.path.sep + arg_file_root
    config_paths = []
    config_paths.append(config_root)
    config = {}
    while len(config_paths):
        f_p = config_paths.pop()
        if os.path.isfile(f_p):
            with open(f_p, 'r', encoding='utf-8') as f:
                args = yaml.safe_load(f.read())
            config.update(args)
        if os.path.isdir(f_p):
            for sub_f_p in os.listdir(f_p):
                config_paths.append(f_p + os.path.sep + sub_f_p)
    return config

def get_txt_from_file(file, encoding='utf-8'):
    data_set = []
    with open(file, "r", encoding=encoding) as f:
        for line in f:
            data_set.append(line.strip())
    return data_set


def put_txt_to_file(txt_list, abspath_file, encoding='utf-8'):
    file_path = os.path.dirname(abspath_file)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(abspath_file, "w", encoding=encoding) as f:
        for txt in txt_list:
            if txt.endswith('\n'):
                f.write(txt)
            else:
                f.write(txt + '\n')


def _process_job(data_tuple, fields):
    return [dt.Example.fromlist(data_pair, fields) for data_pair in tqdm(data_tuple)]


def build_examples(data_tuple, source_field, target_field, is_train=True, num_workers=1):
    fields = [("Source", source_field), ("Target", target_field)]
    examples = []
    if (num_workers == -1) or (num_workers > int(cpu_count() * 0.8)):
        num_workers = int(cpu_count() * 0.8)
    if num_workers > 1:
        p_pool = Pool(num_workers)
        sub_len = int(len(data_tuple) / num_workers)   # 不能整除地部分丢弃，
    if is_train:
        if num_workers > 1:
            p_list = []
            examples = []
            for i in range(num_workers):
                p_list.append(p_pool.apply_async(func=_process_job, args=(data_tuple[i*sub_len: (i+1)*sub_len], fields)))    #不会出现索引溢出，因为计算sub_len时已经通过除法取整保证了这一点。
            p_pool.close()
            p_pool.join()
            for i in range(num_workers -1, -1, -1):  # 倒序，为了能从队尾删除
                examples += p_list[i].get()
                del p_list[i]
        else:
            examples = [dt.Example.fromlist(data_pair, fields) for data_pair in tqdm(data_tuple)]
    else:
        for source, target in tqdm(data_tuple):
            examples.append(dt.Example.fromlist([source, None], fields))
    return dt.Dataset(examples, fields)




# class DataExamples_withTorchText(dt.Dataset):
#     def __init__(self, data_tuple, source_field, target_field, is_train=True, workers=1):
#         fields = [("Source", source_field), ("Target", target_field)]
#         examples = []
#         if workers > 1:
#             if workers > cpu_count():
#                 workers = cpu_count()
#             p_pool = Pool(workers)
#         if is_train:
#             if workers > 1:
#                 p_list = []
#                 examples = []
#                 for i in range(workers):
#                     p_list.append(p_pool.apply_async(func=self._run_job, args=(data_tuple, fields, i, workers)))
#                 p_pool.close()
#                 p_pool.join()
#                 for i in examples:
#                     examples.extend(i.get())
#             else:
#                 examples = [dt.Example.fromlist(data_pair, fields) for data_pair in tqdm(data_tuple)]
#             # for source, target in tqdm(data_tuple):
#             #     examples.append(dt.Example.fromlist([source, target], fields))
#         else:
#             for source, target in tqdm(data_tuple):
#                 examples.append(dt.Example.fromlist([source, None], fields))
#         super().__init__(examples, fields)
#
#     def _run_job(self, data_tuple, fields, index, total):
#         return [dt.Example.fromlist(data_pair, fields) for data_pair in tqdm(data_tuple[index::total])]

# class DataExamples_withTorchText(dt.Dataset):
#     def __init__(self, data_tuple, source_field, target_field, is_train=True):
#         fields = [("Source", source_field), ("Target", target_field)]
#         examples = []
#         if is_train:
#             examples = [dt.Example.fromlist(data_pair, fields) for data_pair in tqdm(data_tuple)]
#             # for source, target in tqdm(data_tuple):
#             #     examples.append(dt.Example.fromlist([source, target], fields))
#         else:
#             for source, target in tqdm(data_tuple):
#                 examples.append(dt.Example.fromlist([source, None], fields))
#         super().__init__(examples, fields)

#
# class build_triplet_examples(dt.Dataset):
#     def __init__(self, data_tuple, source_field, target_field, negative_field, is_train=True):
#         fields = [("Source", source_field), ("Target", target_field), ("Negative", negative_field)]
#         examples = []
#         data_len = len(data_tuple)
#         if is_train:
#             # examples = [dt.Example.fromlist(data_pair, fields) for data_pair in tqdm(data_tuple)]
#             for idx, (source, target) in tqdm(enumerate(data_tuple)):
#                 i = random.randint(0, data_len - 1)
#                 while data_tuple[i][0] == source:
#                     i = random.randint(0, data_len - 1)
#                 examples.append(dt.Example.fromlist([source, target, data_tuple[i][1]], fields))
#         else:
#             for source, target in tqdm(data_tuple):
#                 examples.append(dt.Example.fromlist([source, None], fields))
#         super().__init__(examples, fields)
#
#
#
# def build_field_vocab_special_tokens_from_bert_tokenizer(field, tokenizer):
#     field.build_vocab()
#     field.vocab.stoi.clear()
#     field.vocab.stoi.update(tokenizer.vocab)
#     field.vocab.itos.clear()
#     field.vocab.itos.extend(tokenizer.vocab.keys())
#     field.vocab.stoi.default_factory = type(0)
#     # 创建vocab时由于没有指定unk_token，则stoi的defaultdict类型缺少了default_factory, 从而跟普通dict没有区别，需要重新设置default_factory.
#     # 四个特殊的token不能直接写到filed对象的定义里，那样会使得build_vocab时vocab增加四个值从而跟vector不匹配。但是又不能不写到field的属性里去。不写的话会给数据集的处理带来问题。所以只能写到这里。
#     # 本函数中特殊字符赋值应该在build_vocab之后。特殊字符已经在模型词典中
#     field.init_token = tokenizer.cls_token
#     field.eos_token = tokenizer.sep_token
#     field.unk_token = tokenizer.unk_token
#     field.pad_token = tokenizer.pad_token
#
#
# def build_field_vocab_special_tokens(field, stoi, itos, sos_token=None, eos_token=None, pad_token=None, unk_token=None):
#     field.build_vocab()
#     field.vocab.stoi.clear()
#     field.vocab.stoi.update(stoi)
#     field.vocab.itos.clear()
#     field.vocab.itos.extend(itos)
#     field.vocab.stoi.default_factory = type(0)
#     # 创建vocab时由于没有指定unk_token，则stoi的defaultdict类型缺少了default_factory, 从而跟普通dict没有区别，需要重新设置default_factory.
#     # 四个特殊的token不能直接写到filed对象的定义里，在运行build_vocab之前不能存在于那样会使得build_vocab时vocab增加四个值从而跟vector不匹配。但是又不能不写到field的属性里去。不写的话会给数据集的处理带来问题。所以只能写到这里。
#     # 本函数中特殊字符赋值应该在build_vocab之后。特殊字符已经在词典中
#     if sos_token is not None:
#         field.init_token = sos_token
#     if eos_token is not None:
#         field.eos_token = eos_token
#     if pad_token is not None:
#         field.pad_token = pad_token
#     if unk_token is not None:
#         field.unk_token = unk_token

