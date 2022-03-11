import sys, random
import pickle as pk
import copy, numpy
from torchtext.legacy.vocab import Vectors
from torchtext.legacy.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig
from collections import defaultdict
from src.utilities.load_data import *
from src.modules.tokenizers.tokenizer import *
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'



def build_train_dataset_and_vocab_pipeline(config):
    # step 1: load dataset
    train_dataset, valid_dataset = load_train_valid_split_dataset(config)
    # step 2: build field
    src_field, tgt_field = build_field(config)
    # step 3: build examples
    train_examples = build_examples(train_dataset, src_field, tgt_field)
    valid_examples = build_examples(valid_dataset, src_field, tgt_field)
    # step 4: build vocab and update config
    build_vocab_in_field(config, src_field, tgt_field, train_examples)
    # step 5: build iterator
    train_iter, valid_iter = build_train_data_iterator(config, train_examples, valid_examples)
    return train_iter, valid_iter, src_field, tgt_field


def build_test_dataset_pipeline(config):
    # step 1: load dataset
    test_dataset = load_test_dataset(config)
    # step 2: build field
    src_field, tgt_field = build_field(config)
    # step 3: build examples
    test_examples = build_examples(test_dataset, src_field, tgt_field)
    # step 4: build iterator
    test_iter = build_test_data_iterator(config, test_examples)
    return test_iter, src_field, tgt_field


def load_train_valid_split_dataset(config):
    # return train set, valid set
    dataset_root = config['dataset_root']
    train_set_file = config['net_structure']['dataset']['train_set_file']
    valid_set_file = config['net_structure']['dataset']['valid_set_file']
    used_model = config['net_structure']['model']
    use_bert = config['model'][used_model]['use_bert']
    if use_bert:
        bert_model_root = config['bert_model_root']
        bert_model_name = bert_model_root + os.path.sep + config['net_structure']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        src_sos_token = tokenizer.cls_token
        src_eos_token = tokenizer.sep_token
    # else: # 如果不用bert，则sos和eos在定义field时直接传入参数了，所以这里构建数据时不需要添加。
    #     src_sos_token = None  # config['dataset']['general_symbol']['sos_token']
    #     src_eos_token = None  # config['dataset']['general_symbol']['eos_token']

    train_list = get_txt_from_file(dataset_root + os.path.sep + train_set_file)
    valid_list = get_txt_from_file(dataset_root + os.path.sep + valid_set_file)
    train_set = []
    for i, text in enumerate(train_list):
        if i % 2 == 0:
            if use_bert:
                s = [src_sos_token] + text.strip().split() + [src_eos_token]
            else:
                s = text.strip().split()
        else:
            t = text.strip().split()
            train_set.append((s,t))
    valid_set = []
    for i, text in enumerate(valid_list):
        if i % 2 == 0:
            if use_bert:
                s = [src_sos_token] + text.strip().split() + [src_eos_token]
            else:
                s = text.strip().split()
        else:
            t = text.strip().split()
            valid_set.append((s,t))
    return train_set, valid_set



def load_test_dataset(config):
    # return test set
    dataset_root = config['dataset_root']
    test_set_file = config['net_structure']['dataset']['test_set_file']
    use_bert = config['model_config']['use_bert']
    if use_bert:
        bert_model_root = config['bert_model_root']
        bert_model_name = bert_model_root + os.path.sep + config['net_structure']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        src_sos_token = tokenizer.cls_token
        src_eos_token = tokenizer.sep_token
    test_list = get_txt_from_file(dataset_root + os.path.sep + test_set_file)
    test_set = []
    for i, text in enumerate(test_list):
        if i % 2 == 0:
            if use_bert:
                s = [src_sos_token] + text.strip().split() + [src_eos_token]
            else:
                s = text.strip().split()
        else:
            t = text.strip().split()
            test_set.append((s,t))
    return test_set


def build_tag_vocab(train_dataset, valid_dataset=None, test_dataset=None):
    tags = defaultdict(int)
    for line in train_dataset:
        for tag in line[1]:
            tags[tag] += 1
    if valid_dataset is not None:     # 把验证集和测试集的标签也加入词典，增加匹配成功率。
        for line in valid_dataset:
            for tag in line[1]:
                tags[tag] += 1
    if test_dataset is not None:     # 把验证集和测试集的标签也加入词典，增加匹配成功率。
        for line in test_dataset:
            for tag in line[1]:
                tags[tag] += 1
    stoi = dict(zip(tags.keys(), range(4, len(tags)+4)))  #考虑pad和unk（sos,eos）已经占据了前几位
    itos = list(tags.keys())
    return stoi, itos


def build_field(config):
    try:
        used_model = config['net_structure']['model']
        use_bert = config['model'][used_model]['use_bert']
    except KeyError:
        use_bert = config['model_config']['use_bert']
    if not use_bert:
        module_obj = sys.modules[tokenizer_package_path]
        fun_name = config['net_structure']['tokenizer']
        try:
            src_sos_token = config['dataset']['general_symbol']['sos_token']
            src_eos_token = config['dataset']['general_symbol']['eos_token']
            src_unk_token = config['dataset']['general_symbol']['unk_token']
            src_pad_token = config['dataset']['general_symbol']['pad_token']
        except KeyError:
            src_sos_token = config['symbol_config']['src_sos_token']
            src_eos_token = config['symbol_config']['src_eos_token']
            src_unk_token = config['symbol_config']['src_unk_token']
            src_pad_token = config['symbol_config']['src_pad_token']
        SOURCE_FIELD = Field(sequential=True, use_vocab=True, tokenize=None,
                             preprocessing=None, batch_first=True,
                             fix_length=None, init_token=src_sos_token, eos_token=src_eos_token,
                             pad_token=src_pad_token, unk_token=src_unk_token)
    else:   # use bert
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_file)
        try:
            src_pad_idx = tokenizer.pad_token_id
            src_unk_idx = tokenizer.unk_token_id
        except KeyError:
            src_pad_idx = config['symbol_config']['src_pad_idx']
            src_unk_idx = config['symbol_config']['src_unk_idx']
        SOURCE_FIELD = Field(sequential=True, use_vocab=False, tokenize=None,  preprocessing=tokenizer.convert_tokens_to_ids, batch_first=True,
                     fix_length=None, init_token=None, eos_token=None, pad_token=src_pad_idx, unk_token=src_unk_idx)
    try:
        tgt_sos_token = config['dataset']['general_symbol']['sos_token']
        tgt_eos_token = config['dataset']['general_symbol']['eos_token']
        tgt_unk_token = config['dataset']['general_symbol']['unk_token']
        tgt_pad_token = config['dataset']['general_symbol']['pad_token']
    except KeyError:
        tgt_sos_token = config['symbol_config']['tgt_sos_token']
        tgt_eos_token = config['symbol_config']['tgt_eos_token']
        tgt_unk_token = config['symbol_config']['tgt_unk_token']
        tgt_pad_token = config['symbol_config']['tgt_pad_token']
    # 注意：如果不用bert的字典，则给Field需要传入pad和unk的话必须是'<pad>'和'<unk>'，因为生成batch源码中给写死了，根据use_vocab和sequential和pad_token来确定vocab中defaultdict的参数。
    # source_field不需要传入sos_token和eos_token。因为序列化已经在使用filed之前完成了，也不需要构建字典词表，sos_token和eos_token的添加不需要通过filed属性来完成，而是在random_sample采样生成数据时加上了。
    TARGET_FIELD = Field(sequential=True, use_vocab=True, tokenize=None, batch_first=True,
                         fix_length=None, init_token=tgt_sos_token, eos_token=tgt_eos_token,
                         pad_token=tgt_pad_token, unk_token=tgt_unk_token)
    # target_filed不一定需要传入sos_token，eos_token，但必须传入unk_token和pad_token，因为在Vocab.py类源码中写死了要根据unk来初始化vocab的defaultdict。
    return SOURCE_FIELD, TARGET_FIELD


def build_vocab_in_field(config, src_field, tgt_field, train_examples):
    used_model = config['net_structure']['model']
    use_bert = config['model'][used_model]['use_bert']
    if ('symbol_config' not in config.keys()) or (config['symbol_config'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'symbol_config': {}})
    if ('model_config' not in config.keys()) or (config['model_config'] is None):     # 用于保存模型参数和测试部署阶段使用
        config.update({'model_config': {}})
    if ('model_vocab' not in config.keys()) or (config['model_vocab'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'model_vocab': {}})
    config['model_config'].update({'model_name': used_model})
    config['model_config'].update(config['model'][used_model])
    # build the src vocab
    if not use_bert:
        # build the vocab and vector
        vector_file = config['word_vector_root'] + os.path.sep + config['net_structure']['word_vector_file']
        word_vectors = Vectors(name=vector_file)
        src_field.build_vocab(train_examples, vectors=word_vectors)
        src_sos_token = config['dataset']['general_symbol']['sos_token']
        src_eos_token = config['dataset']['general_symbol']['eos_token']
        src_unk_token = config['dataset']['general_symbol']['unk_token']
        src_pad_token = config['dataset']['general_symbol']['pad_token']
        config['symbol_config'].update({'src_sos_token': src_sos_token,
                                          'src_eos_token': src_eos_token,
                                          'src_unk_token': src_unk_token,
                                          'src_pad_token': src_pad_token,
                                          'src_sos_idx': src_field.vocab.stoi[src_sos_token],
                                          'src_eos_idx': src_field.vocab.stoi[src_eos_token],
                                          'src_unk_idx': src_field.vocab.stoi[src_unk_token],
                                          'src_pad_idx': src_field.vocab.stoi[src_pad_token]
                                          })
        config['model_config'].update({'vocab_len': src_field.vocab.vectors.shape[0]})
        config['model_vocab'].update({'src_vocab_stoi': src_field.vocab.stoi,
                                      'src_vocab_itos': src_field.vocab.itos
                                      })
    else:   # use bert
        # build the symbol only
        # 注意在filed里面已经加载了token2id的接口，所以这里不需要在创建vocab
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_file)
        configer = BertConfig.from_pretrained(bert_model_file)
        config['symbol_config'].update({'src_sos_token': tokenizer.cls_token,
                                         'src_eos_token': tokenizer.sep_token,
                                         'src_unk_token': tokenizer.unk_token,
                                         'src_pad_token': tokenizer.pad_token,
                                         'src_sos_idx': tokenizer.cls_token_id,
                                         'src_eos_idx': tokenizer.sep_token_id,
                                         'src_unk_idx': tokenizer.unk_token_id,
                                         'src_pad_idx': tokenizer.pad_token_id
                                          })
        config['model_config'].update({'bert_config': configer})
    # build the target vocab
    train_dataset, valid_dataset = load_train_valid_split_dataset(config)
    test_dataset = load_test_dataset(config)
    stoi, itos = build_tag_vocab(train_dataset, valid_dataset, test_dataset)
    # 注意：如果不用bert的字典，则给Field传入的pad和unk必须是'<pad>'和'<unk>'，因为源码中给写死了。
    # 否则stoi这个defaultdict的default_factory会为None，则跟dict一样了。这样在使用字典遇到字典中没有的关键词，则报KeyError错误。
    tgt_field.build_vocab()
    tgt_field.vocab.stoi.update(stoi)
    tgt_field.vocab.itos.extend(itos)
    tgt_sos_token = config['dataset']['general_symbol']['sos_token']
    tgt_eos_token = config['dataset']['general_symbol']['eos_token']
    tgt_unk_token = config['dataset']['general_symbol']['unk_token']
    tgt_pad_token = config['dataset']['general_symbol']['pad_token']
    config['symbol_config'].update({'tgt_sos_token': tgt_sos_token,
                                    'tgt_eos_token': tgt_eos_token,
                                    'tgt_unk_token': tgt_unk_token,
                                    'tgt_pad_token': tgt_pad_token,
                                    'tgt_sos_idx': tgt_field.vocab.stoi[tgt_sos_token],
                                    'tgt_eos_idx': tgt_field.vocab.stoi[tgt_eos_token],
                                    'tgt_unk_idx': tgt_field.vocab.stoi[tgt_unk_token],
                                    'tgt_pad_idx': tgt_field.vocab.stoi[tgt_pad_token]
                                    })
    config['model_config'].update({'num_tags': len(itos) + 4})  # 考虑真正的标签加上pad和unk(sos,eos)
    config['model_vocab'].update({'tgt_vocab_stoi': tgt_field.vocab.stoi,
                                  'tgt_vocab_itos': tgt_field.vocab.itos
                                  })


def build_train_data_iterator(config, train_examples, valid_examples=None):
    batch_size = config['training']['batch_size']
    device = config['general']['device']
    #负样本最好是在训练过程中引入，这样可以在不同的epoch中使用不同的负样本。
    train_iter = BucketIterator(dataset=train_examples, batch_size=batch_size, sort_key=lambda x: len(x.Source),
                                shuffle=True, sort_within_batch=True, sort=True, device=device)
    config['net_structure']['dataset'].update({'train_set_size': len(train_examples.examples)})
    # 如果train_set为空，则不管valid_set是否为空，都不做处理，因为valid是针对train的结果而言的。valid可以看作是train的一个过程。
    if valid_examples is not None:
        valid_iter = BucketIterator(dataset=valid_examples, batch_size=batch_size,
                                    sort_key=lambda x: len(x.Source), shuffle=True, sort_within_batch=True,
                                    sort=True, train=False, device=device)
    if valid_examples is not None:
        return train_iter, valid_iter
    else:
        return train_iter


def build_test_data_iterator(config, test_examples):
    batch_size_for_test = 1   #config['testing']['batch_size']
    device = config['general']['device']
    test_iter = BucketIterator(dataset=test_examples, batch_size=batch_size_for_test, train=False,
                               sort=False, device=device)
    return test_iter

