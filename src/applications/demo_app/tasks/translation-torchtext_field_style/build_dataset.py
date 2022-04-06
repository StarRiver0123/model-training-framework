import sys,time
from torchtext.legacy.vocab import Vectors
from torchtext.legacy.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.modules.tokenizers.tokenizer import *
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'



def build_train_dataset_and_vocab_pipeline(config):
    # step 1: load dataset
    train_dataset, valid_dataset = load_train_valid_split_dataset(config)
    # step 2: build field
    src_field, tgt_field = build_field(config)
    # step 3: build examples
    train_examples = build_examples(train_dataset, src_field, tgt_field, num_workers=-1)
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
    src_corpus_file = config['net_structure']['dataset']['src_train_file']
    tgt_corpus_file = config['net_structure']['dataset']['tgt_train_file']
    valid_size = config['net_structure']['dataset']['valid_size']
    random_state = config['general']['random_state']
    src_data_text = get_txt_from_file(dataset_root + os.path.sep + src_corpus_file)
    tgt_data_text = get_txt_from_file(dataset_root + os.path.sep + tgt_corpus_file)
    data_set = list(zip(src_data_text, tgt_data_text))
    train_set, valid_set = train_test_split(data_set, test_size=valid_size, shuffle=True, random_state=random_state)
    return train_set, valid_set


def load_test_dataset(config):
    # return test set
    dataset_root = config['dataset_root']
    src_test_file = config['net_structure']['dataset']['src_test_file']
    tgt_test_file = config['net_structure']['dataset']['tgt_test_file']
    src_data_text = get_txt_from_file(dataset_root + os.path.sep + src_test_file)
    tgt_data_text = get_txt_from_file(dataset_root + os.path.sep + tgt_test_file)
    data_set = list(zip(src_data_text, tgt_data_text))
    return data_set


def build_field(config):
    module_obj = sys.modules[tokenizer_package_path]
    src_tokenizer = config['net_structure']['tokenizer']['src_tokenizer']
    tgt_tokenizer = config['net_structure']['tokenizer']['tgt_tokenizer']
    try:
        sos_token = config['dataset']['general_symbol']['sos_token']
        eos_token = config['dataset']['general_symbol']['eos_token']
        unk_token = config['dataset']['general_symbol']['unk_token']
        pad_token = config['dataset']['general_symbol']['pad_token']
    except KeyError:
        sos_token = config['symbol_config']['sos_token']
        eos_token = config['symbol_config']['eos_token']
        unk_token = config['symbol_config']['unk_token']
        pad_token = config['symbol_config']['pad_token']
    SOURCE_FIELD = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, src_tokenizer), batch_first=True,
                     fix_length=None, init_token=sos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token)
    TARGET_FIELD = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, tgt_tokenizer), batch_first=True,
                         fix_length=None, init_token=sos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token)
    return SOURCE_FIELD, TARGET_FIELD



def build_vocab_in_field(config, src_field, tgt_field, train_examples):
    used_model = config['net_structure']['model']
    if ('symbol_config' not in config.keys()) or (config['symbol_config'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'symbol_config': {}})
    if ('model_config' not in config.keys()) or (config['model_config'] is None):     # 用于保存模型参数和测试部署阶段使用
        config.update({'model_config': {}})
    if ('model_vocab' not in config.keys()) or (config['model_vocab'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'model_vocab': {}})
    config['model_config'].update({'model_name': used_model})
    config['model_config'].update(config['model'][used_model])

    # build the vocab and vector
    src_vectors_file = config['word_vector_root'] + os.path.sep + config['net_structure']['word_vector']['src_vectors_file']
    tgt_vectors_file = config['word_vector_root'] + os.path.sep + config['net_structure']['word_vector']['tgt_vectors_file']
    src_vectors = Vectors(name=src_vectors_file)
    tgt_vectors = Vectors(name=tgt_vectors_file)
    src_field.build_vocab(train_examples, vectors=src_vectors)
    tgt_field.build_vocab(train_examples, vectors=tgt_vectors)
    config['symbol_config'].update({'sos_token': config['dataset']['general_symbol']['sos_token'],
                                      'eos_token': config['dataset']['general_symbol']['eos_token'],
                                      'unk_token': config['dataset']['general_symbol']['unk_token'],
                                      'pad_token': config['dataset']['general_symbol']['pad_token'],
                                      'sos_idx': src_field.vocab.stoi[config['dataset']['general_symbol']['sos_token']],
                                      'eos_idx': src_field.vocab.stoi[config['dataset']['general_symbol']['eos_token']],
                                      'unk_idx': src_field.vocab.stoi[config['dataset']['general_symbol']['unk_token']],
                                      'pad_idx': src_field.vocab.stoi[config['dataset']['general_symbol']['pad_token']]
                                      })
    config['model_config'].update({'src_vocab_len': src_field.vocab.vectors.shape[0],
                                   'tgt_vocab_len': tgt_field.vocab.vectors.shape[0]
                                   })
    config['model_vocab'].update({'src_vocab_stoi': src_field.vocab.stoi,
                                  'src_vocab_itos': src_field.vocab.itos,
                                  'tgt_vocab_stoi': tgt_field.vocab.stoi,
                                  'tgt_vocab_itos': tgt_field.vocab.itos
                                  })


def build_train_data_iterator(config, train_examples, valid_examples=None):
    batch_size = config['training']['batch_size']
    device = config['general']['device']
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
    test_iter = BucketIterator(dataset=test_examples, batch_size=batch_size_for_test, shuffle=True, train=False,
                               sort=False, device=device)
    return test_iter