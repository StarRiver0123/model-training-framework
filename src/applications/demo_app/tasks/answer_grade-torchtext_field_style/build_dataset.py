import sys
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
    text_field = build_field(config)
    # step 3: build examples
    train_examples = build_triplet_examples(train_dataset, text_field, text_field, text_field)
    valid_examples = build_examples(valid_dataset, text_field, text_field)
    # step 4: build vocab and update config
    build_vocab_in_field(config, text_field, train_examples)
    # step 5: build iterator
    train_iter, valid_iter = build_train_data_iterator(config, train_examples, valid_examples)
    return train_iter, valid_iter, text_field


def build_test_dataset_pipeline(config):
    # step 1: load dataset
    test_dataset = load_test_dataset(config)
    # step 2: build field
    text_field = build_field(config)
    # step 3: build examples
    test_examples = build_examples(test_dataset, text_field, text_field)
    # step 4: build iterator
    test_iter = build_test_data_iterator(config, test_examples)
    return test_iter, text_field


def load_train_valid_split_dataset(config):
    # return train set, valid set
    dataset_root = config['dataset_root']
    train_q_file = config['net_structure']['dataset']['train_q_file']
    train_a_file = config['net_structure']['dataset']['train_a_file']
    valid_size = config['training']['valid_size']
    random_state = config['general']['random_state']
    data_text_q = get_txt_from_file(dataset_root + os.path.sep + train_q_file)
    data_text_a = get_txt_from_file(dataset_root + os.path.sep + train_a_file)
    data_set = list(zip(data_text_q, data_text_a))
    train_dataset, valid_dataset = train_test_split(data_set, test_size=valid_size, shuffle=True, random_state=random_state)
    return train_dataset, valid_dataset


def load_test_dataset(config):
    # return test set
    dataset_root = config['dataset_root']
    test_q_file = config['net_structure']['dataset']['test_q_file']
    test_a_file = config['net_structure']['dataset']['test_a_file']
    data_text_q = get_txt_from_file(dataset_root + os.path.sep + test_q_file)
    data_text_a = get_txt_from_file(dataset_root + os.path.sep + test_a_file)
    data_dataset = list(zip(data_text_q, data_text_a))
    return data_dataset


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
            unk_token = config['dataset']['general_symbol']['unk_token']
            pad_token = config['dataset']['general_symbol']['pad_token']
        except KeyError:
            unk_token = config['symbol_config']['unk_token']
            pad_token = config['symbol_config']['pad_token']
        if fun_name == 'tokenize_zh_bySpacy':
            TEXT_FIELD = Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='zh',
                             lower=True, batch_first=True,
                             fix_length=None, init_token=None, eos_token=None, pad_token=pad_token,
                             unk_token=unk_token)
        else:
            TEXT_FIELD = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, fun_name), lower=True, batch_first=True,
                             fix_length=None, init_token=None, eos_token=None, pad_token=pad_token, unk_token=unk_token)
    else:   # use bert
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['bert_model']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_file)
        TEXT_FIELD = Field(sequential=True, use_vocab=True, tokenize=tokenizer.tokenize, batch_first=True,
                         fix_length=None, init_token=None, eos_token=None, pad_token=None, unk_token=None)
    return TEXT_FIELD


def build_vocab_in_field(config, text_field, train_examples):
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
    if not use_bert:
        # build the vocab and vector
        vector_file = config['word_vector_root'] + os.path.sep + config['net_structure']['word_vector_file']
        word_vectors = Vectors(name=vector_file)
        text_field.build_vocab(train_examples, vectors=word_vectors)
        sos_token = config['dataset']['general_symbol']['sos_token']
        eos_token = config['dataset']['general_symbol']['eos_token']
        unk_token = config['dataset']['general_symbol']['unk_token']
        pad_token = config['dataset']['general_symbol']['pad_token']
        config['symbol_config'].update({'sos_token': sos_token,
                                          'eos_token': eos_token,
                                          'unk_token': unk_token,
                                          'pad_token': pad_token,
                                          'sos_idx': text_field.vocab.stoi[sos_token],
                                          'eos_idx': text_field.vocab.stoi[eos_token],
                                          'unk_idx': text_field.vocab.stoi[unk_token],
                                          'pad_idx': text_field.vocab.stoi[pad_token]
                                          })
        config['model_config'].update({'vocab_len': text_field.vocab.vectors.shape[0]})
    else:   # use bert
        # build the vocab only
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['bert_model']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_file)
        configer = BertConfig.from_pretrained(bert_model_file)
        build_field_vocab_special_tokens_from_bert_tokenizer(text_field, tokenizer)
        config['symbol_config'].update({'sos_token': tokenizer.cls_token,
                                         'eos_token': tokenizer.sep_token,
                                         'unk_token': tokenizer.unk_token,
                                         'pad_token': tokenizer.pad_token,
                                         'sos_idx': tokenizer.cls_token_id,
                                         'eos_idx': tokenizer.sep_token_id,
                                         'unk_idx': tokenizer.unk_token_id,
                                         'pad_idx': tokenizer.pad_token_id
                                          })
        config['model_config'].update({'bert_config': configer})
        # config['model_config'].update({'d_model': configer.hidden_size,
        #                                'nhead': configer.num_attention_heads,
        #                                'vocab_len': configer.vocab_size})
    config['model_vocab'].update({'vocab_stoi': text_field.vocab.stoi,
                                  'vocab_itos': text_field.vocab.itos
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
    test_iter = BucketIterator(dataset=test_examples, batch_size=batch_size_for_test, train=False, shuffle=True,
                               sort=False, device=device)
    return test_iter

