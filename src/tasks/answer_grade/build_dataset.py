import sys
from torchtext.legacy.vocab import Vectors
from torchtext.legacy.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.models.tokenizers.tokenizer import *


def load_train_valid_split_set(arguments):
    # return train set, valid set
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    train_q = arguments['tasks'][running_task]['dataset']['train_q']
    train_a = arguments['tasks'][running_task]['dataset']['train_a']
    valid_size = arguments['training'][running_task]['valid_size']
    random_state = arguments['general']['random_state']
    data_text_q = get_txt_from_file(project_root + os.path.sep + train_q)
    data_text_a = get_txt_from_file(project_root + os.path.sep + train_a)
    data_set = list(zip(data_text_q, data_text_a))
    train_set, valid_set = train_test_split(data_set, test_size=valid_size, shuffle=True, random_state=random_state)
    return train_set, valid_set


def load_test_set(arguments):
    # return test set
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    test_q = arguments['tasks'][running_task]['dataset']['test_q']
    test_a = arguments['tasks'][running_task]['dataset']['test_a']
    data_text_q = get_txt_from_file(project_root + os.path.sep + test_q)
    data_text_a = get_txt_from_file(project_root + os.path.sep + test_a)
    data_set = list(zip(data_text_q, data_text_a))
    return data_set


def get_data_iterator(arguments, train_set=None, valid_set=None, test_set=None):
    running_task = arguments['general']['running_task']
    module_obj = sys.modules['src.utilities.load_data']
    use_bert = arguments['tasks'][running_task]['tokenizer']['use_bert']
    batch_size = arguments['training'][running_task]['batch_size']
    batch_size_for_test = arguments['testing'][running_task]['batch_size']
    device = arguments['general']['device']
    # compute the field
    if not use_bert:
        fun_name = arguments['tasks'][running_task]['word_vector']['tokenizer_zh']
        unk_token = arguments['dataset']['general']['unk_token']
        pad_token = arguments['dataset']['general']['pad_token']
        if fun_name == 'tokenize_zh_bySpacy':
            TEXT_FIELD = Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='zh',
                             lower=True, batch_first=True,
                             fix_length=None, init_token=None, eos_token=None, pad_token=pad_token,
                             unk_token=unk_token)
        else:
            TEXT_FIELD = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, fun_name), lower=True, batch_first=True,
                             fix_length=None, init_token=None, eos_token=None, pad_token=pad_token, unk_token=unk_token)
    else:   # use bert
        tokenizer = get_bert_tokenizer(arguments, language='zh')
        configer = get_bert_configer(arguments, language='zh')
        if ('bert' not in arguments['dataset'].keys()) or (arguments['dataset']['bert'] is None):
            arguments['dataset'].update({'bert': {}})
        arguments['dataset']['bert'].update({'sos_token': tokenizer.cls_token,
                                             'eos_token': tokenizer.sep_token,
                                             'unk_token': tokenizer.unk_token,
                                             'pad_token': tokenizer.pad_token,
                                             'sos_idx': tokenizer.cls_token_id,
                                             'eos_idx': tokenizer.sep_token_id,
                                             'unk_idx': tokenizer.unk_token_id,
                                             'pad_idx': tokenizer.pad_token_id
                                              })
        TEXT_FIELD = Field(sequential=True, use_vocab=True, tokenize=tokenizer.tokenize, batch_first=True,
                         fix_length=None, init_token=None, eos_token=None, pad_token=None, unk_token=None)
        task_model = arguments['tasks'][running_task]['model']
        arguments['model'][task_model].update({'d_model': configer.hidden_size,
                                               'nhead': configer.num_attention_heads,
                                               'vocab_len': configer.vocab_size})

    if train_set is not None:
        train_examples = TripletDataExamples_withTorchText(train_set, TEXT_FIELD, TEXT_FIELD, TEXT_FIELD)
        train_iter = BucketIterator(dataset=train_examples, batch_size=batch_size, sort_key=lambda x: len(x.Source),
                                    shuffle=True, sort_within_batch=True, sort=True, device=device)
        arguments['tasks'][running_task]['dataset'].update({'train_set_size': len(train_examples.examples)})
        # build the vocab and vector
        if not use_bert:
            vector_file = arguments['general']['project_root'] + os.path.sep + \
                          arguments['tasks'][running_task]['word_vector']['word_vectors_zh']
            word_vectors = Vectors(name=vector_file)
            TEXT_FIELD.build_vocab(train_examples, vectors=word_vectors)
            arguments['dataset']['general'].update({'unk_idx': TEXT_FIELD.vocab.stoi[unk_token],
                                                    'pad_idx': TEXT_FIELD.vocab.stoi[pad_token]
                                                   })
            used_model = arguments['tasks'][running_task]['model']
            arguments['model'][used_model].update({'vocab_len': TEXT_FIELD.vocab.vectors.shape[0]})
        #build the vocab only
        else:
            init_field_vocab_special_tokens_from_model(TEXT_FIELD, tokenizer)

        # 如果train_set为空，则不管valid_set是否为空，都不做处理，因为valid是针对train的结果而言的。valid可以看作是train的一个过程。
        if valid_set is not None:
            valid_examples = DataExamples_withTorchText(valid_set, TEXT_FIELD, TEXT_FIELD)
            valid_iter = BucketIterator(dataset=valid_examples, batch_size=batch_size,
                                        sort_key=lambda x: len(x.Source), shuffle=True, sort_within_batch=True,
                                        sort=True, train=False, device=device)
    if test_set is not None:
        test_examples = DataExamples_withTorchText(test_set, TEXT_FIELD, TEXT_FIELD)
        test_iter = BucketIterator(dataset=test_examples, batch_size=batch_size_for_test, train=False, shuffle=True,
                                   sort=False, device=device)

    if train_set is not None:
        if valid_set is not None:
            if test_set is not None:
                return train_iter, valid_iter, test_iter, TEXT_FIELD
            else:
                return train_iter, valid_iter, TEXT_FIELD
        else:
            if test_set is not None:
                return train_iter, test_iter, TEXT_FIELD
            else:
                return train_iter, TEXT_FIELD
    else:
        if valid_set is not None:
            if test_set is not None:
                return valid_iter, test_iter, TEXT_FIELD
            else:
                return valid_iter, TEXT_FIELD
        else:
            if test_set is not None:
                return test_iter, TEXT_FIELD
            else:
                return TEXT_FIELD