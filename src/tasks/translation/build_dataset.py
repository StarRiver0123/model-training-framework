import sys
from torchtext.legacy.vocab import Vectors
from torchtext.legacy.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
import spacy
spacy_tokenizer = spacy.load("en_core_web_sm")


def load_train_valid_split_set(arguments):
    # return train set, valid set
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    corpus_en = arguments['tasks'][running_task]['dataset']['train_en']
    corpus_zh = arguments['tasks'][running_task]['dataset']['train_zh']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
    valid_size = arguments['training'][running_task]['valid_size']
    random_state = arguments['general']['random_state']
    data_text_en = get_txt_from_file(project_root + os.path.sep + corpus_en)
    data_text_zh = get_txt_from_file(project_root + os.path.sep + corpus_zh)
    if trans_direct == 'en2zh':
        data_set = list(zip(data_text_en, data_text_zh))
    elif trans_direct == 'zh2en':
        data_set = list(zip(data_text_zh, data_text_en))
    else:
        data_set = None
        assert (trans_direct != 'en2zh') or (trans_direct != 'zh2en')
    train_set, valid_set = train_test_split(data_set, test_size=valid_size, shuffle=True, random_state=random_state)
    return train_set, valid_set


def load_test_set(arguments):
    # return test set
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    corpus_en = arguments['tasks'][running_task]['dataset']['test_en']
    corpus_zh = arguments['tasks'][running_task]['dataset']['test_zh']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
    data_text_en = get_txt_from_file(project_root + os.path.sep + corpus_en)
    data_text_zh = get_txt_from_file(project_root + os.path.sep + corpus_zh)
    if trans_direct == 'en2zh':
        data_set = list(zip(data_text_en, data_text_zh))
    elif trans_direct == 'zh2en':
        data_set = list(zip(data_text_zh, data_text_en))
    else:
        data_set = None
        assert (trans_direct != 'en2zh') or (trans_direct != 'zh2en')
    return data_set


def get_data_iterator(arguments, train_set=None, valid_set=None, test_set=None):
    running_task = arguments['general']['running_task']
    module_obj = sys.modules['src.utilities.load_data']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
    assert (trans_direct != 'en2zh') or (trans_direct != 'zh2en')
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
    batch_size = arguments['training'][running_task]['batch_size']
    batch_size_for_test = arguments['testing'][running_task]['batch_size']
    device = arguments['general']['device']
    # compute the field
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        vector_file_en = arguments['general']['project_root'] + os.path.sep + arguments['tasks'][running_task]['word_vector']['word_vectors_en']
        vector_file_zh = arguments['general']['project_root'] + os.path.sep + arguments['tasks'][running_task]['word_vector']['word_vectors_zh']
        fun_name_en = arguments['tasks'][running_task]['word_vector']['tokenizer_en']
        fun_name_zh = arguments['tasks'][running_task]['word_vector']['tokenizer_zh']
        start_token = arguments['dataset']['general']['start_token']
        end_token = arguments['dataset']['general']['end_token']
        unk_token = arguments['dataset']['general']['unk_token']
        pad_token = arguments['dataset']['general']['pad_token']
        if fun_name_en == 'tokenize_en_bySpacy':
            FIELD_en = Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='en_core_web_sm',
                             lower=True, batch_first=True,
                             fix_length=None, init_token=start_token, eos_token=end_token, pad_token=pad_token,
                             unk_token=unk_token)
        else:
            FIELD_en = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, fun_name_en), lower=True, batch_first=True,
                             fix_length=None, init_token=start_token, eos_token=end_token, pad_token=pad_token, unk_token=unk_token)


        FIELD_zh = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, fun_name_zh), batch_first=True,
                             fix_length=None, init_token=start_token, eos_token=end_token, pad_token=pad_token, unk_token=unk_token)
        if trans_direct == 'en2zh':
            SOURCE_FIELD = FIELD_en
            TARGET_FIELD = FIELD_zh
        elif trans_direct == 'zh2en':
            SOURCE_FIELD = FIELD_zh
            TARGET_FIELD = FIELD_en
    else:   # use bert
        tokenizer_en = get_bert_tokenizer(arguments, language='en')
        tokenizer_zh = get_bert_tokenizer(arguments, language='zh')
        configer_en = get_bert_configer(arguments, language='en')
        configer_zh = get_bert_configer(arguments, language='zh')
        if ('bert' not in arguments['dataset'].keys()) or (arguments['dataset']['bert'] is None):
            arguments['dataset'].update({'bert': {}})
        arguments['dataset']['bert'].update({'start_token': tokenizer_en.cls_token,
                                             'end_token': tokenizer_en.sep_token,
                                             'unk_token': tokenizer_en.unk_token,
                                             'pad_token': tokenizer_en.pad_token,
                                             'start_idx': tokenizer_en.cls_token_id,
                                             'end_idx': tokenizer_en.sep_token_id,
                                             'unk_idx': tokenizer_en.unk_token_id,
                                             'pad_idx': tokenizer_en.pad_token_id
                                              })
        FIELD_en = Field(sequential=True, use_vocab=True, tokenize=tokenizer_en.tokenize, batch_first=True,
                         fix_length=None, init_token=None, eos_token=None, pad_token=None, unk_token=None)
        FIELD_zh = Field(sequential=True, use_vocab=True, tokenize=tokenizer_zh.tokenize, batch_first=True,
                         fix_length=None, init_token=None, eos_token=None, pad_token=None, unk_token=None)
        if trans_direct == 'en2zh':
            SOURCE_FIELD = FIELD_en
            TARGET_FIELD = FIELD_zh
            tokenizer_src = tokenizer_en
            tokenizer_tgt = tokenizer_zh
            configer_src = configer_en
            configer_tgt = configer_zh
        elif trans_direct == 'zh2en':
            SOURCE_FIELD = FIELD_zh
            TARGET_FIELD = FIELD_en
            tokenizer_src = tokenizer_zh
            tokenizer_tgt = tokenizer_en
            configer_src = configer_zh
            configer_tgt = configer_en

    if train_set is not None:
        train_examples = DataExamples_withTorchText(train_set, SOURCE_FIELD, TARGET_FIELD)
        train_iter = BucketIterator(dataset=train_examples, batch_size=batch_size, sort_key=lambda x: len(x.Source),
                                    shuffle=True, sort_within_batch=True, sort=True, device=device)
        # build the vocab and vector
        if (use_bert != 'static') and (use_bert != 'dynamic'):
            if trans_direct == 'en2zh':
                vectors_src = Vectors(name=vector_file_en)
                vectors_tgt = Vectors(name=vector_file_zh)
            elif trans_direct == 'zh2en':
                vectors_src = Vectors(name=vector_file_zh)
                vectors_tgt = Vectors(name=vector_file_en)
            SOURCE_FIELD.build_vocab(train_examples, vectors=vectors_src)
            TARGET_FIELD.build_vocab(train_examples, vectors=vectors_tgt)
            arguments['dataset']['general'].update({'start_idx': SOURCE_FIELD.vocab.stoi[start_token],
                                                  'end_idx': SOURCE_FIELD.vocab.stoi[end_token],
                                                  'unk_idx': SOURCE_FIELD.vocab.stoi[unk_token],
                                                  'pad_idx': SOURCE_FIELD.vocab.stoi[pad_token]
                                                  })
        # build the vocab only
        else:
            init_field_vocab_special_tokens_from_model(SOURCE_FIELD, tokenizer_src)
            init_field_vocab_special_tokens_from_model(TARGET_FIELD, tokenizer_tgt)
            task_model = arguments['tasks'][running_task]['model']
            arguments['model'][task_model].update({'d_model': configer_src.hidden_size,
                                                   'nhead': configer_src.num_attention_heads})

        # 如果train_set为空，则不管valid_set是否为空，都不做处理，因为valid是针对train的结果而言的。valid可以看作是train的一个过程。
        if valid_set is not None:
            valid_examples = DataExamples_withTorchText(valid_set, SOURCE_FIELD, TARGET_FIELD)
            valid_iter = BucketIterator(dataset=valid_examples, batch_size=batch_size,
                                        sort_key=lambda x: len(x.Source), shuffle=True, sort_within_batch=True,
                                        sort=True, train=False, device=device)
    if test_set is not None:
        test_examples = DataExamples_withTorchText(test_set, SOURCE_FIELD, TARGET_FIELD)
        test_iter = BucketIterator(dataset=test_examples, batch_size=batch_size_for_test, train=False,
                                   sort=False, device=device)

    if train_set is not None:
        if valid_set is not None:
            if test_set is not None:
                return train_iter, valid_iter, test_iter, SOURCE_FIELD, TARGET_FIELD
            else:
                return train_iter, valid_iter, SOURCE_FIELD, TARGET_FIELD
        else:
            if test_set is not None:
                return train_iter, test_iter, SOURCE_FIELD, TARGET_FIELD
            else:
                return train_iter, SOURCE_FIELD, TARGET_FIELD
    else:
        if valid_set is not None:
            if test_set is not None:
                return valid_iter, test_iter, SOURCE_FIELD, TARGET_FIELD
            else:
                return valid_iter, SOURCE_FIELD, TARGET_FIELD
        else:
            if test_set is not None:
                return test_iter, SOURCE_FIELD, TARGET_FIELD
            else:
                return SOURCE_FIELD, TARGET_FIELD