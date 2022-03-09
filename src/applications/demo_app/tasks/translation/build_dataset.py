import sys,time
from torchtext.legacy.vocab import Vectors
from torchtext.legacy.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
from src.modules.tokenizers.tokenizer import *
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'

def load_train_valid_split_set(arguments):
    # return train set, valid set
    dataset_root = arguments['dataset_root']
    corpus_en_file = arguments['net_structure']['dataset']['train_en_file']
    corpus_zh_file = arguments['net_structure']['dataset']['train_zh_file']
    trans_direct = arguments['net_structure']['trans_direct']
    valid_size = arguments['training']['valid_size']
    random_state = arguments['general']['random_state']
    data_text_en = get_txt_from_file(dataset_root + os.path.sep + corpus_en_file)
    data_text_zh = get_txt_from_file(dataset_root + os.path.sep + corpus_zh_file)
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
    dataset_root = arguments['dataset_root']
    corpus_en_file = arguments['net_structure']['dataset']['test_en_file']
    corpus_zh_file = arguments['net_structure']['dataset']['test_zh_file']
    trans_direct = arguments['net_structure']['trans_direct']
    data_text_en = get_txt_from_file(dataset_root + os.path.sep + corpus_en_file)
    data_text_zh = get_txt_from_file(dataset_root + os.path.sep + corpus_zh_file)
    if trans_direct == 'en2zh':
        data_set = list(zip(data_text_en, data_text_zh))
    elif trans_direct == 'zh2en':
        data_set = list(zip(data_text_zh, data_text_en))
    else:
        data_set = None
        assert (trans_direct != 'en2zh') or (trans_direct != 'zh2en')
    return data_set


def get_data_iterator(arguments, train_set=None, valid_set=None, test_set=None):
    module_obj = sys.modules[tokenizer_package_path]
    trans_direct = arguments['net_structure']['trans_direct']
    assert (trans_direct != 'en2zh') or (trans_direct != 'zh2en')
    use_bert = arguments['net_structure']['use_bert']
    batch_size = arguments['training']['batch_size']
    batch_size_for_test = arguments['testing']['batch_size']
    device = arguments['general']['device']
    # compute the field
    if use_bert not in ['static', 'dynamic']:
        vector_file_en = arguments['word_vector_root'] + os.path.sep + arguments['net_structure']['word_vector']['word_vectors_en_file']
        vector_file_zh = arguments['word_vector_root'] + os.path.sep + arguments['net_structure']['word_vector']['word_vectors_zh_file']
        fun_name_en = arguments['net_structure']['tokenizer']['tokenizer_en']
        fun_name_zh = arguments['net_structure']['tokenizer']['tokenizer_zh']
        sos_token = arguments['dataset']['symbol']['sos_token']
        eos_token = arguments['dataset']['symbol']['eos_token']
        unk_token = arguments['dataset']['symbol']['unk_token']
        pad_token = arguments['dataset']['symbol']['pad_token']
        if fun_name_en == 'tokenize_en_bySpacy':
            FIELD_en = Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='en',
                             lower=True, batch_first=True,
                             fix_length=None, init_token=sos_token, eos_token=eos_token, pad_token=pad_token,
                             unk_token=unk_token)
        else:
            FIELD_en = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, fun_name_en), lower=True, batch_first=True,
                             fix_length=None, init_token=sos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token)


        FIELD_zh = Field(sequential=True, use_vocab=True, tokenize=getattr(module_obj, fun_name_zh), batch_first=True,
                             fix_length=None, init_token=sos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token)
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
        if ('symbol' not in arguments['dataset'].keys()) or (arguments['dataset']['symbol'] is None):
            arguments['dataset'].update({'symbol': {}})
        arguments['dataset']['symbol'].update({'sos_token': tokenizer_en.cls_token,
                                             'eos_token': tokenizer_en.sep_token,
                                             'unk_token': tokenizer_en.unk_token,
                                             'pad_token': tokenizer_en.pad_token,
                                             'sos_idx': tokenizer_en.cls_token_id,
                                             'eos_idx': tokenizer_en.sep_token_id,
                                             'unk_idx': tokenizer_en.unk_token_id,
                                             'pad_idx': tokenizer_en.pad_token_id
                                              })

        # if use_bert == 'static':
        #     FIELD_en = Field(sequential=True, use_vocab=True, tokenize=tokenizer_en.tokenize, batch_first=True,
        #                      fix_length=None, init_token=None, eos_token=None, pad_token=None, unk_token=None)
        #     FIELD_zh = Field(sequential=True, use_vocab=True, tokenize=tokenizer_zh.tokenize, batch_first=True,
        #                      fix_length=None, init_token=None, eos_token=None, pad_token=None, unk_token=None)
        # else:    #这个还需要修改
        # 如果不需要field维护词典的话，可以这么写，直接用bert模型的分词和数字化。
        # FIELD_en = Field(sequential=True, use_vocab=False, tokenize=tokenizer_en.tokenize, preprocessing=tokenizer_en.convert_tokens_to_ids, batch_first=True,
        #                  fix_length=None, init_token=tokenizer_en.cls_token_id, eos_token=tokenizer_en.eos_token_id, pad_token=tokenizer_en.pad_token_id, unk_token=tokenizer_en.unk_token_id)
        # FIELD_zh = Field(sequential=True, use_vocab=False, tokenize=tokenizer_zh.tokenize, preprocessing=tokenizer_zh.convert_tokens_to_ids, batch_first=True,
        #                  fix_length=None, init_token=tokenizer_zh.cls_token_id, eos_token=tokenizer_zh.eos_token_id, pad_token=tokenizer_zh.pad_token_id, unk_token=tokenizer_zh.unk_token_id)
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
        task_model = arguments['net_structure']['model']
        arguments['model'][task_model].update({'d_model': configer_src.hidden_size,
                                               'nhead': configer_src.num_attention_heads,
                                               'src_vocab_len': configer_src.vocab_size,
                                               'tgt_vocab_len': configer_tgt.vocab_size})

    if train_set is not None:
        train_examples = getDataExamples_withTorchText(train_set, SOURCE_FIELD, TARGET_FIELD, num_workers=-1)
        train_iter = BucketIterator(dataset=train_examples, batch_size=batch_size, sort_key=lambda x: len(x.Source),
                                    shuffle=True, sort_within_batch=True, sort=True, device=device)
        arguments['net_structure']['dataset'].update({'train_set_size': len(train_examples.examples)})
        # build the vocab and vector
        if use_bert not in ['static', 'dynamic']:
            if trans_direct == 'en2zh':
                vectors_src = Vectors(name=vector_file_en)
                vectors_tgt = Vectors(name=vector_file_zh)
            elif trans_direct == 'zh2en':
                vectors_src = Vectors(name=vector_file_zh)
                vectors_tgt = Vectors(name=vector_file_en)
            SOURCE_FIELD.build_vocab(train_examples, vectors=vectors_src)
            TARGET_FIELD.build_vocab(train_examples, vectors=vectors_tgt)
            arguments['dataset']['symbol'].update({'sos_idx': SOURCE_FIELD.vocab.stoi[sos_token],
                                                  'eos_idx': SOURCE_FIELD.vocab.stoi[eos_token],
                                                  'unk_idx': SOURCE_FIELD.vocab.stoi[unk_token],
                                                  'pad_idx': SOURCE_FIELD.vocab.stoi[pad_token]
                                                  })
            used_model = arguments['net_structure']['model']
            arguments['model'][used_model].update({'src_vocab_len': SOURCE_FIELD.vocab.vectors.shape[0],
                                                   'tgt_vocab_len': TARGET_FIELD.vocab.vectors.shape[0]})
        #build the vocab only
        else:
            init_field_vocab_special_tokens_from_model(SOURCE_FIELD, tokenizer_src)
            init_field_vocab_special_tokens_from_model(TARGET_FIELD, tokenizer_tgt)


        # 如果train_set为空，则不管valid_set是否为空，都不做处理，因为valid是针对train的结果而言的。valid可以看作是train的一个过程。
        if valid_set is not None:
            valid_examples = getDataExamples_withTorchText(valid_set, SOURCE_FIELD, TARGET_FIELD)
            valid_iter = BucketIterator(dataset=valid_examples, batch_size=batch_size,
                                        sort_key=lambda x: len(x.Source), shuffle=True, sort_within_batch=True,
                                        sort=True, train=False, device=device)
    if test_set is not None:
        test_examples = getDataExamples_withTorchText(test_set, SOURCE_FIELD, TARGET_FIELD)
        test_iter = BucketIterator(dataset=test_examples, batch_size=batch_size_for_test, shuffle=True, train=False,
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