import sys, os
import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import *
from torchtext.data.utils import get_tokenizer
from functools import partial
from src.utilities.load_data import *
from src.modules.tokenizers.tokenizer import *
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'


def clean_data(batch):
    pass


def collate_fn(config, text_transforming_adaptor, batch):
    reshaped_batch = list(zip(*batch))
    batch_list = []
    for text_field_idx, transforming_key in text_transforming_adaptor:
        if transforming_key is None or transforming_key == '':
            batch_text = torch.tensor(reshaped_batch[text_field_idx])
        else:
            transforming_pipeline = create_transforming_pipeline(config, transforming_key)
            batch_text = list(map(transforming_pipeline, reshaped_batch[text_field_idx]))
            use_bert_style = config['text_transforming'][transforming_key]['use_bert_style']
            try:
                use_padding = config['text_transforming'][transforming_key]['use_padding']
            except KeyError:
                use_padding = False
            if use_bert_style or use_padding:
                pad_idx = config['symbol_config'][transforming_key]['pad_idx']
                batch_text = pad_sequence(batch_text, batch_first=True, padding_value=pad_idx)
        batch_list.append(batch_text)
    return batch_list


def create_transforming_pipeline(config, transforming_key):
    if transforming_key is None or transforming_key == '':
        return None
    use_tokenizing = config['text_transforming'][transforming_key]['use_tokenizing']
    use_numericalizing = config['text_transforming'][transforming_key]['use_numericalizing']
    try:
        use_start_end_symbol = config['text_transforming'][transforming_key]['use_start_end_symbol']
    except KeyError:
        use_start_end_symbol = False
    use_bert_style = config['text_transforming'][transforming_key]['use_bert_style']
    pipeline = []
    if use_tokenizing:
        pipeline.append(config['tokenizer_config'][transforming_key])
    if use_numericalizing:
        pipeline.append(config['vocab_config'][transforming_key])
    if use_bert_style or use_start_end_symbol:
        sos_idx = config['symbol_config'][transforming_key]['sos_idx']
        eos_idx = config['symbol_config'][transforming_key]['eos_idx']
        add_start_end_symbol = partial(_add_start_end_symbol, sos_idx, eos_idx)
        pipeline.append(add_start_end_symbol)
    else:
        pipeline.append(to_tensor)
    if len(pipeline) > 0:
        return sequential_transforms(*pipeline)
    return None


def _add_start_end_symbol(sos_idx, eos_idx, batch):
    return torch.cat((torch.tensor([sos_idx]),
                     torch.tensor(batch),
                     torch.tensor([eos_idx])))


def to_tensor(batch):
    return torch.tensor(batch)


def sequential_transforms(*transforms):
    def func(text_seq):
        for trans in transforms:
            text_seq = trans(text_seq)
        return text_seq
    return func


def build_tokenizer_into_config(config, train_text_transforming_adaptor):
    if ('tokenizer_config' not in config.keys()) or (config['tokenizer_config'] is None):
        config.update({'tokenizer_config': {}})
    for text_field_idx, transforming_key in train_text_transforming_adaptor:
        if transforming_key is None or transforming_key == '':
            continue
        use_tokenizing = config['text_transforming'][transforming_key]['use_tokenizing']
        if not use_tokenizing:
            continue
        try:
            tokenizer = config['text_transforming'][transforming_key]['tokenizer']
        except KeyError:
            tokenizer = None
        if tokenizer == '':
            tokenizer = None
        try:
            language = config['text_transforming'][transforming_key]['language']
        except KeyError:
            language = 'zh'
        if tokenizer == 'bert':
            bert_model_root = config['bert_model_root']
            bert_model_file = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            tokenizer = BertTokenizer.from_pretrained(bert_model_file)
        elif tokenizer not in [None, 'spacy', 'moses', 'toktok', 'revtok', 'subword', 'basic_english']:
            module_obj = sys.modules[tokenizer_package_path]
            tokenizer = getattr(module_obj, tokenizer)
        tokenizer = get_tokenizer(tokenizer, language)
        config['tokenizer_config'][transforming_key] = tokenizer


def build_vocab_into_config(config, train_text_transforming_adaptor, vocab_corpus=None):
    def yield_tokens(vocab_corpus, tokenizer, vocab_corpus_field_idx):
        if tokenizer is not None:
            if isinstance(vocab_corpus_field_idx, int):
                for dataline in tqdm(vocab_corpus):
                    yield tokenizer(dataline[vocab_corpus_field_idx])
            elif isinstance(vocab_corpus_field_idx, list):
                for dataline in tqdm(vocab_corpus):
                    yield tokenizer(' '.join([dataline[i] for i in vocab_corpus_field_idx]))
        else:
            if isinstance(vocab_corpus_field_idx, int):
                for dataline in tqdm(vocab_corpus):
                    yield dataline[vocab_corpus_field_idx]
            elif isinstance(vocab_corpus_field_idx, list):
                for dataline in tqdm(vocab_corpus):
                    yield ' '.join([dataline[i] for i in vocab_corpus_field_idx])
    if ('vocab_config' not in config.keys()) or (config['vocab_config'] is None):     # 用于保存模型参数和测试部署阶段使用
        config.update({'vocab_config': {}})
    if ('model_config' not in config.keys()) or (config['model_config'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'model_config': {'vocab_len': {}}})
    for text_field_idx, transforming_key in train_text_transforming_adaptor:
        if transforming_key is None or transforming_key == '':
            continue
        use_numericalizing = config['text_transforming'][transforming_key]['use_numericalizing']
        if not use_numericalizing:
            continue
        use_bert_style = config['text_transforming'][transforming_key]['use_bert_style']
        if use_bert_style:
            use_tokenizing = config['text_transforming'][transforming_key]['use_tokenizing']
            if use_tokenizing and config['text_transforming'][transforming_key]['tokenizer'] == 'bert':
                tokenizer = config['tokenizer_config'][transforming_key]
            else:
                bert_model_root = config['bert_model_root']
                bert_model_file = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
                tokenizer = BertTokenizer.from_pretrained(bert_model_file)
            vocab = tokenizer.convert_tokens_to_ids
        else:
            use_padding = config['text_transforming'][transforming_key]['use_padding']
            use_start_end_symbol = config['text_transforming'][transforming_key]['use_start_end_symbol']
            unk_token = config['dataset']['general_symbol']['unk_token']
            special_tokens = [unk_token]
            if use_padding:
                special_tokens.append(config['dataset']['general_symbol']['pad_token'])
            if use_start_end_symbol:
                special_tokens.append(config['dataset']['general_symbol']['sos_token'])
                special_tokens.append(config['dataset']['general_symbol']['eos_token'])
            vocab_corpus_field_idx = config['text_transforming'][transforming_key]['corpus_field_index_for_vocab_building']
            if isinstance(vocab_corpus_field_idx, str):
                vocab_corpus_field_idx = eval(vocab_corpus_field_idx)
            try:
                tokenizer = config['tokenizer_config'][transforming_key]
            except KeyError:
                tokenizer = None
            vocab = build_vocab_from_iterator(yield_tokens(vocab_corpus, tokenizer, vocab_corpus_field_idx), specials=special_tokens, special_first=True)
            vocab.set_default_index(vocab[unk_token])
            config['model_config']['vocab_len'][transforming_key] = len(vocab)
        config['vocab_config'][transforming_key] = vocab


def build_vectors_from_pretrained_into_config(config, train_text_transforming_adaptor):
    # 预定义词向量不能直接使用，需要根据词典进行选择。
    if ('vector_config' not in config.keys()) or (config['vector_config'] is None):     # 用于保存模型参数和测试部署阶段使用
        config.update({'vector_config': {}})
    for text_field_idx, transforming_key in train_text_transforming_adaptor:
        if transforming_key is None or transforming_key == '':
            continue
        use_bert_style = config['text_transforming'][transforming_key]['use_bert_style']
        if use_bert_style:
            continue
        use_pretrained_word_vector = config['text_transforming'][transforming_key]['use_pretrained_word_vector']
        if not use_pretrained_word_vector:
            continue
        vector_file = config['word_vector_root'] + os.path.sep + config['text_transforming'][transforming_key]['pretrained_word_vector_file']
        word_vectors = Vectors(name=vector_file)
        vocab_tokens = list(config['vocab_config'][transforming_key].get_stoi().keys())
        config['vector_config'][transforming_key] = word_vectors.get_vecs_by_tokens(vocab_tokens)


def build_special_tokens_into_config(config, train_text_transforming_adaptor):
    if ('symbol_config' not in config.keys()) or (config['symbol_config'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'symbol_config': {}})
    for text_field_idx, transforming_key in train_text_transforming_adaptor:
        if transforming_key is None or transforming_key == '':
            continue
        try:
            use_padding = config['text_transforming'][transforming_key]['use_padding']
        except KeyError:
            use_padding = False
        try:
            use_start_end_symbol = config['text_transforming'][transforming_key]['use_start_end_symbol']
        except KeyError:
            use_start_end_symbol = False
        use_bert_style = config['text_transforming'][transforming_key]['use_bert_style']
        if use_bert_style:   # use bert
            bert_model_root = config['bert_model_root']
            bert_model_file = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            tokenizer = BertTokenizer.from_pretrained(bert_model_file)
            config['symbol_config'][transforming_key] = {'unk_token': tokenizer.unk_token,
                                                         'unk_idx': tokenizer.unk_token_id
                                                          }
            if use_padding:
                config['symbol_config'][transforming_key] = {'pad_token': tokenizer.pad_token,
                                                             'pad_idx': tokenizer.pad_token_id
                                                             }
            if use_start_end_symbol:
                config['symbol_config'][transforming_key] = {'sos_token': tokenizer.cls_token,
                                                             'eos_token': tokenizer.sep_token,
                                                             'sos_idx': tokenizer.cls_token_id,
                                                             'eos_idx': tokenizer.sep_token_id
                                                             }
        else:
            unk_token = config['dataset']['general_symbol']['unk_token']
            vocab = config['vocab_config'][transforming_key]
            config['symbol_config'][transforming_key] = {'unk_token': unk_token,
                                                         'unk_idx': vocab.get_stoi()[unk_token]
                                                         }
            if use_padding:
                pad_token = config['dataset']['general_symbol']['pad_token']
                config['symbol_config'][transforming_key] = {'pad_token': pad_token,
                                                             'pad_idx': vocab.get_stoi()[pad_token]
                                                             }
            if use_start_end_symbol:
                sos_token = config['dataset']['general_symbol']['sos_token']
                eos_token = config['dataset']['general_symbol']['eos_token']
                config['symbol_config'][transforming_key] = {'sos_token': sos_token,
                                                             'eos_token': eos_token,
                                                             'sos_idx': vocab.get_stoi()[sos_token],
                                                             'eos_idx': vocab.get_stoi()[eos_token]
                                                             }