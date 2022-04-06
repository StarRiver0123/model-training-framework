import sys
import torch
from operator import itemgetter
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.modules.tester.tester_framework import Tester
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_model import load_model_states_into_config
from build_model import create_inference_model, create_evaluator
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'


def apply_model(config):
    # step 1: load model and config states
    load_model_states_into_config(config)
    # step 2: create model
    model = create_inference_model(config)
    # step 3: load vocab
    use_bert = config['model_config']['use_bert']
    if not use_bert:
        src_vocab_stoi = config['model_vocab']['src_vocab_stoi']
        src_vocab_itos = config['model_vocab']['src_vocab_itos']
    else:
        src_vocab_stoi = None
        src_vocab_itos = None
    tgt_vocab_stoi = config['model_vocab']['tgt_vocab_stoi']
    tgt_vocab_itos = config['model_vocab']['tgt_vocab_itos']
    src_sos_idx = config['symbol_config']['src_sos_idx']
    src_eos_idx = config['symbol_config']['src_eos_idx']
    # step 4: create tokenizer
    if use_bert:
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_file)
    else:
        tokenizer = "list"
    # step 5: start main process
    while 1:
        # step 5-1: get the input
        input_sentence = input("请输入一句话（输入字母'q'可退出）：")
        if input_sentence == 'q':
            break
        print('\n')
        # step 5-2: get the token ids
        if not use_bert:
            source = [src_sos_idx] + [src_vocab_stoi[word] for word in list(input_sentence)] + [src_eos_idx]
        else:
            source = tokenizer.encode(input_sentence)
        # step 5-3: adjust the input shape
        # step 5-4: start to inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        max_len = config['model_config']['max_len']
        split_overlap_size = config['model_config']['split_overlap_size']
        model.eval()
        with torch.no_grad():
            key_list, tag_list = long_text_predict(model, source, device, tokenizer, max_len, split_overlap_size,
                                                   src_vocab_itos, tgt_vocab_itos)
            found_out = str(key_list)
            confirmed = str([tag for tag in tag_list if tag[3] == 'confirmed']) if len(tag_list) > 0 else ''
            suspected = str([tag for tag in tag_list if tag[3] == 'suspected']) if len(tag_list) > 0 else ''
            print("原文输入: " + input_sentence)
            print("发现目标: \n" + found_out + '\n')
            print("确认结果: \n" + confirmed + '\n')
            print("疑似结果: \n" + suspected + '\n')
