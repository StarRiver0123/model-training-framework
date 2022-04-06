import sys
import torch
import torch.nn.functional as F
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.modules.tester.tester_framework import Tester
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_model import load_model_states_into_config, greedy_decode
from build_model import create_inference_model, create_evaluator
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'

def apply_model(config):
    # step 1: load model and config states
    load_model_states_into_config(config)
    # step 2: create model
    model = create_inference_model(config)
    # step 3: load vocab
    used_model = config['model_config']['model_name']
    src_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['source_seqs'])[1]
    src_vocab_stoi = config['vocab_config'][src_transforming_key]
    src_vocab_itos = config['vocab_config'][src_transforming_key].get_itos()
    tgt_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['target_seqs'])[1]
    tgt_vocab_stoi = config['vocab_config'][tgt_transforming_key]
    tgt_vocab_itos = config['vocab_config'][tgt_transforming_key].get_itos()
    sos_idx = config['symbol_config'][tgt_transforming_key]['sos_idx']
    eos_idx = config['symbol_config'][tgt_transforming_key]['eos_idx']

    # step 4: create tokenizer
    tokenizer_name = 'tokenize_en_byJieba'
    module_obj = sys.modules[tokenizer_package_path]
    tokenizer = getattr(module_obj, tokenizer_name)
    # step 5: start main process
    max_len = config['model_config']['max_len']
    while 1:
        # step 5-1: get the input
        input_sentence = input("请输入一句英语(input a single 'q' to quit)：")
        if input_sentence == 'q':
            break
        print('\n')
        # step 5-2: get the token ids
        seqs = [sos_idx] + [src_vocab_stoi[word] for word in tokenizer(input_sentence)] + [eos_idx]
        # step 5-3: adjust the input shape
        input_seq = torch.tensor(seqs).unsqueeze(0)         # bert的输入需要2维。
        # step 5-4: start to inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            source = input_seq.to(device)
            sos_idx = source[0, 0].item()
            eos_idx = source[0, -1].item()
            predict = greedy_decode(model, source, device, max_len, sos_idx, eos_idx)
            print("识别出原文: " + ' '.join(src_vocab_itos[index] for index in source[0, 1:-1]))
            print("翻译出译文: " + ' '.join(tgt_vocab_itos[index] for index in predict[0, 1:]) + '\n`')
