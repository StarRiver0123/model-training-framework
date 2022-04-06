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
    used_model = config['model_config']['model_name']
    src_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['input_seqs'])[1]
    src_sos_idx = config['symbol_config'][src_transforming_key]['sos_idx']
    src_eos_idx = config['symbol_config'][src_transforming_key]['eos_idx']
    src_vocab_stoi = config['vocab_config'][src_transforming_key]
    src_vocab_itos = config['vocab_config'][src_transforming_key].get_itos()
    tgt_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['ner_labels'])[1]
    tgt_vocab_stoi = config['vocab_config'][tgt_transforming_key]
    tgt_vocab_itos = config['vocab_config'][tgt_transforming_key].get_itos()
    # step 4: create tokenizer
    use_bert = config['model_config']['use_bert']
    if use_bert:
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
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


# this function needs to be defined from the view of concrete task
def default_compute_predict(model, input_seq, device, tokenizer, src_vocab_itos, tgt_vocab_itos, start_pos=0):
    # model, data_example, device, log_string_list are from inner tester framework
    # model input: N,L
    # output: predict:
    source = torch.tensor(input_seq).unsqueeze(0).to(device)
    emission = model.emit(seq_input=source)    # bert的输入需要2维。
    # emission是3维：N,L,D
    predict = model.crf.decode(emission)[0]           # 模型输出是2层list：N，L
    predict = itemgetter(*predict)(tgt_vocab_itos)
    # only for test
    # input_seq = '他张三丰是存放识别出的关键词列表，每个关键词的结构为存放识别出的关键词列表，每个关键词的结构为'
    # predict = ('O', 'B-name', 'I-name', 'E-name', 'B-company', 'O', 'O', 'O', 'S', 'O', 'I-company','O', 'B-name', 'I-name', 'E-name', 'B-company', 'E-name', 'O', 'O', 'O', 'S', 'O')
    if tokenizer == "list":
        input_seq = [src_vocab_itos[id] for id in input_seq[1:-1]]
    else:
        input_seq = tokenizer.convert_ids_to_tokens(input_seq)[1:-1]
    tag_list = []    #存放识别出的关键词列表，每个关键词的结构为：['关键词'，[每个字的tag标签], [字在句子中的位置]，识别结果分类：'confirmed','suspected']
    tag_temp = ['',[],[],'confirming']   # 临时保存识别校验结果，如果完整则存入到tag_list中去。
    previous_key, previous_name, previous_tag = '', '', ''
    for idx, tag in enumerate(predict[1:-1]):  # I和E的前一个必须是B或I，其他字符的前面不能是B或I
        if '-' in tag:
            this_key, this_name = tag.split('-')
        else:
            this_key, this_name = tag, ''
        if '-' in previous_tag:
            previous_key, previous_name = previous_tag.split('-')
        else:
            previous_key, previous_name = previous_tag, ''
        if this_key == 'I' or this_key == 'E':
            if (previous_key == 'B' or previous_key == 'I') and (this_name == previous_name):   #当前tag和前一个tag完美匹配，但需要等待下一个符号时再判断。
                tag_temp[0] += input_seq[idx]
                tag_temp[1] += [tag]
                tag_temp[2] += [start_pos + idx]
            else:                                   #当前tag和前一个tag不匹配，则前一个和当前tag搞不清楚哪个是正常的，则一并归为疑似。
                tag_temp[0] += input_seq[idx]
                tag_temp[1] += [tag]
                tag_temp[2] += [start_pos + idx]
                tag_temp[3] = 'suspected'

        elif this_key == 'B' or this_key == 'S':
            if (previous_key != 'B' and previous_key != 'I'):
                if (previous_key == 'E' or previous_key == 'S'):    #当前tag和前一个tag完美匹配，则前一个入列，开始新一个。
                    if tag_temp[3] == 'confirming':                 #还要判断一下之前是否被记为了疑似。
                        tag_temp[3] = 'confirmed'
                    tag_list.append(tag_temp)
                    tag_temp = ['', [], [], 'confirming']
                tag_temp[0] += input_seq[idx]
                tag_temp[1] += [tag]
                tag_temp[2] += [start_pos + idx]
            else:                                #当前tag和前一个tag不匹配，则前一个和当前tag搞不清楚哪个是正常的，则一并归为疑似。
                tag_temp[0] += input_seq[idx]
                tag_temp[1] += [tag]
                tag_temp[2] += [start_pos + idx]
                tag_temp[3] = 'suspected'
        elif this_key == 'O':
            if (previous_key != 'B' and previous_key != 'I'):
                if (previous_key == 'E' or previous_key == 'S'):    #当前tag和前一个tag完美匹配，则前一个入列，当前符号丢弃，临时tag清空。
                    if tag_temp[3] == 'confirming':                 #还要判断一下之前是否被记为了疑似。
                        tag_temp[3] = 'confirmed'
                    tag_list.append(tag_temp)
                    tag_temp = ['', [], [], 'confirming']
            else:                                #当前tag和前一个tag不匹配，则前一个和当前tag搞不清楚哪个是正常的，则一并归为疑似并入列。临时tag清空。
                tag_temp[0] += input_seq[idx]
                tag_temp[1] += [tag]
                tag_temp[2] += [start_pos + idx]
                tag_temp[3] = 'suspected'
                tag_list.append(tag_temp)
                tag_temp = ['', [], [], 'confirming']
        previous_tag = tag
    key_list = [tag[0] for tag in tag_list if tag[3] == 'confirmed']
    return list(set(key_list)), tag_list


def long_text_predict(model, input_seq, device, tokenizer, max_len, split_overlap_size, src_vocab_itos, tgt_vocab_itos):
    len_seq = len(input_seq)
    if len_seq <= max_len:             #输入的input_seq是带cls和sep的列表
        return default_compute_predict(model, input_seq, device, tokenizer, src_vocab_itos, tgt_vocab_itos)
    max_piece_len = max_len - split_overlap_size
    num_pieces = len_seq // max_piece_len + 1
    key_list = []
    tag_list = []
    for i in range(num_pieces):
        if i == 0:
            seq_piece = input_seq[0: max_piece_len]
            start_pos = 0
        else:
            seq_piece = input_seq[i * max_piece_len - split_overlap_size: min(len_seq, (i + 1) * max_piece_len)]
            start_pos = i * max_piece_len - split_overlap_size
        k_list, t_list= default_compute_predict(model, seq_piece, device, tokenizer, tgt_vocab_itos, start_pos=start_pos)
        key_list += k_list
        tag_list += t_list
        key_list = list(set(key_list))
    return key_list, tag_list