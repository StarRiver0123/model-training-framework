import sys
import torch
from operator import itemgetter
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from src.utilities.load_data import *
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.models.tester.tester_framework import Tester
from src.tasks.ner.test_model import manage_model_state


def apply_model(arguments):
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    used_model = arguments['tasks'][running_task]['model']
    max_len = arguments['model'][used_model]['max_len']
    split_overlap_size = arguments['model'][used_model]['split_overlap_size']
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_vocab, _, _, _ = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    tgt_vocab_stoi = model_vocab['tgt_vocab_stoi']
    tgt_vocab_itos = model_vocab['tgt_vocab_itos']

    bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_zh']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    while 1:
        input_sentence = input("请输入一句话（输入字母'q'可退出）：")
        if input_sentence == 'q':
            break
        print('\n')
        # input_seq = torch.tensor(tokenizer.convert_tokens_to_ids([word for word in input_sentence.strip()])).unsqueeze(0)  # bert的输入需要2维。
        # start to run
        tester.apply(model=model, input_seq=input_sentence, compute_predict_func=compute_predict,
                      compute_predict_outer_params={'tokenizer': tokenizer, 'tgt_vocab_itos': tgt_vocab_itos, 'max_len': max_len, 'split_overlap_size': split_overlap_size})


# this function needs to be defined from the view of concrete task
def compute_predict(model, input_seq, device, log_string_list, tokenizer, tgt_vocab_itos, max_len, split_overlap_size):
    # model, data_example, device, log_string_list are from inner tester framework
    # model input: N,L
    # output: predict:
    # source = torch.tensor(tokenizer.encode(input_seq)[1:-1]).unsqueeze(0).to(device)
    source = tokenizer.encode(input_seq)[1:-1]
    key_list, tag_list = long_text_predict(model, source, device, tokenizer, max_len, split_overlap_size, tgt_vocab_itos)
    found_out = str(key_list)
    confirmed = str([tag for tag in tag_list if tag[3] == 'confirmed']) if len(tag_list) > 0 else ''
    suspected = str([tag for tag in tag_list if tag[3] == 'suspected']) if len(tag_list) > 0 else ''
    log_string_list.append("原文输入: " + input_seq)
    log_string_list.append("发现目标: \n" + found_out + '\n')
    log_string_list.append("确认结果: \n" + confirmed + '\n')
    log_string_list.append("疑似结果: \n" + suspected + '\n')
    return None


# this function needs to be defined from the view of concrete task
def default_compute_predict(model, input_seq, device, tokenizer, tgt_vocab_itos, start_pos=0):
    # model, data_example, device, log_string_list are from inner tester framework
    # model input: N,L
    # output: predict:
    source = torch.tensor(input_seq).unsqueeze(0).to(device)
    emission = model.model.bert_emit(seq_input=source)    # bert的输入需要2维。
    # emission是3维：N,L,D
    predict = model.model.crf.decode(emission)[0]           # 模型输出是2层list
    predict = itemgetter(*predict)(tgt_vocab_itos)
    # only for test
    # input_seq = '他张三丰是存放识别出的关键词列表，每个关键词的结构为存放识别出的关键词列表，每个关键词的结构为'
    # predict = ('O', 'B-name', 'I-name', 'E-name', 'B-company', 'O', 'O', 'O', 'S', 'O', 'I-company','O', 'B-name', 'I-name', 'E-name', 'B-company', 'E-name', 'O', 'O', 'O', 'S', 'O')
    input_seq = tokenizer.convert_ids_to_tokens(input_seq)[1:-1]
    tag_list = []    #存放识别出的关键词列表，每个关键词的结构为：['关键词'，[每个字的tag标签], [字在句子中的位置]，识别结果分类：'confirmed','suspected']
    tag_temp = ['',[],[],'confirming']   # 临时保存识别校验结果，如果完整则存入到tag_list中去。
    previous_key, previous_name, previous_tag = '', '', ''
    for idx, tag in enumerate(predict):  # I和E的前一个必须是B或I，其他字符的前面不能是B或I
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


def long_text_predict(model, input_seq, device, tokenizer, max_len, split_overlap_size, tgt_vocab_itos):
    len_seq = len(input_seq)
    if len_seq <= max_len - 2:             #输入的input_seq是不带cls和sep的列表
        input_seq = [101] + input_seq + [102]
        return default_compute_predict(model, input_seq, device, tokenizer, tgt_vocab_itos)
    max_piece_len = max_len - split_overlap_size - 2
    num_pieces = len_seq // max_piece_len + 1
    key_list = []
    tag_list = []
    for i in range(num_pieces):
        if i == 0:
            seq_piece = [101] + input_seq[0: max_piece_len] + [102]
            start_pos = 0
        else:
            seq_piece = [101] + input_seq[i * max_piece_len - split_overlap_size : min(len_seq, (i+1)*max_piece_len)] + [102]
            start_pos = i * max_piece_len - split_overlap_size
        k_list, t_list= default_compute_predict(model, seq_piece, device, tokenizer, tgt_vocab_itos, start_pos=start_pos)
        key_list += k_list
        tag_list += t_list
        key_list = list(set(key_list))
    return key_list, tag_list