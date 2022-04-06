import torch.nn.functional as F
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.modules.trainer.trainer_framework import Trainer
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import *
from build_model import TrainingModel


def train_model(config):
    # step 1: build dataset and vocab
    # 注意这里需要优化，数据量很大时，train_iter消耗了太多地内存资源。需要加上无用资源释放代码!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_iter, valid_iter = build_train_dataset_and_vocab_pipeline(config)
    # step 2: build model
    used_model = config['net_structure']['model']
    src_transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['source_seqs'])[1]
    tgt_transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['target_seqs'])[1]
    src_vectors = config['vector_config'][src_transforming_key]
    tgt_vectors = config['vector_config'][tgt_transforming_key]
    src_vocab = config['vocab_config'][src_transforming_key]
    tgt_vocab = config['vocab_config'][tgt_transforming_key]
    model = TrainingModel(config, src_vectors, tgt_vectors)
    # step 3: get the trainer
    trainer = Trainer(config)
    # step 4: start train
    tgt_pad_idx = config['symbol_config'][tgt_transforming_key]['pad_idx']
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={
                      'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab, 'pad_idx': tgt_pad_idx},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={
                      'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab, 'pad_idx': tgt_pad_idx}, save_model_state_func=save_model_state_func,
                  save_model_state_outer_params={})



# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, src_vocab, tgt_vocab, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if data_example[0].size(1) > max_len:
        source = data_example[0][:, :max_len].to(device)
    else:
        source = data_example[0].to(device)
    if data_example[1].size(1) > max_len:
        target = data_example[1][:, :max_len].to(device)
    else:
        target = data_example[1].to(device)
    # 要注意decoder的input和target要错一位。！！！！！！！！很重要
    target_input = target[:, :-1]
    target_real = target[:, 1:]
    # src_mask = gen_full_false_mask(source, source)  # L,L
    tgt_mask = gen_seq_only_mask(target_input, target_input)  # L,L
    src_key_padding_mask = gen_pad_only_mask(source, pad_idx)   #N,L
    tgt_key_padding_mask = gen_pad_only_mask(target_input, pad_idx)  #N,L
    memory_key_padding_mask = src_key_padding_mask
    # logit = model.model(enc_input=source, dec_input=target_input, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    logit = model.model(enc_input=source, dec_input=target_input, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    # 模型的输出维度是N，L，D_target_vocab_len,
    # pytorch CrossEntropyLoss的输入维度有两种方式：
    # （1） input为N，C；target为N，需要对predict做reshape（-1，D_target_vocab_len）
    # （2） input为N，C，L，target为N，L。要把分类放在第二维，需要对predict进行转置transpose(-1,-2)
    logit_flatten = logit.reshape(-1, logit.size(-1))
    target_real_flatten = target_real.reshape(-1)
    loss = model.criterion(logit_flatten, target_real_flatten)
    if do_log:
        log_string_list.append(
            "Source string:  " + ' '.join(src_vocab.get_itos()[index] for index in source[0, 1:-1]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:-1]))
        log_string_list.append(
            "Target string:  " + ' '.join(tgt_vocab.get_itos()[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:-1]))
        log_string_list.append("Predict string: " + ' '.join(
            tgt_vocab.get_itos()[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(
            str(index.item()) for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)) + '\n')
    return logit, target_real, loss

# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, src_vocab, tgt_vocab, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if data_example[0].size(1) > max_len:
        source = data_example[0][:, :max_len].to(device)
    else:
        source = data_example[0].to(device)
    if data_example[1].size(1) > max_len:
        target = data_example[1][:, :max_len].to(device)
    else:
        target = data_example[1].to(device)

    # 要注意decoder的input和target要错一位。！！！！！！！！很重要
    target_input = target[:, :-1]
    target_real = target[:, 1:]
    # src_mask = gen_full_false_mask(source, source)  # L,L
    tgt_mask = gen_seq_only_mask(target_input, target_input)  # L,L
    src_key_padding_mask = gen_pad_only_mask(source, pad_idx)   #N,L
    tgt_key_padding_mask = gen_pad_only_mask(target_input, pad_idx)  #N,L
    memory_key_padding_mask = src_key_padding_mask
    # logit = model.model(enc_input=source, dec_input=target_input, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    logit = model.model(enc_input=source, dec_input=target_input, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    # 模型的输出维度是N，L，D_target_vocab_len,
    evaluation = model.evaluator(logit, target_real)

    if do_log:
        log_string_list.append(
            "Source string:  " + ' '.join(src_vocab.get_itos()[index] for index in source[0, 1:-1]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:-1]))
        log_string_list.append(
            "Target string:  " + ' '.join(tgt_vocab.get_itos()[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:-1]))
        log_string_list.append("Predict string: " + ' '.join(
            tgt_vocab.get_itos()[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(
            str(index.item()) for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)) + '\n')
    return logit, target_real, evaluation


def save_model_state_func(model, config):
    # model is from inner trainer framework,
    model_state_dict = model.model.state_dict()
    model_config = config['model_config']
    vocab_config = config['vocab_config']
    symbol_config = config['symbol_config']
    return {'model_state_dict': model_state_dict, 'model_config': model_config, 'vocab_config': vocab_config, 'symbol_config': symbol_config}

