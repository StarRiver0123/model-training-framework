import torch
import torch.nn.functional as F
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.modules.trainer.trainer_framework import Trainer
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import *
from build_model import TrainingModel


def train_model(config):
    # step 1: build dataset and vocab
    # 注意这里需要优化，数据量很大时，train_iter消耗了太多地内存资源。需要加上无用资源释放代码!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_iter, valid_iter, src_field, tgt_field = build_train_dataset_and_vocab_pipeline(config)
    # step 2: build model
    model = TrainingModel(config, src_field.vocab.vectors, tgt_field.vocab.vectors)
    # step 3: get the trainer
    trainer = Trainer(config)
    # step 4: start train
    pad_idx = config['symbol_config']['pad_idx']
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={
                      'source_field': src_field, 'target_field': tgt_field, 'pad_idx': pad_idx},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={
                      'source_field': src_field, 'target_field': tgt_field, 'pad_idx': pad_idx}, save_model_state_func=save_model_state_func,
                  save_model_state_outer_params={})



# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, source_field, target_field, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if data_example.Source.size(1) > max_len:
        source = data_example.Source[:, :max_len].to(device)
    else:
        source = data_example.Source.to(device)
    if data_example.Target.size(1) > max_len:
        target = data_example.Target[:, :max_len].to(device)
    else:
        target = data_example.Target.to(device)
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
            "Source string:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:-1]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:-1]))
        log_string_list.append(
            "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:-1]))
        log_string_list.append("Predict string: " + ' '.join(
            target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(
            str(index.item()) for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)) + '\n')
    return logit, target_real, loss

# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, source_field, target_field, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if data_example.Source.size(1) > max_len:
        source = data_example.Source[:, :max_len].to(device)
    else:
        source = data_example.Source.to(device)
    if data_example.Target.size(1) > max_len:
        target = data_example.Target[:, :max_len].to(device)
    else:
        target = data_example.Target.to(device)

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
            "Source string:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:-1]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:-1]))
        log_string_list.append(
            "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:-1]))
        log_string_list.append("Predict string: " + ' '.join(
            target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(
            str(index.item()) for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)) + '\n')
    return logit, target_real, evaluation


def save_model_state_func(model, config):
    # model is from inner trainer framework,
    model_state_dict = model.model.state_dict()
    model_vocab = config['model_vocab']
    model_config = config['model_config']
    symbol_config = config['symbol_config']
    return {'model_state_dict': model_state_dict, 'model_vocab': model_vocab, 'model_config': model_config, 'symbol_config': symbol_config}

