import torch
import torch.nn.functional as F
from src.utilities.load_data import *
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.models.trainer.trainer_framework import Trainer
from src.tasks.translation.build_dataset import load_train_valid_split_set, get_data_iterator
from src.tasks.translation.build_model import TranslatorModel


def train_model(arguments):
    # get the dataset and data iterator
    train_set, valid_set = load_train_valid_split_set(arguments)
    train_iter, valid_iter, source_field, target_field = get_data_iterator(arguments, train_set=train_set, valid_set=valid_set)
    # get the model and arguments
    running_task = arguments['general']['running_task']
    use_bert = arguments['tasks'][running_task]['use_bert']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        pad_idx = arguments['dataset']['general']['pad_idx']
        model = TranslatorModel(arguments, src_vector=source_field.vocab.vectors, tgt_vector=target_field.vocab.vectors)
    else:
        pad_idx = arguments['dataset']['bert']['pad_idx']
        if trans_direct == 'en2zh':
            src_bert_model = get_bert_model(arguments, language='en')
            tgt_bert_model = get_bert_model(arguments, language='zh')
        elif trans_direct == 'zh2en':
            src_bert_model = get_bert_model(arguments, language='zh')
            tgt_bert_model = get_bert_model(arguments, language='en')
        model = TranslatorModel(arguments, src_bert_model=src_bert_model, tgt_bert_model=tgt_bert_model)
    # get the trainer
    trainer = Trainer(arguments)
    # start train
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'pad_idx': pad_idx},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'pad_idx': pad_idx}, get_model_state_func=get_model_state_func,
                  get_model_state_outer_params={'src_vocab': source_field.vocab, 'tgt_vocab': target_field.vocab})

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


def get_model_state_func(model, arguments, src_vocab, tgt_vocab):
    # model is from inner trainer framework,
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    use_bert = arguments['tasks'][running_task]['use_bert']
    save_model = arguments['training'][running_task]['model_save']['save_model']
    save_criterion = arguments['training'][running_task]['model_save']['save_criterion']
    save_optimizer = arguments['training'][running_task]['model_save']['save_optimizer']
    save_lr_scheduler = arguments['training'][running_task]['model_save']['save_lr_scheduler']
    save_evaluator = arguments['training'][running_task]['model_save']['save_evaluator']
    model_state_dict = {}
    if save_model:
        model_state_dict.update({'model': model.model.state_dict()})
    if save_criterion:
        model_state_dict.update({'criterion': model.criterion.state_dict()})
    if save_optimizer:
        model_state_dict.update({'optimizer': model.optimizer.state_dict()})
    if save_lr_scheduler:
        model_state_dict.update({'lr_scheduler': model.lr_scheduler.state_dict()})
    if save_evaluator:
        model_state_dict.update({'evaluator': model.evaluator.state_dict()})

    model_creation_args = {}
    model_vocab = {'src_vocab_stoi': src_vocab.stoi,
                    'src_vocab_itos': src_vocab.itos,
                    'tgt_vocab_stoi': tgt_vocab.stoi,
                    'tgt_vocab_itos': tgt_vocab.itos
                    }
    extra_states = {'d_model': arguments['model'][used_model]['d_model'],
                    'nhead': arguments['model'][used_model]['nhead'],
                    'src_vocab_len': arguments['model'][used_model]['src_vocab_len'],
                    'tgt_vocab_len': arguments['model'][used_model]['tgt_vocab_len']
                    }

    if use_bert not in ['static', 'dynamic']:
        bert_model_name = 'general'
    else:
        bert_model_name = 'bert'
    extra_states.update({'sos_token': arguments['dataset'][bert_model_name]['sos_token'],
                         'eos_token': arguments['dataset'][bert_model_name]['eos_token'],
                         'unk_token': arguments['dataset'][bert_model_name]['unk_token'],
                         'pad_token': arguments['dataset'][bert_model_name]['pad_token'],
                         'sos_idx': arguments['dataset'][bert_model_name]['sos_idx'],
                         'eos_idx': arguments['dataset'][bert_model_name]['eos_idx'],
                         'unk_idx': arguments['dataset'][bert_model_name]['unk_idx'],
                         'pad_idx': arguments['dataset'][bert_model_name]['pad_idx'],
                         })

    return {'model_state_dict': model_state_dict, 'model_creation_args': model_creation_args, 'model_vocab': model_vocab, 'extra_states': extra_states}
