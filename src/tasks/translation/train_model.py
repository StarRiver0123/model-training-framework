import torch.nn.functional as F
from src.tasks.translation.build_dataset import load_train_valid_split_set, get_data_iterator
from src.tasks.translation.build_model import TranslatorModel
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.models.trainer.trainer_framework import Trainer
from src.utilities.load_data import *


def train_model(arguments):
    # get the dataset and data iterator
    train_set, valid_set = load_train_valid_split_set(arguments)
    train_iter, valid_iter, source_field, target_field = get_data_iterator(arguments, train_set=train_set, valid_set=valid_set)
    # get the model and arguments
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        pad_token = arguments['dataset']['general']['pad_token']
        arguments['model'][used_model].update({'src_vocab_len': source_field.vocab.vectors.shape[0],
                                               'tgt_vocab_len': target_field.vocab.vectors.shape[0]})
        model = TranslatorModel(arguments, src_vector=source_field.vocab.vectors, tgt_vector=target_field.vocab.vectors)
    else:
        pad_token = arguments['dataset']['bert']['pad_token']
        if trans_direct == 'en2zh':
            src_bert_model = get_bert_model(arguments, language='en')
            tgt_bert_model = get_bert_model(arguments, language='zh')
        elif trans_direct == 'zh2en':
            src_bert_model = get_bert_model(arguments, language='zh')
            tgt_bert_model = get_bert_model(arguments, language='en')
        arguments['model'][used_model].update({'src_vocab_len': len(source_field.vocab),
                                               'tgt_vocab_len': len(target_field.vocab)})
        model = TranslatorModel(arguments, src_bert_model=src_bert_model, tgt_bert_model=tgt_bert_model)
    pad_idx = source_field.vocab.stoi[pad_token]
    # get the trainer
    trainer = Trainer(arguments)
    # start train
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_func=compute_predict,
                  compute_predict_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'pad_idx': pad_idx},
                  valid_iter=valid_iter, get_model_state_func=get_model_state_func,
                  get_model_state_outer_params={'src_vocab': source_field.vocab, 'tgt_vocab': target_field.vocab})

# this function needs to be defined from the view of concrete task
def compute_predict(model, data_example, device, do_log, log_string_list, source_field, target_field, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    # 要注意decoder的input和target要错一位。！！！！！！！！很重要
    target_input = target[:, :-1]
    target_real = target[:, 1:]
    src_mask = gen_full_false_mask(source, source)  # L,L
    tgt_mask = gen_seq_only_mask(target_input, target_input)  # L,L
    src_key_padding_mask = gen_pad_only_mask(source, pad_idx)   #N,L
    tgt_key_padding_mask = gen_pad_only_mask(target_input, pad_idx)  #N,L
    memory_key_padding_mask = src_key_padding_mask
    # logit = model.model(enc_input=source, dec_input=target_input, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    logit = model.model(enc_input=source, dec_input=target_input, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    if do_log:
        log_string_list.append(
            "Source string:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:]))
        log_string_list.append(
            "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:]))
        log_string_list.append("Predict string: " + ' '.join(
            target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(
            str(index.item()) for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)) + '\n')
    return logit, target_real


def get_model_state_func(model, arguments, src_vocab, tgt_vocab):
    # model is from inner trainer framework,
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
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

    extra_states = {'d_model': arguments['model'][used_model]['d_model'],
                    'nhead': arguments['model'][used_model]['nhead'],
                    'src_vocab_len': arguments['model'][used_model]['src_vocab_len'],
                    'tgt_vocab_len': arguments['model'][used_model]['tgt_vocab_len'],
                    'src_vocab_stoi': src_vocab.stoi,
                    'src_vocab_itos': src_vocab.itos,
                    'tgt_vocab_stoi': tgt_vocab.stoi,
                    'tgt_vocab_itos': tgt_vocab.itos
                    }

    if (use_bert != 'static') and (use_bert != 'dynamic'):
        bert_model_name = 'general'
    else:
        bert_model_name = 'bert'
    extra_states.update({'start_token': arguments['dataset'][bert_model_name]['start_token'],
                         'end_token': arguments['dataset'][bert_model_name]['end_token'],
                         'unk_token': arguments['dataset'][bert_model_name]['unk_token'],
                         'pad_token': arguments['dataset'][bert_model_name]['pad_token'],
                         'start_idx': arguments['dataset'][bert_model_name]['start_idx'],
                         'end_idx': arguments['dataset'][bert_model_name]['end_idx'],
                         'unk_idx': arguments['dataset'][bert_model_name]['unk_idx'],
                         'pad_idx': arguments['dataset'][bert_model_name]['pad_idx'],
                         })

    return {'model_state_dict': model_state_dict, 'model_creation_args': model_creation_args, 'extra_states': extra_states}
