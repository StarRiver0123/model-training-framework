import numpy as np
import torch.nn.functional as F
from src.utilities.load_data import *
from src.models.trainer.trainer_framework import Trainer
from src.tasks.ner.build_dataset import load_train_valid_split_set, get_data_iterator
from src.tasks.ner.build_model import NERModel


def train_model(arguments):
    # get the dataset and data iterator
    train_set, valid_set = load_train_valid_split_set(arguments)
    train_iter, valid_iter, source_field, target_field = get_data_iterator(arguments, train_set=train_set, valid_set=valid_set)
    # get the model and arguments
    model = NERModel(arguments)
    # get the trainer
    trainer = Trainer(arguments)
    # start train
    special_token_ids = [target_field.vocab.stoi[target_field.init_token], target_field.vocab.stoi[target_field.eos_token], target_field.vocab.stoi[target_field.pad_token], target_field.vocab.stoi[target_field.unk_token]]
    labels = [tag for tag in list(target_field.vocab.stoi.values()) if tag not in special_token_ids]
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={
                      'source_field': source_field, 'target_field': target_field},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'labels': labels}, get_model_state_func=get_model_state_func,
                  get_model_state_outer_params={'tgt_vocab': target_field.vocab})


# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, device, do_log, log_string_list, source_field, target_field):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    pad_idx = target_field.vocab.stoi[target_field.pad_token]
    unk_idx = target_field.vocab.stoi[target_field.unk_token]
    sos_idx = target_field.vocab.stoi[target_field.init_token]
    eos_idx = target_field.vocab.stoi[target_field.eos_token]
    O_idx = target_field.vocab.stoi['O']
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    emission = model.model.bert_emit(seq_input=source)  # emission是3维：N,L,D
    mask1 = ((target != pad_idx) * (target != unk_idx) * (target != sos_idx) * (target != eos_idx) * (target != O_idx)).byte()   # 屏蔽pad,unk,sos,eos对loss的影响, 并加大正例tag的权重
    mask2 = (target == O_idx).byte()
    # mask3 = ((target == unk_idx) + (target == sos_idx) + (target == eos_idx)).byte() * 2
    mask = (mask1 + mask2).to(device)
    # mask = (target != target_field.vocab.stoi[target_field.pad_token]).byte().to(device)   # 屏蔽pad对loss的影响
    loss = -model.model.crf(emissions=emission, tags=target, mask=mask, reduction='token_mean')
    predict = model.model.crf.decode(emission, mask=mask)   # 模型输出是2层list
    if do_log:
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, :]))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(str(index) for index in predict[0]) + '\n')
    return emission, target, loss


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, device, do_log, log_string_list, source_field, target_field, labels):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    pad_idx = target_field.vocab.stoi[target_field.pad_token]
    unk_idx = target_field.vocab.stoi[target_field.unk_token]
    sos_idx = target_field.vocab.stoi[target_field.init_token]
    eos_idx = target_field.vocab.stoi[target_field.eos_token]
    O_idx = target_field.vocab.stoi['O']
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    emission = model.model.bert_emit(seq_input=source)
    mask1 = ((target != pad_idx) * (target != unk_idx) * (target != sos_idx) * (target != eos_idx) * (target != O_idx)).byte() * 10  # 屏蔽pad,unk,sos,eos对loss的影响, 并加大正例tag的权重
    mask2 = (target == O_idx).byte()
    mask = (mask1 + mask2).to(device)
    # mask = (target != target_field.vocab.stoi[target_field.pad_token]).byte().to(device)
    predict = model.model.crf.decode(emission, mask=mask)
    predict_flattened = []
    for pred in predict:
        predict_flattened += pred
    target_flattened = target[mask.bool()].to('cpu').tolist()
    evaluation = model.evaluator(predict_flattened, target_flattened, labels=labels)
    if do_log:
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, :]))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(str(index) for index in predict[0]) + '\n')
    return predict, target, evaluation



def get_model_state_func(model, arguments, tgt_vocab):
    # model is from inner trainer framework,
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
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
    model_vocab = {'tgt_vocab_stoi': tgt_vocab.stoi,
                    'tgt_vocab_itos': tgt_vocab.itos
                    }
    extra_states = {'num_tags': arguments['model'][used_model]['num_tags']}
    return {'model_state_dict': model_state_dict, 'model_creation_args': model_creation_args, 'model_vocab': model_vocab, 'extra_states': extra_states}
