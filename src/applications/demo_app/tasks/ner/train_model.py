import numpy as np
import torch.nn.functional as F
from src.utilities.load_data import *
from src.modules.trainer.trainer_framework import Trainer
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import load_train_valid_split_set, get_data_iterator
from build_model import NERModel
# from src.applications.demo_app.tasks.ner.build_dataset import load_train_valid_split_set, get_data_iterator
# from src.applications.demo_app.tasks.ner.build_model import NERModel


def train_model(arguments):
    used_model = arguments['net_structure']['model']
    # get the dataset and data iterator
    train_set, valid_set = load_train_valid_split_set(arguments)
    train_iter, valid_iter, source_field, target_field = get_data_iterator(arguments, train_set=train_set, valid_set=valid_set)
    # get the model and arguments
    if used_model == 'lstm_crf':
        word_vector = source_field.vocab.vectors
    else:
        word_vector = None
    model = NERModel(arguments, word_vector)
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
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, source_field, target_field):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    pad_idx = target_field.vocab.stoi[target_field.pad_token]
    unk_idx = target_field.vocab.stoi[target_field.unk_token]
    sos_idx = target_field.vocab.stoi[target_field.init_token]
    eos_idx = target_field.vocab.stoi[target_field.eos_token]
    O_idx = target_field.vocab.stoi['O']
    if data_example.Source.size(1) > max_len:
        source = data_example.Source[:, :max_len].to(device)
    else:
        source = data_example.Source.to(device)
    if data_example.Target.size(1) > max_len:
        target = data_example.Target[:, :max_len].to(device)
    else:
        target = data_example.Target.to(device)

    emission = model.model.emit(seq_input=source)  # emission是3维：N,L,D
    # mask1 = ((target != pad_idx) * (target != unk_idx) * (target != sos_idx) * (target != eos_idx) * (target != O_idx)).byte()   # 屏蔽pad,unk,sos,eos对loss的影响, 并加大正例tag的权重
    # mask2 = (target == O_idx).byte()
    # mask3 = ((target == unk_idx) + (target == sos_idx) + (target == eos_idx)).byte() * 2
    # mask = (mask1 + mask2).to(device)
    # mask = (target != target_field.vocab.stoi[target_field.pad_token]).byte().to(device)   # 屏蔽pad对loss的影响
    if model.criterion == 'crf':
        mask = (target != pad_idx).byte()
        loss = -model.model.crf(emissions=emission, tags=target, mask=mask, reduction='token_mean')
        predict = model.model.crf.decode(emission, mask=mask)   # 模型输出是2层list：N，L
        # loss = -model.model.crf(emissions=emission, tags=target)
        # predict = model.model.crf.decode(emission)  # 模型输出是2层list：N，L
    else:
        # pytorch CrossEntropyLoss的输入维度有两种方式：
        # （1） input为N，C；target为N，需要对predict做reshape（-1，D_target_vocab_len）
        # （2） input为N，C，L，target为N，L。要把分类放在第二维，需要对predict进行转置transpose(-1,-2)
        logits_flatten = emission.reshape(-1, emission.size(-1))
        target_flatten = target.reshape(-1)
        loss = model.criterion(logits_flatten, target_flatten)
        predict = F.softmax(emission, dim=-1).argmax(dim=-1)
    if do_log:
        # print(model.model.bert_model.encoder.layer[11].output.dense.weight)  #打印bert模型权重，检查训练过程中参数值是否发生变化。
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, :]))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        if model.criterion == 'crf':
            log_string_list.append("Predict code:   " + ' '.join(str(index) for index in predict[0]) + '\n')
        else:
            log_string_list.append("Predict code:   " + ' '.join(str(index.item()) for index in predict[0]) + '\n')
    return emission, target, loss


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, source_field, target_field, labels):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    pad_idx = target_field.vocab.stoi[target_field.pad_token]
    unk_idx = target_field.vocab.stoi[target_field.unk_token]
    sos_idx = target_field.vocab.stoi[target_field.init_token]
    eos_idx = target_field.vocab.stoi[target_field.eos_token]
    O_idx = target_field.vocab.stoi['O']
    if data_example.Source.size(1) > max_len:
        source = data_example.Source[:, :max_len].to(device)
    else:
        source = data_example.Source.to(device)
    if data_example.Target.size(1) > max_len:
        target = data_example.Target[:, :max_len].to(device)
    else:
        target = data_example.Target.to(device)
    emission = model.model.emit(seq_input=source)
    # mask1 = ((target != pad_idx) * (target != unk_idx) * (target != sos_idx) * (target != eos_idx) * (target != O_idx)).byte() * 10  # 屏蔽pad,unk,sos,eos对loss的影响, 并加大正例tag的权重
    # mask2 = (target == O_idx).byte()
    # mask = (mask1 + mask2).to(device)
    # mask = (target != target_field.vocab.stoi[target_field.pad_token]).byte().to(device)
    mask = (target != pad_idx).byte()
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
    used_model = arguments['net_structure']['model']
    save_model = arguments['training']['model_save']['save_model']
    save_criterion = arguments['training']['model_save']['save_criterion']
    save_optimizer = arguments['training']['model_save']['save_optimizer']
    save_lr_scheduler = arguments['training']['model_save']['save_lr_scheduler']
    save_evaluator = arguments['training']['model_save']['save_evaluator']
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
