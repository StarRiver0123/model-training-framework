import torch.nn.functional as F
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.modules.trainer.trainer_framework import Trainer
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import load_train_valid_split_set, get_data_iterator
from build_model import AnswerGradeModel


def train_model(arguments):
    # get the dataset and data iterator
    train_set, valid_set = load_train_valid_split_set(arguments)
    train_iter, valid_iter, text_field = get_data_iterator(arguments, train_set=train_set, valid_set=valid_set)
    # get the model and arguments
    use_bert = arguments['net_structure']['use_bert']
    pad_idx = arguments['dataset']['symbol']['pad_idx']
    if not use_bert:
        model = AnswerGradeModel(arguments, word_vector=text_field.vocab.vectors)
    else:
        bert_model = get_bert_model(arguments, language='zh')
        model = AnswerGradeModel(arguments, bert_model=bert_model)
    # get the trainer
    trainer = Trainer(arguments)
    # start train
    trainer.train(model=model, train_iter=train_iter, compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={'text_field': text_field},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'text_field': text_field},
                  get_model_state_func=get_model_state_func, get_model_state_outer_params={'text_vocab': text_field.vocab})

# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, text_field):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    negative = data_example.Negative.to(device)
    source_vector, target_vector, negative_vector = model.model(source, target, negative)
    loss = model.criterion(source_vector, target_vector, negative_vector)
    if do_log:
        log_string_list.append("Source string:  " + ' '.join(text_field.vocab.itos[index] for index in source[0]))
        log_string_list.append("Target string:  " + ' '.join(text_field.vocab.itos[index] for index in target[0]))
        log_string_list.append("Negative string: " + ' '.join(text_field.vocab.itos[index] for index in negative[0]) + '\n')
    return None, None, loss

# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, text_field):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    source_vector, target_vector = model.model(source, target)
    evaluation = model.evaluator(source_vector, target_vector)
    if do_log:
        log_string_list.append("Source string:  " + ' '.join(text_field.vocab.itos[index] for index in source[0]))
        log_string_list.append("Target string:  " + ' '.join(text_field.vocab.itos[index] for index in target[0]) + '\n')
    return None, None, evaluation


def get_model_state_func(model, arguments, text_vocab):
    # model is from inner trainer framework,
    used_model = arguments['net_structure']['model']
    use_bert = arguments['net_structure']['use_bert']
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
    model_vocab = {'text_vocab_stoi': text_vocab.stoi,
                    'text_vocab_itos': text_vocab.itos
                    }
    extra_states = {'vocab_len': arguments['model'][used_model]['vocab_len']}
    extra_states.update({'unk_token': arguments['dataset']['symbol']['unk_token'],
                         'pad_token': arguments['dataset']['symbol']['pad_token'],
                         'unk_idx': arguments['dataset']['symbol']['unk_idx'],
                         'pad_idx': arguments['dataset']['symbol']['pad_idx']
                         })
    if use_bert:
        extra_states.update({'sos_token': arguments['dataset']['symbol']['sos_token'],
                             'eos_token': arguments['dataset']['symbol']['eos_token'],
                             'sos_idx': arguments['dataset']['symbol']['sos_idx'],
                             'eos_idx': arguments['dataset']['symbol']['eos_idx']
                             })

    return {'model_state_dict': model_state_dict, 'model_creation_args': model_creation_args, 'model_vocab': model_vocab, 'extra_states': extra_states}