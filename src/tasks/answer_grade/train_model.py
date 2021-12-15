import torch.nn.functional as F
from src.utilities.load_data import *
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.models.trainer.trainer_framework import Trainer
from src.tasks.answer_grade.build_dataset import load_train_valid_split_set, get_data_iterator
from src.tasks.answer_grade.build_model import AnswerGradeModel


def train_model(arguments):
    # get the dataset and data iterator
    train_set, valid_set = load_train_valid_split_set(arguments)
    train_iter, valid_iter, text_field = get_data_iterator(arguments, train_set=train_set, valid_set=valid_set)
    # get the model and arguments
    running_task = arguments['general']['running_task']
    use_bert = arguments['tasks'][running_task]['use_bert']

    if not use_bert:
        pad_idx = arguments['dataset']['general']['pad_idx']
        model = AnswerGradeModel(arguments, word_vector=text_field.vocab.vectors)
    else:
        pad_idx = arguments['dataset']['bert']['pad_idx']
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
def compute_predict_loss(model, data_example, device, do_log, log_string_list, text_field):
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
def compute_predict_evaluation(model, data_example, device, do_log, log_string_list, text_field):
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
    model_vocab = {'text_vocab_stoi': text_vocab.stoi,
                    'text_vocab_itos': text_vocab.itos
                    }
    extra_states = {'vocab_len': arguments['model'][used_model]['vocab_len']}
    if not use_bert:
        extra_states.update({'unk_token': arguments['dataset']['general']['unk_token'],
                             'pad_token': arguments['dataset']['general']['pad_token'],
                             'unk_idx': arguments['dataset']['general']['unk_idx'],
                             'pad_idx': arguments['dataset']['general']['pad_idx'],
                             })
    else:
        extra_states.update({'sos_token': arguments['dataset']['bert']['sos_token'],
                             'eos_token': arguments['dataset']['bert']['eos_token'],
                             'unk_token': arguments['dataset']['bert']['unk_token'],
                             'pad_token': arguments['dataset']['bert']['pad_token'],
                             'sos_idx': arguments['dataset']['bert']['sos_idx'],
                             'eos_idx': arguments['dataset']['bert']['eos_idx'],
                             'unk_idx': arguments['dataset']['bert']['unk_idx'],
                             'pad_idx': arguments['dataset']['bert']['pad_idx'],
                             })

    return {'model_state_dict': model_state_dict, 'model_creation_args': model_creation_args, 'model_vocab': model_vocab, 'extra_states': extra_states}
