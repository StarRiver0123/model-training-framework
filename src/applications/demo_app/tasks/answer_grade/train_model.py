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
    train_iter, valid_iter = build_train_dataset_and_vocab_pipeline(config)
    # step 2: build model
    used_model = config['net_structure']['model']
    use_bert = config['model'][used_model]['use_bert']
    transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['question_seqs'])[1]
    if not use_bert:
        word_vectors = config['vector_config'][transforming_key]
    else:
        word_vectors = None
    vocab = config['vocab_config'][transforming_key]
    model = TrainingModel(config, word_vectors)
    # step 3: get the trainer
    trainer = Trainer(config)
    # step 4: start train
    trainer.train(model=model, train_iter=train_iter, compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={'vocab': vocab},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'vocab': vocab},
                  save_model_state_func=save_model_state_func, save_model_state_outer_params={})

# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, vocab):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    source = data_example[0].to(device)
    target = data_example[1].to(device)
    negative = data_example[2].to(device)
    source_vector, target_vector, negative_vector = model.model(source, target, negative)
    loss = model.criterion(source_vector, target_vector, negative_vector)
    if do_log:
        log_string_list.append("Source string:  " + ' '.join(vocab.get_itos()[index] for index in source[0]))
        log_string_list.append("Target string:  " + ' '.join(vocab.get_itos()[index] for index in target[0]))
        log_string_list.append("Negative string: " + ' '.join(vocab.get_itos()[index] for index in negative[0]) + '\n')
    return None, None, loss

# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, vocab):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    source = data_example[0].to(device)
    target = data_example[1].to(device)
    source_vector, target_vector = model.model(source, target)
    evaluation = model.evaluator(source_vector, target_vector)
    if do_log:
        log_string_list.append("Source string:  " + ' '.join(vocab.get_itos()[index] for index in source[0]))
        log_string_list.append("Target string:  " + ' '.join(vocab.get_itos()[index] for index in target[0]) + '\n')
    return None, None, evaluation


def save_model_state_func(model, config):
    # model and config is from inner trainer framework
    model_state_dict = model.model.state_dict()
    model_config = config['model_config']
    vocab_config = config['vocab_config']
    symbol_config = config['symbol_config']
    return {'model_state_dict': model_state_dict, 'model_config': model_config, 'vocab_config': vocab_config,
            'symbol_config': symbol_config}

