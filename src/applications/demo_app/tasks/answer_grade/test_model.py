import torch
import torch.nn.functional as F
import importlib
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.modules.tester.tester_framework import Tester
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import *
from build_model import TestingModel

def test_model(config):
    # step 1: load model and config states
    load_model_states_into_config(config)
    # step 2: get the tester
    tester = Tester(config)
    # step 3: get the data iterator and field
    test_iter, text_field = build_test_dataset_pipeline(config)
    # step 4: get the model and update config
    model, _, _ = load_model_and_vocab(config, text_field)
    # step 5: start test
    tester.test(model=model, test_iter=test_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'text_field': text_field})


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, text_field):
    # model, data_example, device, do_log, log_string_list are from inner tester framework
    # output: predict: N,L,D,  target: N,L
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    source_vector, target_vector = model.model(source, target)
    evaluation = model.evaluator(source_vector, target_vector)
    if do_log:
        log_string_list.append("Source string:  " + ' '.join(text_field.vocab.itos[index] for index in source[0]))
        log_string_list.append("Target string:  " + ' '.join(text_field.vocab.itos[index] for index in target[0]) + '\n')
    return None, None, evaluation


def load_model_and_vocab(config, text_field=None):
    model_state_dict = config['model_state_dict']
    model = TestingModel(config)
    model.model.load_state_dict(model_state_dict)
    vocab_stoi = config['model_vocab']['vocab_stoi']
    vocab_itos = config['model_vocab']['vocab_itos']
    sos_token = config['symbol_config']['sos_token']
    eos_token = config['symbol_config']['eos_token']
    unk_token = config['symbol_config']['unk_token']
    pad_token = config['symbol_config']['pad_token']
    use_bert = config['model_config']['use_bert']
    if not use_bert:
        build_field_vocab_special_tokens(text_field, vocab_stoi, vocab_itos)
    else:
        build_field_vocab_special_tokens(text_field, vocab_stoi, vocab_itos, sos_token, eos_token, pad_token, unk_token)
    return model, vocab_stoi, vocab_itos

def load_model_states_into_config(config):
    model_save_root = config['check_point_root']
    saved_model_file = config['net_structure']['saved_model_file']
    model_file_path = model_save_root + os.path.sep + saved_model_file
    model_states = torch.load(model_file_path)
    config.update(model_states)
    config['net_structure'].update({'use_bert': model_states['model_config']['use_bert']})
# return model_states
