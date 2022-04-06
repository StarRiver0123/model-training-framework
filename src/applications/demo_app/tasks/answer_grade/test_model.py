import torch
import torch.nn.functional as F
import importlib
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.modules.tester.tester_framework import Tester
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import *
from build_model import *

def test_model(config):
    # step 1: load model and config states
    load_model_states_into_config(config)
    # step 2: get the tester
    tester = Tester(config)
    # step 3: get the data iterator and field
    test_iter = build_test_dataset_pipeline(config)
    # step 4: get the model and update config
    model = load_model_and_vocab(config)
    # step 5: start test
    used_model = config['model_config']['model_name']
    transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['question_seqs'])[1]
    vocab = config['vocab_config'][transforming_key]
    tester.test(model=model, test_iter=test_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'vocab': vocab})


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, vocab):
    # model, data_example, device, do_log, log_string_list are from inner tester framework
    # output: predict: N,L,D,  target: N,L
    source = data_example[0].to(device)
    target = data_example[1].to(device)
    source_vector, target_vector = model.model(source, target)
    evaluation = model.evaluator(source_vector, target_vector)
    if do_log:
        log_string_list.append("Source string:  " + ' '.join(vocab.get_itos()[index] for index in source[0]))
        log_string_list.append("Target string:  " + ' '.join(vocab.get_itos()[index] for index in target[0]) + '\n')
    return None, None, evaluation


def load_model_and_vocab(config):
    model_state_dict = config['model_state_dict']
    model = TestingModel(config)
    model.model.load_state_dict(model_state_dict)
    return model


def load_model_states_into_config(config):
    model_save_root = config['check_point_root']
    saved_model_file = config['net_structure']['saved_model_file']
    model_file_path = model_save_root + os.path.sep + saved_model_file
    model_states = torch.load(model_file_path)
    config.update(model_states)
