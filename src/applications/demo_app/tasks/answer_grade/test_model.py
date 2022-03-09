import torch
import torch.nn.functional as F
import importlib
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.modules.tester.tester_framework import Tester
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import load_test_set, get_data_iterator, init_field_vocab_special_tokens


def test_model(arguments):
    used_model = arguments['net_structure']['model']
    # max_len = arguments['model'][used_model]['max_len']
    use_bert = arguments['net_structure']['use_bert']
    # get the dataset and data iterator
    test_set = load_test_set(arguments)
    test_iter, text_field = get_data_iterator(arguments, test_set=test_set)
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_vocab, _, _, _ = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    # re-init the field vocab
    text_vocab_stoi = model_vocab['text_vocab_stoi']
    text_vocab_itos = model_vocab['text_vocab_itos']
    unk_idx = arguments['dataset']['symbol']['unk_idx']
    unk_token = arguments['dataset']['symbol']['unk_token']
    pad_token = arguments['dataset']['symbol']['pad_token']
    pad_idx = arguments['dataset']['symbol']['pad_idx']
    if use_bert:
        sos_token = arguments['dataset']['symbol']['sos_token']
        eos_token = arguments['dataset']['symbol']['eos_token']
        sos_idx = arguments['dataset']['symbol']['sos_idx']
        eos_idx = arguments['dataset']['symbol']['eos_idx']
    if not use_bert:
        init_field_vocab_special_tokens(text_field, text_vocab_stoi, text_vocab_itos)
    else:
        init_field_vocab_special_tokens(text_field, text_vocab_stoi, text_vocab_itos, sos_token, eos_token, pad_token, unk_token)

    # start test
    tester.test(model=model, test_iter=test_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'text_field': text_field})


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, device, do_log, log_string_list, text_field):
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


def manage_model_state(arguments, loaded_weights):
    # 这里需要动态加载模型创建的类
    model_creator_module = arguments['net_structure']['model_creator_module']
    model_creator_class = arguments['net_structure']['model_creator_class']
    # module_path = src_code_root.replace('/', '.') + '.' + running_task + '.' + model_creator_module
    _model_creator_module = importlib.import_module(model_creator_module)
    creator_class = getattr(_model_creator_module, model_creator_class)

    model_state_dict = loaded_weights['model_state_dict']
    model_creation_args = loaded_weights['model_creation_args']
    model_vocab = loaded_weights['model_vocab']
    extra_states = loaded_weights['extra_states']
    training_states = loaded_weights['training_states']
    used_model = arguments['net_structure']['model']
    use_bert = arguments['net_structure']['use_bert']
    # arguments['model'][used_model].update({'text_vocab_stoi': model_vocab['text_vocab_stoi'],
    #                                        'text_vocab_itos': model_vocab['text_vocab_itos']
    #                                        })
    arguments['model'][used_model].update({'vocab_len': extra_states['vocab_len']})
    if ('symbol' not in arguments['dataset'].keys()) or (arguments['dataset']['symbol'] is None):
        arguments['dataset'].update({'symbol': {}})
    arguments['dataset']['symbol'].update({'unk_token': extra_states['unk_token'],
                                             'pad_token': extra_states['pad_token'],
                                             'unk_idx': extra_states['unk_idx'],
                                             'pad_idx': extra_states['pad_idx']
                                            })
    if use_bert:
        arguments['dataset']['symbol'].update({'sos_token': extra_states['sos_token'],
                                             'eos_token': extra_states['eos_token'],
                                             'sos_idx': extra_states['sos_idx'],
                                             'eos_idx': extra_states['eos_idx']
                                            })
    if use_bert:
        bert_model = get_bert_model(arguments, language='zh')
        #如果是用bert的话，需要把bert模型参数传入，创建model的时候创建bert对象，否则后面装载参数会出错。
        model_creation_args.update({'bert_model': bert_model})
    model = creator_class(arguments, **model_creation_args)
    if 'model' in model_state_dict.keys():
        model.model.load_state_dict(model_state_dict['model'])
        model_state_dict['model'] = None
    if 'criterion' in model_state_dict.keys():
        model.criterion.load_state_dict(model_state_dict['criterion'])
        model_state_dict['criterion'] = None
    if 'optimizer' in model_state_dict.keys():
        model.optimizer.load_state_dict(model_state_dict['optimizer'])
        model_state_dict['optimizer'] = None
    if 'lr_scheduler' in model_state_dict.keys():
        model.lr_scheduler.load_state_dict(model_state_dict['lr_scheduler'])
        model_state_dict['lr_scheduler'] = None
    if 'evaluator' in model_state_dict.keys():
        model.evaluator.load_state_dict(model_state_dict['evaluator'])
        model_state_dict['evaluator'] = None

    return (model, model_vocab, model_creation_args, extra_states, training_states)