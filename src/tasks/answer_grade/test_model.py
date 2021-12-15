import torch
import torch.nn.functional as F
import importlib
from src.utilities.load_data import *
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.models.tester.tester_framework import Tester
from src.tasks.answer_grade.build_dataset import load_test_set, get_data_iterator, init_field_vocab_special_tokens


def test_model(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    # max_len = arguments['model'][used_model]['max_len']
    use_bert = arguments['tasks'][running_task]['use_bert']
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
    if not use_bert:
        unk_idx = arguments['dataset']['general']['unk_idx']
        unk_token = arguments['dataset']['general']['unk_token']
        pad_token = arguments['dataset']['general']['pad_token']
        pad_idx = arguments['dataset']['general']['pad_idx']
    else:
        sos_token = arguments['dataset']['bert']['sos_token']
        eos_token = arguments['dataset']['bert']['eos_token']
        unk_token = arguments['dataset']['bert']['unk_token']
        pad_token = arguments['dataset']['bert']['pad_token']
        sos_idx = arguments['dataset']['bert']['sos_idx']
        eos_idx = arguments['dataset']['bert']['eos_idx']
        unk_idx = arguments['dataset']['bert']['unk_idx']
        pad_idx = arguments['dataset']['bert']['pad_idx']
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
    running_task = arguments['general']['running_task']
    model_creator_module = arguments['tasks'][running_task]['model_creator_module']
    model_creator_class = arguments['tasks'][running_task]['model_creator_class']
    src_code_root = arguments['file']['src']['tasks']
    module_path = src_code_root.replace('/', '.') + '.' + running_task + '.' + model_creator_module
    _model_creator_module = importlib.import_module(module_path)
    creator_class = getattr(_model_creator_module, model_creator_class)

    model_state_dict = loaded_weights['model_state_dict']
    model_creation_args = loaded_weights['model_creation_args']
    model_vocab = loaded_weights['model_vocab']
    extra_states = loaded_weights['extra_states']
    training_states = loaded_weights['training_states']
    used_model = arguments['tasks'][running_task]['model']
    use_bert = arguments['tasks'][running_task]['use_bert']
    # arguments['model'][used_model].update({'text_vocab_stoi': model_vocab['text_vocab_stoi'],
    #                                        'text_vocab_itos': model_vocab['text_vocab_itos']
    #                                        })
    arguments['model'][used_model].update({'vocab_len': extra_states['vocab_len']})
    if not use_bert:
        if ('general' not in arguments['dataset'].keys()) or (arguments['dataset']['general'] is None):
            arguments['dataset'].update({'general': {}})
        arguments['dataset']['general'].update({'unk_token': extra_states['unk_token'],
                                                 'pad_token': extra_states['pad_token'],
                                                 'unk_idx': extra_states['unk_idx'],
                                                 'pad_idx': extra_states['pad_idx']
                                                })
    else:
        if ('bert' not in arguments['dataset'].keys()) or (arguments['dataset']['bert'] is None):
            arguments['dataset'].update({'bert': {}})
        arguments['dataset']['bert'].update({'sos_token': extra_states['sos_token'],
                                             'eos_token': extra_states['eos_token'],
                                             'unk_token': extra_states['unk_token'],
                                             'pad_token': extra_states['pad_token'],
                                             'sos_idx': extra_states['sos_idx'],
                                             'eos_idx': extra_states['eos_idx'],
                                             'unk_idx': extra_states['unk_idx'],
                                             'pad_idx': extra_states['pad_idx']
                                            })
    if use_bert:
        bert_model = get_bert_model(arguments, language='zh')
        #如果是用bert的话，需要把bert模型参数传入，创建model的时候创建bert对象，否则后面装载参数会出错。
        model_creation_args.update({'bert_model': bert_model})
    model = creator_class(arguments, **model_creation_args)
    if 'model' in model_state_dict.keys():
        model.model.load_state_dict(model_state_dict['model'])
    if 'criterion' in model_state_dict.keys():
        model.criterion.load_state_dict(model_state_dict['criterion'])
    if 'optimizer' in model_state_dict.keys():
        model.optimizer.load_state_dict(model_state_dict['optimizer'])
    if 'lr_scheduler' in model_state_dict.keys():
        model.lr_scheduler.load_state_dict(model_state_dict['lr_scheduler'])
    if 'evaluator' in model_state_dict.keys():
        model.evaluator.load_state_dict(model_state_dict['evaluator'])

    return (model, model_vocab, model_creation_args, extra_states, training_states)