import torch
import torch.nn.functional as F
import importlib
from src.utilities.load_data import *
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask, gen_full_false_mask
from src.models.tester.tester_framework import Tester
from src.tasks.translation.build_dataset import load_test_set, get_data_iterator, init_field_vocab_special_tokens


def test_model(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    max_len = arguments['model'][used_model]['max_len']
    use_bert = arguments['tasks'][running_task]['use_bert']
    # get the dataset and data iterator
    test_set = load_test_set(arguments)
    test_iter, source_field, target_field = get_data_iterator(arguments, test_set=test_set)
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_vocab, _, _, _ = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    # re-init the field vocab
    src_vocab_stoi = model_vocab['src_vocab_stoi']
    src_vocab_itos = model_vocab['src_vocab_itos']
    tgt_vocab_stoi = model_vocab['tgt_vocab_stoi']
    tgt_vocab_itos = model_vocab['tgt_vocab_itos']
    if use_bert not in ['static', 'dynamic']:
        bert_model_name = 'general'
    else:
        bert_model_name = 'bert'
    sos_token = arguments['dataset'][bert_model_name]['sos_token']
    eos_token = arguments['dataset'][bert_model_name]['eos_token']
    unk_token = arguments['dataset'][bert_model_name]['unk_token']
    pad_token = arguments['dataset'][bert_model_name]['pad_token']
    pad_idx = arguments['dataset'][bert_model_name]['pad_idx']
    if use_bert not in ['static', 'dynamic']:
        init_field_vocab_special_tokens(source_field, src_vocab_stoi, src_vocab_itos)
        init_field_vocab_special_tokens(target_field, tgt_vocab_stoi, tgt_vocab_itos)
    else:
        init_field_vocab_special_tokens(source_field, src_vocab_stoi, src_vocab_itos, sos_token, eos_token, pad_token, unk_token)
        init_field_vocab_special_tokens(target_field, tgt_vocab_stoi, tgt_vocab_itos, sos_token, eos_token, pad_token, unk_token)

    # start test
    tester.test(model=model, test_iter=test_iter,
                  compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'pad_idx': pad_idx})


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, source_field, target_field, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner tester framework
    # output: predict: N,L,D,  target: N,L
    if data_example.Source.size(1) > max_len:
        source = data_example.Source[:, :max_len].to(device)
    else:
        source = data_example.Source.to(device)
    if data_example.Target.size(1) > max_len:
        target = data_example.Target[:, :max_len].to(device)
    else:
        target = data_example.Target.to(device)
    target_real = target[:, 1:-1]     # remove sos_token and end_sos_token, there should be no pad_token
    sos_idx = target[0, 0].item()
    eos_idx = target[0, -1].item()
    predict = greedy_decode(model, source, device, max_len, sos_idx, eos_idx, pad_idx)
    evaluation = model.evaluator(predict, target_real)
    if do_log:
        log_string_list.append(
            "Source string:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:-1]))    #可能有的词会被替换成<unk>输出，因为在soti的时候在字典里查不到。
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:-1]))
        log_string_list.append(
            "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:-1]))
        log_string_list.append("Predict string: " + ' '.join(target_field.vocab.itos[index] for index in predict[0, 1:]))
        log_string_list.append("Predict code:   " + ' '.join(str(index.item()) for index in predict[0, 1:]) + '\n')
    return predict, target_real, evaluation


def greedy_decode(model, source, device, max_len, sos_idx, eos_idx, pad_idx=None):
    src_key_padding_mask = memory_key_padding_mask = gen_pad_only_mask(source, pad_idx)   #N,L
    enc_out = model.model.encoder(source, src_key_padding_mask=src_key_padding_mask)
    target_input = torch.ones(1, 1).fill_(sos_idx).type(torch.long).to(device)
    for i in range(max_len - 1):
        tgt_key_padding_mask = gen_pad_only_mask(target_input, pad_idx)  # N,L
        tgt_mask = gen_seq_only_mask(target_input, target_input)  # L,L
        dec_out = model.model.decoder(target_input, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        last_logit = model.model.predictor(dec_out[:, -1:])
        last_word = F.softmax(last_logit, dim=-1).argmax(dim=-1)     # last_word size: (1,1)
        if last_word.item() == eos_idx:
            break
        target_input = torch.cat([target_input, last_word], dim=-1)
    predict = target_input
    return predict


def manage_model_state(arguments, loaded_weights):
    # 这里需要动态加载模型创建的类
    running_task = arguments['general']['running_task']
    model_creator_module = arguments['tasks'][running_task]['model_creator_module']
    model_creator_class = arguments['tasks'][running_task]['model_creator_class']
    src_code_root = arguments['file']['src']['tasks']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
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
    # arguments['model'][used_model].update({ 'src_vocab_stoi': model_vocab['src_vocab_stoi'],
    #                                         'src_vocab_itos': model_vocab['src_vocab_itos'],
    #                                         'tgt_vocab_stoi': model_vocab['tgt_vocab_stoi'],
    #                                         'tgt_vocab_itos': model_vocab['tgt_vocab_itos']
    #                                         })
    arguments['model'][used_model].update({'d_model': extra_states['d_model'],
                                            'nhead': extra_states['nhead'],
                                            'src_vocab_len': extra_states['src_vocab_len'],
                                            'tgt_vocab_len': extra_states['tgt_vocab_len']
                                            })
    if use_bert not in ['static', 'dynamic']:
        bert_model_name = 'general'
    else:
        bert_model_name = 'bert'
    if (bert_model_name not in arguments['dataset'].keys()) or (
            arguments['dataset'][bert_model_name] is None):
        arguments['dataset'].update({bert_model_name: {}})
    arguments['dataset'][bert_model_name].update({'sos_token': extra_states['sos_token'],
                                                       'eos_token': extra_states['eos_token'],
                                                       'unk_token': extra_states['unk_token'],
                                                       'pad_token': extra_states['pad_token'],
                                                       'sos_idx': extra_states['sos_idx'],
                                                       'eos_idx': extra_states['eos_idx'],
                                                       'unk_idx': extra_states['unk_idx'],
                                                       'pad_idx': extra_states['pad_idx']
                                                       })
    if use_bert == 'dynamic':
        if trans_direct == 'en2zh':
            src_bert_model = get_bert_model(arguments, language='en')
            tgt_bert_model = get_bert_model(arguments, language='zh')
        elif trans_direct == 'zh2en':
            src_bert_model = get_bert_model(arguments, language='zh')
            tgt_bert_model = get_bert_model(arguments, language='en')
        #如果是用动态bert的话，需要把bert模型参数传入，创建model的时候创建bert对象，否则后面装载参数会出错。
        model_creation_args.update({'src_bert_model': src_bert_model, 'tgt_bert_model': tgt_bert_model})
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