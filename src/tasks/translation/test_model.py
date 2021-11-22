import torch
import torch.nn.functional as F
from src.tasks.translation.build_dataset import load_test_set, get_data_iterator, init_field_vocab_special_tokens
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.models.tester.tester_framework import Tester
import importlib


def test_model(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    max_len = arguments['model'][used_model]['max_len']
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
    # get the dataset and data iterator
    test_set = load_test_set(arguments)
    test_iter, source_field, target_field = get_data_iterator(arguments, test_set=test_set)
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_creation_args, extra_states, training_states = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    # re-init the field vocab
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        bert_model_name = 'general'
    else:
        bert_model_name = 'bert'
    start_token = arguments['dataset'][bert_model_name]['start_token']
    end_token = arguments['dataset'][bert_model_name]['end_token']
    unk_token = arguments['dataset'][bert_model_name]['unk_token']
    pad_token = arguments['dataset'][bert_model_name]['pad_token']
    pad_idx = arguments['dataset'][bert_model_name]['pad_idx']
    src_vocab_stoi = arguments['model'][used_model]['src_vocab_stoi']
    src_vocab_itos = arguments['model'][used_model]['src_vocab_itos']
    tgt_vocab_stoi = arguments['model'][used_model]['tgt_vocab_stoi']
    tgt_vocab_itos = arguments['model'][used_model]['tgt_vocab_itos']
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        init_field_vocab_special_tokens(source_field, src_vocab_stoi, src_vocab_itos)
        init_field_vocab_special_tokens(target_field, tgt_vocab_stoi, tgt_vocab_itos)
    else:
        init_field_vocab_special_tokens(source_field, src_vocab_stoi, src_vocab_itos, start_token, end_token, pad_token, unk_token)
        init_field_vocab_special_tokens(target_field, tgt_vocab_stoi, tgt_vocab_itos, start_token, end_token, pad_token, unk_token)

    # start test
    tester.test(model=model, test_iter=test_iter,
                  compute_predict_func=compute_predict,
                  compute_predict_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'pad_idx': pad_idx, 'max_len': max_len})


# this function needs to be defined from the view of concrete task
def compute_predict(model, data_example, device, do_log, log_string_list, source_field, target_field, pad_idx, max_len):
    # model, data_example, device, do_log, log_string_list are from inner tester framework
    # output: predict: N,L,D,  target: N,L
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    target_real = target[:, 1:-1]     # remove start_token and end_start_token, there should be no pad_token
    end_index = target[:,-1]       # dims == 1
    src_key_padding_mask = memory_key_padding_mask = gen_pad_only_mask(source, pad_idx)   #N,L
    # enc_input = model.model.encoder_embedding(source)
    enc_out = model.model.encoder(source, src_key_padding_mask=src_key_padding_mask)
    target_input = target[:, 0:1]   # initialized as start_token
    for i in range(max_len - 1):
        tgt_key_padding_mask = gen_pad_only_mask(target_input, pad_idx)  # N,L
        tgt_mask = gen_seq_only_mask(target_input, target_input)  # L,L
        # dec_input = model.model.decoder_embedding(target_input)
        dec_out = model.model.decoder(target_input, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        predict = model.model.predictor(dec_out)
        last_word = F.softmax(predict[:, -1], dim=-1).argmax(dim=-1, keepdim=True)     # last_word size: (1,1)
        if last_word.item() == end_index.item():
            break
        target_input = torch.cat([target_input, last_word], dim=-1)
    if do_log:
        log_string_list.append(
            "Source string:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:-1]))    #可能有的词会被替换成<unk>输出，因为在soti的时候在字典里查不到。
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, 1:-1]))
        log_string_list.append(
            "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, 1:-1]))
        log_string_list.append("Predict string: " + ' '.join(
            target_field.vocab.itos[index] for index in F.softmax(predict[0, :, :-1], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(
            str(index.item()) for index in F.softmax(predict[0, :, :-1], dim=-1).argmax(dim=-1)) + '\n')
    return predict[:, :-1], target_real


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
    extra_states = loaded_weights['extra_states']
    training_states = loaded_weights['training_states']
    used_model = arguments['tasks'][running_task]['model']
    arguments['model'][used_model].update({'d_model': extra_states['d_model'],
                                            'nhead': extra_states['nhead'],
                                            'src_vocab_len': extra_states['src_vocab_len'],
                                            'tgt_vocab_len': extra_states['tgt_vocab_len'],
                                            'src_vocab_stoi': extra_states['src_vocab_stoi'],
                                            'src_vocab_itos': extra_states['src_vocab_itos'],
                                            'tgt_vocab_stoi': extra_states['tgt_vocab_stoi'],
                                            'tgt_vocab_itos': extra_states['tgt_vocab_itos']
                                            })
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        bert_model_name = 'general'
    else:
        bert_model_name = 'bert'
    if (bert_model_name not in arguments['dataset'].keys()) or (
            arguments['dataset'][bert_model_name] is None):
        arguments['dataset'].update({bert_model_name: {}})
    arguments['dataset'][bert_model_name].update({'start_token': extra_states['start_token'],
                                                       'end_token': extra_states['end_token'],
                                                       'unk_token': extra_states['unk_token'],
                                                       'pad_token': extra_states['pad_token'],
                                                       'start_idx': extra_states['start_idx'],
                                                       'end_idx': extra_states['end_idx'],
                                                       'unk_idx': extra_states['unk_idx'],
                                                       'pad_idx': extra_states['pad_idx']
                                                       })
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

    return (model, model_creation_args, extra_states, training_states)