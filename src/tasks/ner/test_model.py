import torch
import numpy as np
import importlib
import torch.nn.functional as F
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.models.tester.tester_framework import Tester
from src.tasks.ner.build_dataset import load_test_set, get_data_iterator, init_field_vocab_special_tokens


def test_model(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    max_len = arguments['model'][used_model]['max_len']
    # get the dataset and data iterator
    test_set = load_test_set(arguments)
    test_iter, source_field, target_field = get_data_iterator(arguments, test_set=test_set)
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_vocab, _, _, _ = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    # re-init the field vocab
    tgt_vocab_stoi = model_vocab['tgt_vocab_stoi']
    tgt_vocab_itos = model_vocab['tgt_vocab_itos']
    init_field_vocab_special_tokens(target_field, tgt_vocab_stoi, tgt_vocab_itos)
    special_token_ids = [target_field.vocab.stoi[target_field.init_token], target_field.vocab.stoi[target_field.eos_token], target_field.vocab.stoi[target_field.pad_token], target_field.vocab.stoi[target_field.unk_token]]
    labels = [tag for tag in list(target_field.vocab.stoi.values()) if tag not in special_token_ids]
    # start test
    tester.test(model=model, test_iter=test_iter,
                  compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={
                      'source_field': source_field, 'target_field': target_field, 'labels': labels})


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, device, do_log, log_string_list, source_field, target_field, labels):
    # model, data_example, device, do_log, log_string_list are from inner tester framework
    # output: predict: N,L,D,  target: N,L
    source = data_example.Source.to(device)
    target = data_example.Target.to(device)
    emission = model.model.bert_emit(seq_input=source)
    mask = (target != target_field.vocab.stoi[target_field.pad_token]).byte().to(device)
    predict = model.model.crf.decode(emission, mask=mask)
    predict_flattened = np.array(predict).reshape(-1).tolist()
    target_flattened = target.reshape(-1).to('cpu').tolist()
    evaluation = model.evaluator(predict_flattened, target_flattened, labels=labels)

    if do_log:
        # log_string_list.append(
        #     "Source string:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, :]))    #可能有的词会被替换成<unk>输出，因为在soti的时候在字典里查不到。
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:-1]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, :]))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(predict[0, :, :-1], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(str(index) for index in predict[0]) + '\n')
    return predict, target, evaluation


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
    # arguments['model'][used_model].update({'tgt_vocab_stoi': model_vocab['tgt_vocab_stoi'],
    #                                        'tgt_vocab_itos': model_vocab['tgt_vocab_itos']
    #                                        })
    arguments['model'][used_model].update({'num_tags': extra_states['num_tags']})
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