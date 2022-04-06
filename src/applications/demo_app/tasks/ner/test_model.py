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
    # step 3: get the data iterator
    test_iter = build_test_dataset_pipeline(config)
    # step 4: get the model
    model = load_model_and_vocab(config)
    # step 5: start test
    used_model = config['model_config']['model_name']
    transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['ner_labels'])[1]
    tgt_sos_idx = config['symbol_config'][transforming_key]['sos_idx']
    tgt_eos_idx = config['symbol_config'][transforming_key]['eos_idx']
    tgt_unk_idx = config['symbol_config'][transforming_key]['unk_idx']
    tgt_pad_idx = config['symbol_config'][transforming_key]['pad_idx']
    special_token_ids = [tgt_sos_idx, tgt_eos_idx, tgt_unk_idx, tgt_pad_idx]
    labels = [tag for tag in list(config['vocab_config'][transforming_key].get_stoi().values()) if
              tag not in special_token_ids]
    tester.test(model=model, test_iter=test_iter,
                  compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'pad_idx': tgt_pad_idx, 'labels': labels})



# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, pad_idx, labels):
    # model, data_example, device, do_log, log_string_list are from inner tester framework
    # output: predict: N,L,D,  target: N,L
    if data_example[0].size(1) > max_len:
        source = data_example[0][:, :max_len].to(device)
    else:
        source = data_example[0].to(device)
    if data_example[1].size(1) > max_len:
        target = data_example[1][:, :max_len].to(device)
    else:
        target = data_example[1].to(device)
    emission = model.model.emit(seq_input=source)
    mask = (target != pad_idx).byte().to(device)
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

