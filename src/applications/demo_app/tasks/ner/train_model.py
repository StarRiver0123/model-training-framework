import torch.nn.functional as F
from src.modules.trainer.trainer_framework import Trainer
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset import *
from build_model import *
# from src.applications.demo_app.tasks.ner.build_dataset import *
# from src.applications.demo_app.tasks.ner.build_model import NERModel


def train_model(config):
    # step 1: build dataset and vocab, vectors...
    train_iter, valid_iter = build_train_dataset_and_vocab_pipeline(config)
    # step 2: build model
    used_model = config['net_structure']['model']
    use_bert = config['model'][used_model]['use_bert']
    transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['input_seqs'])[1]
    if not use_bert:
        word_vectors = config['vector_config'][transforming_key]
    else:
        word_vectors = None
    model = TrainingModel(config, word_vectors)
    # step 3: get the trainer
    trainer = Trainer(config)
    # step 4: start train
    transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['ner_labels'])[1]
    tgt_sos_idx = config['symbol_config'][transforming_key]['sos_idx']
    tgt_eos_idx = config['symbol_config'][transforming_key]['eos_idx']
    tgt_unk_idx = config['symbol_config'][transforming_key]['unk_idx']
    tgt_pad_idx = config['symbol_config'][transforming_key]['pad_idx']
    special_token_ids = [tgt_sos_idx, tgt_eos_idx, tgt_unk_idx, tgt_pad_idx]
    labels = [tag for tag in list(config['vocab_config'][transforming_key].get_stoi().values()) if tag not in special_token_ids]
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={'pad_idx': tgt_pad_idx},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={'pad_idx': tgt_pad_idx, 'labels': labels},
                  save_model_state_func=save_model_state_func,
                  save_model_state_outer_params={})


# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, pad_idx):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if data_example[0].size(1) > max_len:
        source = data_example[0][:, :max_len].to(device)
    else:
        source = data_example[0].to(device)
    if data_example[1].size(1) > max_len:
        target = data_example[1][:, :max_len].to(device)
    else:
        target = data_example[1].to(device)

    emission = model.model.emit(seq_input=source)  # emission是3维：N,L,D

    if model.criterion == 'crf':
        mask = (target != pad_idx).byte()
        loss = -model.model.crf(emissions=emission, tags=target, mask=mask, reduction='token_mean')
        predict = model.model.crf.decode(emission, mask=mask)   # 模型输出是2层list：N，L
        # loss = -model.model.crf(emissions=emission, tags=target)
        # predict = model.model.crf.decode(emission)  # 模型输出是2层list：N，L
    else:
        # pytorch CrossEntropyLoss的输入维度有两种方式：
        # （1） input为N，C；target为N，需要对predict做reshape（-1，D_target_vocab_len）
        # （2） input为N，C，L，target为N，L。要把分类放在第二维，需要对predict进行转置transpose(-1,-2)
        logits_flatten = emission.reshape(-1, emission.size(-1))
        target_flatten = target.reshape(-1)
        loss = model.criterion(logits_flatten, target_flatten)
        predict = F.softmax(emission, dim=-1).argmax(dim=-1)
    if do_log:
        # print(model.model.bert_model.encoder.layer[11].output.dense.weight)  #打印bert模型权重，检查训练过程中参数值是否发生变化。
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, :]))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        if model.criterion == 'crf':
            log_string_list.append("Predict code:   " + ' '.join(str(index) for index in predict[0]) + '\n')
        else:
            log_string_list.append("Predict code:   " + ' '.join(str(index.item()) for index in predict[0]) + '\n')
    return emission, target, loss


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list, pad_idx, labels):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
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
    mask = (target != pad_idx).byte()
    predict = model.model.crf.decode(emission, mask=mask)
    predict_flattened = []
    for pred in predict:
        predict_flattened += pred
    target_flattened = target[mask.bool()].to('cpu').tolist()
    evaluation = model.evaluator(predict_flattened, target_flattened, labels=labels)
    if do_log:
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, 1:]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + ' '.join(str(index.item()) for index in target[0, :]))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + ' '.join(str(index) for index in predict[0]) + '\n')
    return predict, target, evaluation


def save_model_state_func(model, config):
    # model is from inner trainer framework,
    model_state_dict = model.model.state_dict()
    model_config = config['model_config']
    vocab_config = config['vocab_config']
    symbol_config = config['symbol_config']
    return {'model_state_dict': model_state_dict, 'model_config': model_config, 'vocab_config': vocab_config, 'symbol_config': symbol_config}

