import numpy as np
import torch.nn.functional as F
from src.utilities.load_data import *
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
    training_whom = config['net_structure']['training_whom']
    if training_whom == 'teacher':
        transforming_key = eval(config['train_text_transforming_adaptor'][training_whom]['input_for_teacher'])[1]
        main_model = config['net_structure']['teacher_model']
    elif training_whom in ['pure_student', 'distilled_student']:
        transforming_key = eval(config['train_text_transforming_adaptor'][training_whom]['input_for_student'])[1]
        main_model = config['net_structure']['student_model']
    use_bert = config['model'][main_model]['use_bert']
    if not use_bert:
        word_vectors = config['vector_config'][transforming_key]
    else:
        word_vectors = None
    if training_whom == 'teacher':
        model = TrainingTeacherModel(config, word_vectors)
    elif training_whom == 'pure_student':
        model = TrainingPureStudentModel(config, word_vectors)
    elif training_whom == 'distilled_student':
        model = TrainingDistilledStudentModel(config, word_vectors)
    # step 3: get the trainer
    trainer = Trainer(config)
    # step 4: start train
    trainer.train(model=model, train_iter=train_iter,
                  compute_predict_loss_func=compute_predict_loss,
                  compute_predict_loss_outer_params={'training_whom': training_whom},
                  valid_iter=valid_iter, compute_predict_evaluation_func=compute_predict_evaluation,
                  compute_predict_evaluation_outer_params={}, save_model_state_func=save_model_state_func,
                  save_model_state_outer_params={})


# this function needs to be defined from the view of concrete task
def compute_predict_loss(model, data_example, max_len, device, do_log, log_string_list, training_whom):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if training_whom != 'distilled_student':
        if data_example[0].size(1) > max_len:
            source = data_example[0][:, :max_len].to(device)
        else:
            source = data_example[0].to(device)
        target = data_example[1].to(device)
        logits = model.model(source).logits  # logits是2维：N,D
        # pytorch CrossEntropyLoss的输入维度有两种方式：
        # （1） input为N，C；target为N，需要对predict做reshape（-1，D_target_vocab_len）
        # （2） input为N，C，L，target为N，L。要把分类放在第二维，需要对predict进行转置transpose(-1,-2)
        loss = model.criterion(logits, target)
    else:
        if data_example[0].size(1) > max_len:
            teacher_source = data_example[0][:, :max_len].to(device)
        else:
            teacher_source = data_example[0].to(device)
        if data_example[1].size(1) > max_len:
            source = data_example[1][:, :max_len].to(device)
        else:
            source = data_example[1].to(device)
        target = data_example[2].to(device)
        teacher_logits = model.teacher_model(teacher_source).logits
        logits = model.model(source).logits
        loss = model.criterion(logits, teacher_logits, target)

    predict = F.softmax(logits, dim=-1).argmax(dim=-1)
    if do_log:
        # print(model.model.bert_model.encoder.layer[11].output.dense.weight)  #打印bert模型权重，检查训练过程中参数值是否发生变化。
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, :]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + str(target[0].item()))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + str(predict[0].item()) + '\n')
    return logits, target, loss


# this function needs to be defined from the view of concrete task
def compute_predict_evaluation(model, data_example, max_len, device, do_log, log_string_list):
    # model, data_example, device, do_log, log_string_list are from inner trainer framework
    # output: predict: N,L,D,  target: N,L
    if data_example[0].size(1) > max_len:
        source = data_example[0][:, :max_len].to(device)
    else:
        source = data_example[0].to(device)
    target = data_example[1].to('cpu')
    logits = model.model(source).logits
    predict = F.softmax(logits, dim=-1).argmax(dim=-1).to('cpu')
    evaluation = model.evaluator(predict, target)
    if do_log:
        # log_string_list.append(
        #     "Source words:  " + ' '.join(source_field.vocab.itos[index] for index in source[0, :]))
        log_string_list.append("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
        # log_string_list.append(
        #     "Target string:  " + ' '.join(target_field.vocab.itos[index] for index in target[0, 1:]))
        log_string_list.append("Target code:    " + str(target[0].item()))
        # log_string_list.append("Predict string: " + ' '.join(
        #     target_field.vocab.itos[index] for index in F.softmax(logit[0, :, :], dim=-1).argmax(dim=-1)))
        log_string_list.append("Predict code:   " + str(predict[0].item()) + '\n')
    return predict, target, evaluation


def save_model_state_func(model, config):
    # model is from inner trainer framework,
    model_state_dict = model.model.state_dict()
    model_config = config['model_config']
    vocab_config = config['vocab_config']
    symbol_config = config['symbol_config']
    return {'model_state_dict': model_state_dict, 'model_config': model_config, 'vocab_config': vocab_config, 'symbol_config': symbol_config}

