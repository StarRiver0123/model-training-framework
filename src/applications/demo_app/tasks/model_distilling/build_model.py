import torch
import torch.nn as nn
import os
from transformers import BertConfig, AdamW, get_cosine_schedule_with_warmup
from src.modules.optimizers.lr_scheduler import *
from src.modules.criteria.lsce_loss import LabelSmoothingCrossEntropyLoss
from src.modules.criteria.distil_loss import ResponseBasedDistilLoss
from src.modules.models.text_c import Bert_C, Lstm_C
from src.modules.evaluators.sklearn_f1_core import SklearnF1Score


class TrainingTeacherModel(nn.Module):
    def __init__(self, config, word_vector=None):
        super().__init__()
        used_model = config['net_structure']['teacher_model']
        used_criterion = config['net_structure']['criterion']
        used_optimizer = config['net_structure']['optimizer']
        used_lr_scheduler = config['net_structure']['lr_scheduler']
        used_evaluator = config['net_structure']['evaluator']
        num_classes = config['model_config']['num_classes']
        device = config['general']['device']
        pretrained_bert_full_fine_tuning = config['training']['pretrained_bert_full_fine_tuning']

        if (used_model == 'bert_c'):
            bert_model_root = config['bert_model_root']
            bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            self.model = Bert_C(bert_model_name, num_classes=num_classes).to(device)

        if used_criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss().to(device)
        elif used_criterion == 'lsce':
            label_smoothing = config['criteria']['lsce']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=None, label_smoothing=label_smoothing).to(device)

        if ((used_model == 'bert_c') and not pretrained_bert_full_fine_tuning):
            param_optimizer = None
            opt_paras = [{'params': [p for n, p in param_optimizer]}]
        else:
            param_all = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            opt_paras = [
                {'params': [p for n, p in param_all if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_all if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        if used_optimizer == 'adam':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps)
        elif used_optimizer == 'adamw':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = AdamW(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps, correct_bias=False)


        if used_lr_scheduler == 'steplr':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            lr_scheduler_gamma = config['lr_scheduler'][used_lr_scheduler]['gamma']
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_scheduler_gamma)
        elif used_lr_scheduler == 'cosdecay':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            epochs = config['training']['epochs']
            batch_size = config['training']['batch_size']
            train_set_size = config['net_structure']['dataset']['train_set_size']
            min_lr = config['lr_scheduler'][used_lr_scheduler]['min_lr']
            warmup_size = config['lr_scheduler'][used_lr_scheduler]['warmup_size']
            # self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epochs=epochs, num_examples=train_set_size, batch_size=batch_size, min_lr=min_lr, warmup_size=warmup_size)
            train_steps_per_epoch = train_set_size // batch_size
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=epochs * train_steps_per_epoch)
        else:
            self.lr_scheduler = None

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


class TestingTeacherModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_model = config['model_config']['model_name']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        num_classes = config['model_config']['num_classes']

        if (used_model == 'bert_c'):
            bert_model_root = config['bert_model_root']
            bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            self.model = Bert_C(bert_model_name, num_classes=num_classes).to(device)
        self.model.eval()

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


def create_teacher_inference_model(config):
    try:
        used_model = config['net_structure']['teacher_model']
    except KeyError:
        used_model = config['model_config']['model_name']
    device = config['general']['device']
    num_classes = config['model_config']['num_classes']
    if (used_model == 'bert_c'):
        bert_model_root = config['bert_model_root']
        bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
        model = Bert_C(bert_model_name, num_classes=num_classes).to(device)
    model.eval()
    return model


class TrainingPureStudentModel(nn.Module):
    def __init__(self, config, word_vector=None):
        super().__init__()
        used_model = config['net_structure']['student_model']
        used_criterion = config['net_structure']['criterion']
        used_optimizer = config['net_structure']['optimizer']
        used_lr_scheduler = config['net_structure']['lr_scheduler']
        used_evaluator = config['net_structure']['evaluator']
        num_classes = config['model_config']['num_classes']
        device = config['general']['device']
        pretrained_word_vector_full_fine_tuning = config['training']['pretrained_word_vector_full_fine_tuning']

        if (used_model == 'lstm_c'):
            training_whom = config['net_structure']['training_whom']
            transforming_key = eval(config['train_text_transforming_adaptor'][training_whom]['input_for_student'])[1]
            vocab_len = config['model_config']['vocab_len'][transforming_key]
            d_model = config['model_config']['d_model']
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            pad_idx = config['symbol_config'][transforming_key]['pad_idx']
            self.model = Lstm_C(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_classes).to(device)
            if word_vector is not None:
                self.model.embedding.from_pretrained(embeddings=word_vector, freeze=not pretrained_word_vector_full_fine_tuning, padding_idx=pad_idx)
            # else:
            #初始化模型参数
            # for p in self.model.fc.parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_normal_(p)

        if used_criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss().to(device)
        elif used_criterion == 'lsce':
            label_smoothing = config['criteria']['lsce']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=None, label_smoothing=label_smoothing).to(device)

        if (not pretrained_word_vector_full_fine_tuning):
            param_optimizer = list(self.model.named_parameters())
            opt_paras = [{'params': [p for n, p in param_optimizer if n != 'embddding']}]
        else:
            param_all = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            opt_paras = [
                {'params': [p for n, p in param_all if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_all if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        if used_optimizer == 'adam':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps)
        elif used_optimizer == 'adamw':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = AdamW(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps, correct_bias=False)


        if used_lr_scheduler == 'warmup':
            factor = config['lr_scheduler'][used_lr_scheduler]['factor']
            step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            lr_warmup_step = config['lr_scheduler'][used_lr_scheduler]['lr_warmup_step']
            self.lr_scheduler = WarmUpLRScheduler(self.optimizer, d_model=d_model, factor=factor, step_size=step_size, warmup_step=lr_warmup_step)
        elif used_lr_scheduler == 'steplr':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            lr_scheduler_gamma = config['lr_scheduler'][used_lr_scheduler]['gamma']
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_scheduler_gamma)
        elif used_lr_scheduler == 'cosdecay':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            epochs = config['training']['epochs']
            batch_size = config['training']['batch_size']
            train_set_size = config['net_structure']['dataset']['train_set_size']
            min_lr = config['lr_scheduler'][used_lr_scheduler]['min_lr']
            warmup_size = config['lr_scheduler'][used_lr_scheduler]['warmup_size']
            # self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epochs=epochs, num_examples=train_set_size, batch_size=batch_size, min_lr=min_lr, warmup_size=warmup_size)
            train_steps_per_epoch = train_set_size // batch_size
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=epochs * train_steps_per_epoch)
        elif used_lr_scheduler == 'self_adjusting':
            epochs = config['training']['epochs']
            batch_size = config['training']['batch_size']
            train_set_size = config['net_structure']['dataset']['train_set_size']
            mean_loss_window = config['lr_scheduler'][used_lr_scheduler]['mean_loss_window']
            cross_mean = config['lr_scheduler'][used_lr_scheduler]['cross_mean']
            adjusting_ratio = config['lr_scheduler'][used_lr_scheduler]['adjusting_ratio']
            warmup_size = config['lr_scheduler'][used_lr_scheduler]['warmup_size']
            smoothing_zero = config['lr_scheduler'][used_lr_scheduler]['smoothing_zero']
            self.lr_scheduler = SelfAdjustingAfterWarmUpLRScheduler(self.optimizer, mean_loss_window=mean_loss_window, cross_mean=cross_mean, adjusting_ratio=adjusting_ratio, epochs=epochs, num_examples=train_set_size, batch_size=batch_size, warmup_size=warmup_size, smoothing_zero=smoothing_zero)
        else:
            self.lr_scheduler = None

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


class TestingPureStudentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_model = config['model_config']['model_name']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        num_classes = config['model_config']['num_classes']

        if (used_model == 'lstm_c'):
            training_whom = config['model_config']['training_whom']
            transforming_key = eval(config['test_text_transforming_adaptor'][training_whom]['input_for_student'])[1]
            vocab_len = config['model_config']['vocab_len'][transforming_key]
            d_model = config['model_config']['d_model']
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            pad_idx = config['symbol_config'][transforming_key]['pad_idx']
            self.model = Lstm_C(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_classes).to(device)
        self.model.eval()

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


class TrainingDistilledStudentModel(nn.Module):
    def __init__(self, config, word_vector=None):
        super().__init__()
        used_teacher_model = config['net_structure']['teacher_model']
        used_student_model = config['net_structure']['student_model']
        used_criterion = config['net_structure']['criterion']
        used_optimizer = config['net_structure']['optimizer']
        used_lr_scheduler = config['net_structure']['lr_scheduler']
        used_evaluator = config['net_structure']['evaluator']
        num_classes = config['model_config']['num_classes']
        device = config['general']['device']
        pretrained_word_vector_full_fine_tuning = config['training']['pretrained_word_vector_full_fine_tuning']

        if (used_teacher_model == 'bert_c'):
            # create the model:
            bert_model_root = config['bert_model_root']
            bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            self.teacher_model = Bert_C(bert_model_name, num_classes=num_classes).to(device)
            self.teacher_model.eval()
            # load the model states:
            teacher_model_file_path = config['resource_root'] + os.path.sep + config['net_structure']['trained_teacher_model_file']
            teacher_model_states = torch.load(teacher_model_file_path)
            self.teacher_model.load_state_dict(teacher_model_states['model_state_dict'])
        if (used_student_model == 'lstm_c'):
            training_whom = config['net_structure']['training_whom']
            transforming_key = eval(config['train_text_transforming_adaptor'][training_whom]['input_for_student'])[1]
            vocab_len = config['model_config']['vocab_len'][transforming_key]
            d_model = config['model_config']['d_model']
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            pad_idx = config['symbol_config'][transforming_key]['pad_idx']
            self.model = Lstm_C(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_classes).to(device)
            if word_vector is not None:
                self.model.embedding.from_pretrained(embeddings=word_vector, freeze=not pretrained_word_vector_full_fine_tuning, padding_idx=pad_idx)

        if used_criterion == 'response_based_distil_loss':
            soft_loss_type = config['criteria']['response_based_distil_loss']['soft_loss_type']
            T = config['criteria']['response_based_distil_loss']['T']
            alpha = config['criteria']['response_based_distil_loss']['alpha']
            self.criterion = ResponseBasedDistilLoss(soft_loss_type, T, alpha).to(device)

        if (not pretrained_word_vector_full_fine_tuning):
            param_optimizer = list(self.model.named_parameters())
            opt_paras = [{'params': [p for n, p in param_optimizer if n != 'embddding']}]
        else:
            param_all = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            opt_paras = [
                {'params': [p for n, p in param_all if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_all if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        if used_optimizer == 'adam':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps)
        elif used_optimizer == 'adamw':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = AdamW(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps, correct_bias=False)

        if used_lr_scheduler == 'cosdecay':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            epochs = config['training']['epochs']
            batch_size = config['training']['batch_size']
            train_set_size = config['net_structure']['dataset']['train_set_size']
            min_lr = config['lr_scheduler'][used_lr_scheduler]['min_lr']
            warmup_size = config['lr_scheduler'][used_lr_scheduler]['warmup_size']
            # self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epochs=epochs, num_examples=train_set_size, batch_size=batch_size, min_lr=min_lr, warmup_size=warmup_size)
            train_steps_per_epoch = train_set_size // batch_size
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=epochs * train_steps_per_epoch)
        else:
            self.lr_scheduler = None

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


class TestingDistilledStudentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_model = config['model_config']['model_name']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        num_classes = config['model_config']['num_classes']

        if (used_model == 'lstm_c'):
            training_whom = config['model_config']['training_whom']
            transforming_key = eval(config['test_text_transforming_adaptor'][training_whom]['input_for_student'])[1]
            vocab_len = config['model_config']['vocab_len'][transforming_key]
            d_model = config['model_config']['d_model']
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            pad_idx = config['symbol_config'][transforming_key]['pad_idx']
            self.model = Lstm_C(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_classes).to(device)
        self.model.eval()

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


def create_distilled_student_inference_model(config):
    used_model = config['model_config']['model_name']
    device = config['general']['device']
    num_classes = config['model_config']['num_classes']
    if (used_model == 'lstm_c'):
        training_whom = config['model_config']['training_whom']
        transforming_key = eval(config['test_text_transforming_adaptor'][training_whom]['input_for_student'])[1]
        vocab_len = config['model_config']['vocab_len'][transforming_key]
        d_model = config['model_config']['d_model']
        hidden_size = config['model_config']['hidden_size']
        num_layers = config['model_config']['num_layers']
        p_dropout = config['model_config']['p_drop']
        pad_idx = config['symbol_config'][transforming_key]['pad_idx']
        model = Lstm_C(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_classes).to(device)
    model.eval()
    return model


def create_evaluator(config):
    used_evaluator = config['net_structure']['evaluator']
    if used_evaluator == 'f1_score':
        evaluator = SklearnF1Score()
    return evaluator