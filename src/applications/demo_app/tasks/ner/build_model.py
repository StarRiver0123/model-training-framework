import torch
import torch.nn as nn
import os
from transformers import BertConfig, AdamW, get_cosine_schedule_with_warmup
from src.modules.optimizers.lr_scheduler import *
from src.modules.criteria.lsce_loss import LabelSmoothingCrossEntropyLoss
from src.modules.models.bert_crf import BertCRF
from src.modules.models.lstm_crf import LstmCRF
from src.modules.evaluators.sklearn_f1_core import SklearnF1Score


class TrainingModel(nn.Module):
    def __init__(self, config, word_vector=None):
        super().__init__()
        used_model = config['net_structure']['model']
        used_criterion = config['net_structure']['criterion']
        used_optimizer = config['net_structure']['optimizer']
        used_lr_scheduler = config['net_structure']['lr_scheduler']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        pretrained_bert_full_fine_tuning = config['training']['pretrained_bert_full_fine_tuning']
        pretrained_word_vector_full_fine_tuning = config['training']['pretrained_word_vector_full_fine_tuning']
        transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['ner_labels'])[1]
        tgt_pad_idx = config['symbol_config'][transforming_key]['pad_idx']
        num_tags = config['model_config']['vocab_len'][transforming_key]

        if (used_model == 'bert_crf'):
            bert_model_root = config['bert_model_root']
            bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            bert_config = BertConfig.from_pretrained(bert_model_name)
            self.model = BertCRF.from_pretrained(bert_model_name, config=bert_config, num_tags=num_tags).to(device)
        elif (used_model == 'lstm_crf'):
            transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['input_seqs'])[1]
            vocab_len = config['model_config']['vocab_len'][transforming_key]
            d_model = config['model_config']['d_model']
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            src_pad_idx = config['symbol_config'][transforming_key]['pad_idx']
            self.model = LstmCRF(vocab_len, d_model, hidden_size, num_layers, p_dropout, src_pad_idx, num_tags).to(device)
            if word_vector is not None:
                self.model.embedding.from_pretrained(embeddings=word_vector, freeze=not pretrained_word_vector_full_fine_tuning, padding_idx=src_pad_idx)

        if used_criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx).to(device)
        elif used_criterion == 'lsce':
            label_smoothing = config['criteria']['lsce']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=label_smoothing).to(device)
        elif used_criterion == 'crf':
            self.criterion = 'crf'

        if ((used_model == 'bert_crf') and not pretrained_bert_full_fine_tuning):
            param_optimizer = list(self.model.fc.named_parameters()) + \
                              list(self.model.crf.named_parameters())
            # list(self.model.layer_norm.named_parameters())
            opt_paras = [{'params': [p for n, p in param_optimizer]}]
        elif ((used_model == 'lstm_crf') and not pretrained_word_vector_full_fine_tuning):
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


class TestingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_model = config['model_config']['model_name']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['ner_labels'])[1]
        num_tags = config['model_config']['vocab_len'][transforming_key]

        if (used_model == 'bert_crf'):
            bert_model_root = config['bert_model_root']
            bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
            bert_config = BertConfig.from_pretrained(bert_model_name)
            self.model = BertCRF.from_pretrained(bert_model_name, config=bert_config, num_tags=num_tags).to(device)
        elif (used_model == 'lstm_crf'):
            transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['input_seqs'])[1]
            vocab_len = config['model_config']['vocab_len'][transforming_key]
            d_model = config['model_config']['d_model']
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            src_pad_idx = config['symbol_config'][transforming_key]['pad_idx']
            self.model = LstmCRF(vocab_len, d_model, hidden_size, num_layers, p_dropout, src_pad_idx, num_tags).to(device)
        self.model.eval()

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()


def create_inference_model(config):
    used_model = config['model_config']['model_name']
    device = config['general']['device']
    transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['ner_labels'])[1]
    num_tags = config['model_config']['vocab_len'][transforming_key]
    if (used_model == 'bert_crf'):
        bert_model_root = config['bert_model_root']
        bert_model_name = bert_model_root + os.path.sep + config['net_structure']['pretrained_bert_model_file']
        bert_config = BertConfig.from_pretrained(bert_model_name)
        model = BertCRF.from_pretrained(bert_model_name, config=bert_config, num_tags=num_tags).to(device)
    elif (used_model == 'lstm_crf'):
        transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['input_seqs'])[1]
        vocab_len = config['model_config']['vocab_len'][transforming_key]
        d_model = config['model_config']['d_model']
        hidden_size = config['model_config']['hidden_size']
        num_layers = config['model_config']['num_layers']
        p_dropout = config['model_config']['p_drop']
        src_pad_idx = config['symbol_config'][transforming_key]['pad_idx']
        model = LstmCRF(vocab_len, d_model, hidden_size, num_layers, p_dropout, src_pad_idx, num_tags).to(device)
    model.eval()
    return model


def create_evaluator(config):
    used_evaluator = config['net_structure']['evaluator']
    if used_evaluator == 'f1_score':
        evaluator = SklearnF1Score()
    return evaluator