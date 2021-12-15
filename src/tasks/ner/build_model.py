import torch
import torch.nn as nn
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from src.models.optimizers.lr_scheduler import WarmUpLRScheduler, CosDecayLRScheduler
from src.models.models.bert_crf import BertCRF
from src.models.evaluators.sklearn_f1_core import SklearnF1Score


class NERModel(nn.Module):
    def __init__(self, arguments):
        super().__init__()
        running_task = arguments['general']['running_task']
        project_root = arguments['general']['project_root']
        used_model = arguments['tasks'][running_task]['model']
        # used_criterion = arguments['tasks'][running_task]['criterion']
        used_optimizer = arguments['tasks'][running_task]['optimizer']
        used_lr_scheduler = arguments['tasks'][running_task]['lr_scheduler']
        used_evaluator = arguments['tasks'][running_task]['evaluator']
        num_tags = arguments['model'][used_model]['num_tags']
        device = arguments['general']['device']
        full_fine_tuning = arguments['training'][running_task]['full_fine_tuning']

        bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model']['bert_model_zh']

        if (used_model == 'bert_crf'):
            self.model = BertCRF(bert_model_name, num_tags).to(device)
            #初始化模型参数
            # for p in self.model.fc.parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_normal_(p)

        #不需要自己定义损失函数，因为crf模型直接返回Loss
        # if used_criterion == 'celoss':
        #     self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        # elif used_criterion == 'lsceloss':
        #     label_smoothing = arguments['criteria']['general']['label_smoothing']
        #     self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing).to(device)

        if full_fine_tuning:
            param_all = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            opt_paras = [
                {'params': [p for n, p in param_all if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_all if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            param_optimizer = list(self.model.fc.named_parameters()) + \
                              list(self.model.crf.named_parameters()) + \
                              list(self.model.layer_norm.named_parameters())
            opt_paras = [{'params': [p for n, p in param_optimizer]}]
        if used_optimizer == 'adam':
            lr = arguments['optimizer'][used_optimizer]['lr']
            beta1 = arguments['optimizer'][used_optimizer]['beta1']
            beta2 = arguments['optimizer'][used_optimizer]['beta2']
            eps = arguments['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps)
        elif used_optimizer == 'adamw':
            lr = arguments['optimizer'][used_optimizer]['lr']
            beta1 = arguments['optimizer'][used_optimizer]['beta1']
            beta2 = arguments['optimizer'][used_optimizer]['beta2']
            eps = arguments['optimizer'][used_optimizer]['eps']
            self.optimizer = AdamW(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps, correct_bias=False)

        if used_lr_scheduler == 'warmup':
            factor = arguments['lr_scheduler'][used_lr_scheduler]['factor']
            step_size = arguments['lr_scheduler'][used_lr_scheduler]['step_size']
            lr_warmup_step = arguments['lr_scheduler'][used_lr_scheduler]['lr_warmup_step']
            self.lr_scheduler = WarmUpLRScheduler(self.optimizer, d_model=d_model, factor=factor, step_size=step_size, warmup_step=lr_warmup_step)
        elif used_lr_scheduler == 'steplr':
            lr_step_size = arguments['lr_scheduler'][used_lr_scheduler]['step_size']
            lr_scheduler_gamma = arguments['lr_scheduler'][used_lr_scheduler]['gamma']
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_scheduler_gamma)
        elif used_lr_scheduler == 'cosdecay':
            lr_step_size = arguments['lr_scheduler'][used_lr_scheduler]['step_size']
            epochs = arguments['training'][running_task]['epochs']
            batch_size = arguments['training'][running_task]['batch_size']
            gen_num_total_examples = arguments['tasks'][running_task]['dataset']['gen_num_total_examples']
            valid_size = arguments['training'][running_task]['valid_size']
            test_size = arguments['training'][running_task]['test_size']
            train_set_size = gen_num_total_examples * (1 - test_size)*(1-valid_size)
            init_lr = arguments['lr_scheduler'][used_lr_scheduler]['init_lr']
            mini_lr = arguments['lr_scheduler'][used_lr_scheduler]['mini_lr']
            warmup_size = arguments['lr_scheduler'][used_lr_scheduler]['warmup_size']
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epoches=epochs, num_examples=train_set_size, batch_size=batch_size, init_lr=init_lr, mini_lr=mini_lr, warmup_size=warmup_size)

        if used_evaluator == 'f1_score':
            self.evaluator = SklearnF1Score()
