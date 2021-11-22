import torch
import torch.nn as nn
from src.models.models.starriver_transformer import StarRiverTransformer
from src.models.models.torch_transformer import TorchTransformer
from src.models.criteria.lsce_loss import LabelSmoothingCrossEntropyLoss
from src.models.optimizers.lr_scheduler import WarmUpLRScheduler, CosDecayLRScheduler
from src.models.evaluators.bleu import TranslationBleuScore


class TranslatorModel():
    def __init__(self, arguments, src_vector=None, tgt_vector=None, src_bert_model=None, tgt_bert_model=None):
        running_task = arguments['general']['running_task']
        used_model = arguments['tasks'][running_task]['model']
        used_criterion = arguments['tasks'][running_task]['criterion']
        used_optimizer = arguments['tasks'][running_task]['optimizer']
        used_lr_scheduler = arguments['tasks'][running_task]['lr_scheduler']
        used_evaluator = arguments['tasks'][running_task]['evaluator']
        d_model = arguments['model'][used_model]['d_model']
        use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
        device = arguments['general']['device']
        src_vocab_len = arguments['model'][used_model]['src_vocab_len']
        tgt_vocab_len = arguments['model'][used_model]['tgt_vocab_len']
        if (use_bert != 'static') and (use_bert != 'dynamic'):
            end_idx = arguments['dataset']['general']['end_idx']
            pad_idx = arguments['dataset']['general']['pad_idx']
        else:
            end_idx = arguments['dataset']['bert']['end_idx']
            pad_idx = arguments['dataset']['bert']['pad_idx']

        if (used_model == 'starriver_transformer') or (used_model == 'torch_transformer'):
            nhead = arguments['model'][used_model]['nhead']
            d_ff = arguments['model'][used_model]['d_ff']
            max_len = arguments['model'][used_model]['max_len']
            num_encoder_layers = arguments['model'][used_model]['num_encoder_layers']
            num_decoder_layers = arguments['model'][used_model]['num_decoder_layers']
            p_drop = arguments['model'][used_model]['p_drop']
            if used_model == 'torch_transformer':
                self.model = TorchTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len).to(device)
            else:
                self.model = StarRiverTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len).to(device)
            #初始化模型参数
            if (src_vector is not None) and (tgt_vector is not None) or (src_bert_model is not None) and (tgt_bert_model is not None):
                for p in self.model.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_normal_(p)
            #初始化词向量， 如果相关传入参数为空，则不做初始化。
            if (use_bert != 'static') and (use_bert != 'dynamic'):
                if (src_vector is not None) and (tgt_vector is not None):
                    # 注意如果requires_grad==True, 那么对self.model.encoder_embedding.embedding.weight.data的赋值最好是用copy_()函数，否则会把源头的向量给修改了。
                    self.model.encoder_embedding.embedding.weight.data = src_vector.to(device)
                    self.model.encoder_embedding.embedding.weight.requires_grad = False
                    self.model.decoder_embedding.embedding.weight.data = tgt_vector.to(device)
                    self.model.decoder_embedding.embedding.weight.requires_grad = False
            elif use_bert == 'static':    # static vector of bert
                if (src_bert_model is not None) and (tgt_bert_model is not None):
                    # 注意如果requires_grad==True, 那么对self.model.encoder_embedding.embedding.weight.data的赋值最好是用copy_()函数，否则会把源头的向量给修改了。
                    self.model.encoder_embedding.embedding.weight.data = src_bert_model.get_input_embeddings().weight.data.to(device)
                    self.model.encoder_embedding.embedding.weight.requires_grad = False
                    self.model.decoder_embedding.embedding.weight.data = tgt_bert_model.get_input_embeddings().weight.data.to(device)
                    self.model.decoder_embedding.embedding.weight.requires_grad = False
            else:     # dynamic vector of bert
                if (src_bert_model is not None) and (tgt_bert_model is not None):
                    self.src_bert_model = src_bert_model.to(device)
                    self.tgt_bert_model = tgt_bert_model.to(device)

        if used_criterion == 'celoss':
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        elif used_criterion == 'lsceloss':
            label_smoothing = arguments['criteria']['general']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing).to(device)

        if used_optimizer == 'adam':
            lr = arguments['optimizer'][used_optimizer]['lr']
            beta1 = arguments['optimizer'][used_optimizer]['beta1']
            beta2 = arguments['optimizer'][used_optimizer]['beta2']
            eps = arguments['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1,beta2), eps=eps)

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
            train_set_size = arguments['tasks'][running_task]['dataset']['train_set_size']
            valid_size = arguments['training'][running_task]['valid_size']
            init_lr = arguments['lr_scheduler'][used_lr_scheduler]['init_lr']
            mini_lr = arguments['lr_scheduler'][used_lr_scheduler]['mini_lr']
            warmup_size = arguments['lr_scheduler'][used_lr_scheduler]['warmup_size']
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epoches=epochs, num_examples=train_set_size*(1-valid_size), batch_size=batch_size, init_lr=init_lr, mini_lr=mini_lr, warmup_size=warmup_size)

        if used_evaluator == 'bleu':
            self.evaluator = TranslationBleuScore(end_index=end_idx, pad_index=pad_idx)
