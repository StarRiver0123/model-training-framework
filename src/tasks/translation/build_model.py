import torch
import torch.nn as nn
from src.models.models.starriver_transformer import StarRiverTransformer
from src.models.models.torch_transformer import TorchTransformer
from src.models.criteria.lsce_loss import LabelSmoothingCrossEntropyLoss
from src.models.optimizers.lr_scheduler import WarmUpLRScheduler, CosDecayLRScheduler
from src.models.evaluators.bleu import TranslationBleuScore


class TranslatorModel(nn.Module):
    def __init__(self, arguments, src_vector=None, tgt_vector=None, src_bert_model=None, tgt_bert_model=None):
        super().__init__()
        running_task = arguments['general']['running_task']
        used_model = arguments['tasks'][running_task]['model']
        used_criterion = arguments['tasks'][running_task]['criterion']
        used_optimizer = arguments['tasks'][running_task]['optimizer']
        used_lr_scheduler = arguments['tasks'][running_task]['lr_scheduler']
        used_evaluator = arguments['tasks'][running_task]['evaluator']
        d_model = arguments['model'][used_model]['d_model']
        use_bert = arguments['tasks'][running_task]['use_bert']
        bert_full_fine_tuning = arguments['training'][running_task]['bert_full_fine_tuning']
        device = arguments['general']['device']
        src_vocab_len = arguments['model'][used_model]['src_vocab_len']
        tgt_vocab_len = arguments['model'][used_model]['tgt_vocab_len']
        if use_bert not in ['static', 'dynamic']:
            eos_idx = arguments['dataset']['general']['eos_idx']
            pad_idx = arguments['dataset']['general']['pad_idx']
        else:
            eos_idx = arguments['dataset']['bert']['eos_idx']
            pad_idx = arguments['dataset']['bert']['pad_idx']

        if used_model in ['starriver_transformer', 'torch_transformer']:
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
            if use_bert not in ['static', 'dynamic']:
                if (src_vector is not None) and (tgt_vector is not None):
                    self.model.encoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=src_vector, freeze=False, padding_idx=pad_idx).to(device)
                    self.model.decoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=tgt_vector, freeze=False, padding_idx=pad_idx).to(device)
                    # self.model.encoder_token_embedding.embedding.weight.data.copy_(src_vector).to(device)
                    # self.model.encoder_token_embedding.embedding.weight.requires_grad = True
                    # self.model.decoder_token_embedding.embedding.weight.data.copy_(tgt_vector).to(device)
                    # self.model.decoder_token_embedding.embedding.weight.requires_grad = True
            elif use_bert == 'static':    # static vector of bert
                if (src_bert_model is not None) and (tgt_bert_model is not None):
                    self.model.encoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=src_bert_model.get_input_embeddings().weight.data, freeze=False,
                                                                           padding_idx=pad_idx).to(device)
                    self.model.decoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=tgt_bert_model.get_input_embeddings().weight.data, freeze=False,
                                                                           padding_idx=pad_idx).to(device)
                    # 注意如果requires_grad==True, 那么对self.model.encoder_token_embedding.embedding.weight.data的赋值最好是用copy_()函数，直接赋值会有问题？否则会把源头的向量给修改了。
                    # self.model.encoder_token_embedding.embedding.weight.data.copy_(src_bert_model.get_input_embeddings().weight.data).to(device)
                    # self.model.encoder_token_embedding.embedding.weight.requires_grad = True
                    # self.model.decoder_token_embedding.embedding.weight.data.copy_(tgt_bert_model.get_input_embeddings().weight.data).to(device)
                    # self.model.decoder_token_embedding.embedding.weight.requires_grad = True
            else:     # dynamic vector of bert
                if (src_bert_model is not None) and (tgt_bert_model is not None):
                    self.model.encoder_token_embedding = None
                    self.model.decoder_token_embedding = None
                    self.model.src_bert_model = src_bert_model.to(device)
                    self.model.tgt_bert_model = tgt_bert_model.to(device)

        if used_criterion == 'celoss':
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        elif used_criterion == 'lsceloss':
            label_smoothing = arguments['criteria']['general']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing).to(device)

        # 设置需要更新的参数，bert模型不做微调，只是取它的动态词向量。
        if (use_bert != 'dynamic') or (use_bert == 'dynamic') and bert_full_fine_tuning:
            param_optimizer = self.model.parameters()
        else:
            if used_model == 'starriver_transformer':
                param_optimizer = list(self.model.encoder_token_embedding.parameters()) + \
                                  list(self.model.decoder_token_embedding.parameters()) + \
                                  list(self.model.encoder_layers.parameters()) + \
                                  list(self.model.decoder_layers.parameters()) + \
                                  list(self.model.predictor.parameters())
            else:
                param_optimizer = list(self.model.transformer.parameters()) + \
                                  list(self.model.predictor.parameters())
        if used_optimizer == 'adam':
            lr = arguments['optimizer'][used_optimizer]['lr']
            beta1 = arguments['optimizer'][used_optimizer]['beta1']
            beta2 = arguments['optimizer'][used_optimizer]['beta2']
            eps = arguments['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(param_optimizer, lr=lr, betas=(beta1, beta2), eps=eps)

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
            if ('train_set_size' not in arguments['tasks'][running_task]['dataset'].keys()) or (arguments['tasks'][running_task]['dataset']['train_set_size'] is None):
                train_set_size = 1000     # 只是为了测试阶段不报错，数量随便写。
            else:
                train_set_size = arguments['tasks'][running_task]['dataset']['train_set_size']
            init_lr = arguments['lr_scheduler'][used_lr_scheduler]['init_lr']
            mini_lr = arguments['lr_scheduler'][used_lr_scheduler]['mini_lr']
            warmup_size = arguments['lr_scheduler'][used_lr_scheduler]['warmup_size']
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epoches=epochs, num_examples=train_set_size, batch_size=batch_size, init_lr=init_lr, mini_lr=mini_lr, warmup_size=warmup_size)

        if used_evaluator == 'bleu':
            self.evaluator = TranslationBleuScore(end_index=eos_idx, pad_index=pad_idx)
