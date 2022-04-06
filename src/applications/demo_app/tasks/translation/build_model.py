import torch
import torch.nn as nn
import os
from src.modules.models.starriver_transformer import StarRiverTransformer
from src.modules.models.torch_transformer import TorchTransformer
from src.modules.criteria.lsce_loss import LabelSmoothingCrossEntropyLoss
from src.modules.optimizers.lr_scheduler import WarmUpLRScheduler, CosDecayLRScheduler
from src.modules.evaluators.bleu import TranslationBleuScore


class TrainingModel(nn.Module):
    def __init__(self, config, src_vector=None, tgt_vector=None):
        super().__init__()
        used_model = config['net_structure']['model']
        used_criterion = config['net_structure']['criterion']
        used_optimizer = config['net_structure']['optimizer']
        used_lr_scheduler = config['net_structure']['lr_scheduler']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        pretrained_word_vector_full_fine_tuning = config['training']['pretrained_word_vector_full_fine_tuning']
        src_transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['source_seqs'])[1]
        tgt_transforming_key = eval(config['train_text_transforming_adaptor'][used_model]['target_seqs'])[1]
        src_pad_idx = config['symbol_config'][src_transforming_key]['pad_idx']
        tgt_pad_idx = config['symbol_config'][tgt_transforming_key]['pad_idx']
        tgt_eos_idx = config['symbol_config'][tgt_transforming_key]['eos_idx']
        src_vocab_len = config['model_config']['vocab_len'][src_transforming_key]
        tgt_vocab_len = config['model_config']['vocab_len'][tgt_transforming_key]

        d_model = config['model'][used_model]['d_model']
        nhead = config['model'][used_model]['nhead']
        d_ff = config['model'][used_model]['d_ff']
        max_len = config['model'][used_model]['max_len']
        num_encoder_layers = config['model'][used_model]['num_encoder_layers']
        num_decoder_layers = config['model'][used_model]['num_decoder_layers']
        p_drop = config['model'][used_model]['p_drop']
        if used_model == 'torch_transformer':
            self.model = TorchTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len).to(device)
        elif used_model == 'starriver_transformer':
            self.model = StarRiverTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len).to(device)
        #初始化模型参数
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        #初始化词向量， 如果相关传入参数为空，则不做初始化。
        if (src_vector is not None) and (tgt_vector is not None):
            self.model.encoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=src_vector, freeze=not pretrained_word_vector_full_fine_tuning, padding_idx=src_pad_idx).to(device)
            self.model.decoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=tgt_vector, freeze=not pretrained_word_vector_full_fine_tuning, padding_idx=tgt_pad_idx).to(device)

        if used_criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx).to(device)
        elif used_criterion == 'lsce':
            label_smoothing = config['criteria']['lsce']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=label_smoothing).to(device)

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
            self.optimizer = torch.optim.AdamW(opt_paras, lr=lr, betas=(beta1, beta2), eps=eps)

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
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epochs=epochs, num_examples=train_set_size, batch_size=batch_size, min_lr=min_lr, warmup_size=warmup_size)
        else:
            self.lr_scheduler = None

        if used_evaluator == 'bleu':
            self.evaluator = TranslationBleuScore(end_index=tgt_eos_idx, pad_index=tgt_pad_idx)


class TestingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        used_model = config['model_config']['model_name']
        d_model = config['model_config']['d_model']
        src_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['source_seqs'])[1]
        tgt_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['target_seqs'])[1]
        tgt_pad_idx = config['symbol_config'][tgt_transforming_key]['pad_idx']
        tgt_eos_idx = config['symbol_config'][tgt_transforming_key]['eos_idx']
        src_vocab_len = config['model_config']['vocab_len'][src_transforming_key]
        tgt_vocab_len = config['model_config']['vocab_len'][tgt_transforming_key]
        nhead = config['model_config']['nhead']
        d_ff = config['model_config']['d_ff']
        max_len = config['model_config']['max_len']
        num_encoder_layers = config['model_config']['num_encoder_layers']
        num_decoder_layers = config['model_config']['num_decoder_layers']
        p_drop = config['model_config']['p_drop']
        if used_model == 'torch_transformer':
            self.model = TorchTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop,
                                          src_vocab_len, tgt_vocab_len).to(device)
        elif used_model == 'starriver_transformer':
            self.model = StarRiverTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers,
                                              p_drop, src_vocab_len, tgt_vocab_len).to(device)

        if used_evaluator == 'bleu':
            self.evaluator = TranslationBleuScore(end_index=tgt_eos_idx, pad_index=tgt_pad_idx)


def create_inference_model(config):
    device = config['general']['device']
    used_model = config['model_config']['model_name']
    d_model = config['model_config']['d_model']
    src_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['source_seqs'])[1]
    tgt_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['target_seqs'])[1]
    src_vocab_len = config['model_config']['vocab_len'][src_transforming_key]
    tgt_vocab_len = config['model_config']['vocab_len'][tgt_transforming_key]
    nhead = config['model_config']['nhead']
    d_ff = config['model_config']['d_ff']
    max_len = config['model_config']['max_len']
    num_encoder_layers = config['model_config']['num_encoder_layers']
    num_decoder_layers = config['model_config']['num_decoder_layers']
    p_drop = config['model_config']['p_drop']
    if used_model == 'torch_transformer':
        model = TorchTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop,
                                      src_vocab_len, tgt_vocab_len).to(device)
    elif used_model == 'starriver_transformer':
        model = StarRiverTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop,
                                          src_vocab_len, tgt_vocab_len).to(device)
    model.eval()
    return model


def create_evaluator(config):
    used_evaluator = config['net_structure']['evaluator']
    used_model = config['model_config']['model_name']
    src_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['source_seqs'])[1]
    tgt_transforming_key = eval(config['test_text_transforming_adaptor'][used_model]['target_seqs'])[1]
    tgt_pad_idx = config['symbol_config'][tgt_transforming_key]['pad_idx']
    tgt_eos_idx = config['symbol_config'][tgt_transforming_key]['eos_idx']
    if used_evaluator == 'bleu':
        evaluator = TranslationBleuScore(end_index=tgt_eos_idx, pad_index=tgt_pad_idx)
    return evaluator