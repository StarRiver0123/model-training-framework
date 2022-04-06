import torch
import torch.nn as nn
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
        d_model = config['model'][used_model]['d_model']
        device = config['general']['device']
        src_vocab_len = config['model_config']['src_vocab_len']
        tgt_vocab_len = config['model_config']['tgt_vocab_len']
        eos_idx = config['symbol_config']['eos_idx']
        pad_idx = config['symbol_config']['pad_idx']

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
            self.model.encoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=src_vector, freeze=False, padding_idx=pad_idx).to(device)
            self.model.decoder_token_embedding.embedding = nn.Embedding.from_pretrained(embeddings=tgt_vector, freeze=False, padding_idx=pad_idx).to(device)
            # self.model.encoder_token_embedding.embedding.weight.data.copy_(src_vector).to(device)
            # self.model.encoder_token_embedding.embedding.weight.requires_grad = True
            # self.model.decoder_token_embedding.embedding.weight.data.copy_(tgt_vector).to(device)
            # self.model.decoder_token_embedding.embedding.weight.requires_grad = True

        if used_criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        elif used_criterion == 'lsce':
            label_smoothing = config['criteria']['lsce']['label_smoothing']
            self.criterion = LabelSmoothingCrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing).to(device)

        if used_optimizer == 'adam':
            lr = config['optimizer'][used_optimizer]['lr']
            beta1 = config['optimizer'][used_optimizer]['beta1']
            beta2 = config['optimizer'][used_optimizer]['beta2']
            eps = config['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

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
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epochs=epochs, num_examples=train_set_size, batch_size=batch_size, min_lr=min_lr, warmup_size=warmup_size)

        if used_evaluator == 'bleu':
            self.evaluator = TranslationBleuScore(end_index=eos_idx, pad_index=pad_idx)


class TestingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']
        used_model = config['model_config']['model_name']
        d_model = config['model_config']['d_model']
        src_vocab_len = config['model_config']['src_vocab_len']
        tgt_vocab_len = config['model_config']['tgt_vocab_len']
        eos_idx = config['symbol_config']['eos_idx']
        pad_idx = config['symbol_config']['pad_idx']
        nhead = config['model_config']['nhead']
        d_ff = config['model_config']['d_ff']
        max_len = config['model_config']['max_len']
        num_encoder_layers = config['model_config']['num_encoder_layers']
        num_decoder_layers = config['model_config']['num_decoder_layers']
        p_drop = config['model_config']['p_drop']
        if used_model == 'torch_transformer':
            self.model = TorchTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len).to(device)
        elif used_model == 'starriver_transformer':
            self.model = StarRiverTransformer(d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len).to(device)

        if used_evaluator == 'bleu':
            self.evaluator = TranslationBleuScore(end_index=eos_idx, pad_index=pad_idx)



def create_inference_model(config):
    device = config['general']['device']
    used_model = config['model_config']['model_name']
    d_model = config['model_config']['d_model']
    src_vocab_len = config['model_config']['src_vocab_len']
    tgt_vocab_len = config['model_config']['tgt_vocab_len']
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
    eos_idx = config['symbol_config']['eos_idx']
    pad_idx = config['symbol_config']['pad_idx']
    if used_evaluator == 'bleu':
        evaluator = TranslationBleuScore(end_index=eos_idx, pad_index=pad_idx)
    return evaluator