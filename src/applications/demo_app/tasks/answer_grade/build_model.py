import torch
import torch.nn as nn
from src.modules.models.twin_classifier import TwinTextRNN
from src.modules.optimizers.lr_scheduler import WarmUpLRScheduler, CosDecayLRScheduler
from src.modules.evaluators.cosine_similarity import CosineSimilarity


class TrainingModel(nn.Module):
    def __init__(self, config, word_vector=None):
        super().__init__()
        used_model = config['net_structure']['model']
        used_criterion = config['net_structure']['criterion']
        used_optimizer = config['net_structure']['optimizer']
        used_lr_scheduler = config['net_structure']['lr_scheduler']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']

        if (used_model == 'twin_textrnn'):
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            vocab_len = config['model_config']['vocab_len']
            d_model = config['model_config']['d_model']
            pad_idx = config['symbol_config']['pad_idx']
            self.model = TwinTextRNN(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx).to(device)
            #初始化词向量， 如果相关传入参数为空，则不做初始化。
            if word_vector is not None:
                self.model.embedding_a.from_pretrained(embeddings=word_vector, freeze=False, padding_idx=pad_idx)
                self.model.embedding_b.from_pretrained(embeddings=word_vector, freeze=False, padding_idx=pad_idx)
            # else:
            #     for p in self.model.parameters():
            #         if p.dim() > 1:
            #             nn.init.xavier_normal_(p)

        if used_criterion == 'ce':
            pad_idx = config['symbol_config']['pad_idx']
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        elif used_criterion == 'triple':
            self.criterion = nn.TripletMarginLoss().to(device)

        # 设置需要更新的参数，bert模型不做微调，只是取它的动态词向量。
        # if ((used_model == 'twin_bert') and not bert_full_fine_tuning):
        #     param_optimizer = list(self.model.fc.named_parameters()) + \
        #                       list(self.model.crf.named_parameters())
        #     # list(self.model.layer_norm.named_parameters())
        #     opt_paras = [{'params': [p for n, p in param_optimizer]}]
        # else:
            param_all = list(self.model.named_parameters())
            #             opt_paras = [{'params': [p for n, p in param_all]}]
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
            self.optimizer = torch.optim.Adam(opt_paras, lr=lr, betas=(beta1,beta2), eps=eps)

        if used_lr_scheduler == 'steplr':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            lr_scheduler_gamma = config['lr_scheduler'][used_lr_scheduler]['gamma']
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_scheduler_gamma)
        elif used_lr_scheduler == 'cosdecay':
            lr_step_size = config['lr_scheduler'][used_lr_scheduler]['step_size']
            epochs = config['training']['epochs']
            batch_size = config['training']['batch_size']
            train_set_size = config['net_structure']['dataset']['train_set_size']
            max_lr = config['lr_scheduler'][used_lr_scheduler]['max_lr']
            min_lr = config['lr_scheduler'][used_lr_scheduler]['min_lr']
            warmup_size = config['lr_scheduler'][used_lr_scheduler]['warmup_size']
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epoches=epochs, num_examples=train_set_size, batch_size=batch_size, max_lr=max_lr, min_lr=min_lr, warmup_size=warmup_size)

        if used_evaluator == 'cosine_similarity':
            self.evaluator = CosineSimilarity()


class TestingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        used_model = config['model_config']['model_name']
        used_evaluator = config['net_structure']['evaluator']
        device = config['general']['device']

        if (used_model == 'twin_textrnn'):
            hidden_size = config['model_config']['hidden_size']
            num_layers = config['model_config']['num_layers']
            p_dropout = config['model_config']['p_drop']
            vocab_len = config['model_config']['vocab_len']
            d_model = config['model_config']['d_model']
            pad_idx = config['symbol_config']['pad_idx']
            self.model = TwinTextRNN(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx).to(device)

        if used_evaluator == 'cosine_similarity':
            self.evaluator = CosineSimilarity()


def create_inference_model(config):
    used_model = config['model_config']['model_name']
    device = config['general']['device']
    if (used_model == 'twin_textrnn'):
        hidden_size = config['model_config']['hidden_size']
        num_layers = config['model_config']['num_layers']
        p_dropout = config['model_config']['p_drop']
        vocab_len = config['model_config']['vocab_len']
        d_model = config['model_config']['d_model']
        pad_idx = config['symbol_config']['pad_idx']
        model = TwinTextRNN(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx).to(device)
    return model

def create_evaluator(config):
    used_evaluator = config['net_structure']['evaluator']
    if used_evaluator == 'cosine_similarity':
        evaluator = CosineSimilarity()
    return evaluator