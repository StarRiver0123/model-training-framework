import torch
import torch.nn as nn
from src.modules.models.twin_classifier import TwinTextRNN
from src.modules.optimizers.lr_scheduler import WarmUpLRScheduler, CosDecayLRScheduler
from src.modules.evaluators.cosine_similarity import CosineSimilarity


class AnswerGradeModel(nn.Module):
    def __init__(self, arguments, word_vector=None, bert_model=None):
        super().__init__()
        used_model = arguments['net_structure']['model']
        used_criterion = arguments['net_structure']['criterion']
        used_optimizer = arguments['net_structure']['optimizer']
        used_lr_scheduler = arguments['net_structure']['lr_scheduler']
        used_evaluator = arguments['net_structure']['evaluator']
        use_bert = arguments['net_structure']['use_bert']
        device = arguments['general']['device']
        vocab_len = arguments['model'][used_model]['vocab_len']
        d_model = arguments['model'][used_model]['d_model']
        pad_idx = arguments['dataset']['symbol']['pad_idx']

        if (used_model == 'twin_textrnn'):
            hidden_size = arguments['model'][used_model]['hidden_size']
            num_layers = arguments['model'][used_model]['num_layers']
            p_dropout = arguments['model'][used_model]['p_drop']
            self.model = TwinTextRNN(vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx).to(device)
            # elif (used_model == 'twin_bert'):
            #     bert_model_name = project_root + os.path.sep + arguments['tasks'][running_task]['bert_model'][
            #         'bert_model_zh']
            #     full_fine_tuning = arguments['training'][running_task]['full_fine_tuning']
            #     self.model = TwinBert(d_vocab, d_model, hidden_size, num_layers, p_dropout, pad_idx).to(device)
            #初始化模型参数
            if (word_vector is not None) or (bert_model is not None):
                for p in self.model.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_normal_(p)
            #初始化词向量， 如果相关传入参数为空，则不做初始化。
            if not use_bert:
                if word_vector is not None:
                    self.model.embedding_a.from_pretrained(embeddings=word_vector, freeze=False, padding_idx=pad_idx)
                    self.model.embedding_b.from_pretrained(embeddings=word_vector, freeze=False, padding_idx=pad_idx)
                    # 注意如果requires_grad==True, 那么对embedding.weight.data的赋值最好是用copy_()函数，直接赋值会有问题？否则会把源头的向量给修改了。
                    # self.model.embedding_a.weight.data = word_vector.to(device)
                    # self.model.embedding_a.weight.requires_grad = False
                    # self.model.embedding_b.weight.data = word_vector.to(device)
                    # self.model.embedding_b.weight.requires_grad = False
            else:     # vector of bert
                if bert_model is not None:
                    self.model.embedding_a = None
                    self.model.embedding_a = None
                    self.model.bert_model = bert_model.to(device)

        if used_criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        elif used_criterion == 'triple':
            self.criterion = nn.TripletMarginLoss().to(device)

        # 设置需要更新的参数，bert模型不做微调，只是取它的动态词向量。
        if not use_bert:
            param_optimizer = self.model.parameters()
        else:
            param_optimizer = list(self.model.fc.named_parameters()) + \
                              list(self.model.layer_norm.named_parameters())
        if used_optimizer == 'adam':
            lr = arguments['optimizer'][used_optimizer]['lr']
            beta1 = arguments['optimizer'][used_optimizer]['beta1']
            beta2 = arguments['optimizer'][used_optimizer]['beta2']
            eps = arguments['optimizer'][used_optimizer]['eps']
            self.optimizer = torch.optim.Adam(param_optimizer, lr=lr, betas=(beta1,beta2), eps=eps)

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
            epochs = arguments['training']['epochs']
            batch_size = arguments['training']['batch_size']
            if ('train_set_size' not in arguments['net_structure']['dataset'].keys()) or (arguments['net_structure']['dataset']['train_set_size'] is None):
                train_set_size = 1000     # 只是为了测试阶段不报错，数量随便写。
            else:
                train_set_size = arguments['net_structure']['dataset']['train_set_size']
            max_lr = arguments['lr_scheduler'][used_lr_scheduler]['max_lr']
            min_lr = arguments['lr_scheduler'][used_lr_scheduler]['min_lr']
            warmup_size = arguments['lr_scheduler'][used_lr_scheduler]['warmup_size']
            self.lr_scheduler = CosDecayLRScheduler(self.optimizer, step_size=lr_step_size, epoches=epochs, num_examples=train_set_size, batch_size=batch_size, max_lr=max_lr, min_lr=min_lr, warmup_size=warmup_size)

        if used_evaluator == 'cosine_similarity':
            self.evaluator = CosineSimilarity()
