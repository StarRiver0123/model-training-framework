import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, label_smoothing=None):
        super().__init__()
        self.ignore_index = ignore_index
        if label_smoothing is not None:
            assert 0 <= label_smoothing < 1
        self.label_smoothing = label_smoothing

    def forward(self, predict, target):
        # input size:
        #   predict: N,L,D 即 N,L,C; target: N,L， 或者predict: N,D 即 N,C; target: N
        # output: scalar
        assert (predict.dim() == 3) and (target.dim() == 2) or (predict.dim() == 2) and (target.dim() == 1)
        if (predict.dim() == 3) and (target.dim() == 2):
            predict = predict.reshape(-1, predict.size(-1))
            target = target.reshape(-1)
        batch_size, class_num = predict.size()
        # got one-hot:
        ls_target = torch.zeros_like(predict).scatter_(1, target.view(-1,1).long(), 1)
        # compute the smoothed label:
        if (self.label_smoothing is not None) and (self.label_smoothing != 0):
            ls_target = ls_target * (1-self.label_smoothing) + self.label_smoothing / class_num
        #process the ignore_idx:
        ignore_num = 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            ignore_num = torch.sum(mask).item()
            ls_target.masked_fill_(mask.reshape(-1,1), 0)
        # compute the cross entropy loss
        # default reduction: mean
        log_softmax = F.log_softmax(predict, dim=-1, dtype=float)
        loss = -torch.sum(log_softmax * ls_target) / (batch_size - ignore_num)
        return loss


