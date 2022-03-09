import torch.nn.functional as F
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction

class TranslationBleuScore(nn.Module):
    def __init__(self, end_index=None, pad_index=None, ngram_weights=[0.25,0.25,0.25,0.25]):
        super().__init__()
        self.end_index = end_index
        self.pad_index = pad_index
        self.ngram_weights = ngram_weights
        self.smoothing = SmoothingFunction()

    def __call__(self, predict, target):
        # input size:
        #   predict: N,L,C; target: N,L
        # output: scalar
        # mask = torch.full(target.shape, True)  # 注意这里不能用full_like，因为填完之后True变成1，则不能用作布尔索引了
        if predict.dim() == 3:
            predict = F.softmax(predict, dim=-1).argmax(dim=-1)
        target = target
        # remove end_index and pad_index
        if (self.end_index is not None) and (self.pad_index is None):
            mask_target = (target != self.end_index).tolist()
            mask_predict = (predict != self.end_index).tolist()
        elif (self.end_index is not None) and (self.pad_index is not None):
            mask_target = ((target != self.end_index) & (target != self.pad_index)).tolist()
            mask_predict = ((predict != self.end_index) & (predict != self.pad_index)).tolist()
        elif (self.end_index is None) and (self.pad_index is not None):
            mask_target = (target != self.pad_index).tolist()
            mask_predict = (predict != self.pad_index).tolist()
        else:
            mask_target = None
            mask_predict = None
        if mask_target is not None:
            n = len(mask_target)
            list_of_references = [[target[i][mask_target[i]].tolist()] for i in range(n)]
        if mask_predict is not None:
            n = len(mask_predict)
            hypotheses = [predict[i][mask_predict[i]].tolist() for i in range(n)]
        else:
            list_of_references = [[t] for t in target.tolist()]
            hypotheses = predict.tolist()
        # compute the bleu
        bleu = corpus_bleu(list_of_references, hypotheses, weights=self.ngram_weights, smoothing_function=self.smoothing.method7)
        return bleu

