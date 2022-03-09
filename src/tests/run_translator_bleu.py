from src.modules.evaluators.bleu import TranslationBleuScore
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

if __name__ == '__main__':
    evaluator = TranslationBleuScore(end_index=None,pad_index=None)

    predict = torch.tensor([[[.8,.1,.05,.04],[.6,.3,.09,.2],[.1,.8,.3,.04],[.8,.1,.05,.04],[.6,.3,.9,.2],[.1,.2,.3,.4]],
                            [[.8,.1,.05,.04],[.6,.3,.09,.2],[.1,.8,.3,.04],[.8,.1,.05,.04],[.6,.3,.9,.2],[.1,.2,.3,.4]]])
    #                       0,0,1,0,2,3
    target = torch.tensor([[1,0,1,0,2,4],
                           [1,0,1,0,2,4]])

    # target = [['你','好','吗'],
    #            ['我','喜','欢','你']]
    # predict = [['你','好','吗'],
    #           ['我','喜','欢','你']]
    # target = [['this','is','book'],
    #            ['how','are','you','?']]
    # predict = [['this','is','book'],
    #            ['how','are','you','?']]
    target2 = [[['this','is','book']],
               [['how','are','you','?']]]
    predict2 = [['this','is','book'],
               ['how','are','you','?']]
    target2 = [[['that','is','book','我','喜','欢','that','is','book','我','喜','欢']]]
    predict2 = [['this','is','book','我','喜','huan','this','is','book','我','喜','huan']]
    # target2 = [[[1,0,1,0,2,4]]]
    # predict2 = [[0,0,1,0,2,3]]

    target3 = [['我','好','吗', '我','喜']]
    predict3 = ['你','好','吗','我','喜']
    bleu1 = evaluator(predict,target)
    bleu2 = corpus_bleu(target2, predict2,weights = [0.25,0.25,0.25,0.25])
    bleu3 = sentence_bleu(target3, predict3,weights = [0.34,0.33,0.33])
    print(bleu1, bleu2, bleu3)
