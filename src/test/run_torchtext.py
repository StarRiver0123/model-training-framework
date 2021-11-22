import sys
from torchtext.legacy.vocab import Vectors
from torchtext.legacy.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
from src.utilities.load_data import *
import torch
from collections import defaultdict
from transformers import BertTokenizer, BertModel

model_name = 'G:\\AI\\projects\\AIPF\\dataset\\bert_model\\chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


FIELD = Field(sequential = True, use_vocab=False, tokenize = None, batch_first = True, fix_length = None, init_token = None,
            eos_token = None, pad_token = None, unk_token = None)

FIELD.build_vocab()
FIELD.vocab.stoi.update(tokenizer.vocab)
FIELD.vocab.itos.extend(tokenizer.vocab.keys())
FIELD.vocab.vectors = model.get_input_embeddings().weight.data

print('over')