import torch
from transformers import BertTokenizer, BertModel, BertConfig
from src.utilities.load_data import *

# model_name_zh = 'G:\\AI\\projects\\AIPF\\dataset\\bert_model\\chinese-bert-wwm-ext'
# tokenizer_zh = BertTokenizer.from_pretrained(model_name_zh)
# model_zh = BertModel.from_pretrained(model_name_zh)
# ids_zh = tokenizer_zh('花了2765000元', return_tensors='pt', padding=True)
# with torch.no_grad():
#     outs_zh = model_zh(**ids_zh)
# vector1 = outs_zh.last_hidden_state[0, 1]
# print(vector1[:10])
#
# vector2 = model_zh.embeddings.word_embeddings.weight[ids_zh.data['input_ids'][0,1]]
# print(vector2[:10])




# ids_zh2 = (tokenizer_zh.tokenize('花了2765000元'))
# ids_zh3 = (tokenizer_zh.encode('花了2765000元'))
# print(ids_zh)
# print(ids_zh2)
# print(ids_zh3)


model_name_en = 'G:\\AI\\projects\\AIPF\\dataset\\bert_model\\bert-base-uncased'
tokenizer_en = BertTokenizer.from_pretrained(model_name_en)
model_en = BertModel.from_pretrained(model_name_en)
configer_en = BertConfig.from_pretrained(model_name_en)
ids_en = tokenizer_en("This is Mike's book with value of $25.32.", return_tensors='pt')
outs_en = model_en(**ids_en)
# print(outs_en.last_hidden_state)
# print(tokenizer_en.tokenize('this is your book with value of $25.85.'))
print(tokenize_en_bySplit("This is Mike's book with value of $25.32."))
print(tokenizer_en.tokenize("This is Mike's book with value of $25.32."))
print('OK.')

