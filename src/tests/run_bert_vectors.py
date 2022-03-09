import torch
from transformers import BertTokenizer, BertModel, BertConfig
from src.utilities.load_data import *
import sys

model_name_zh = 'G:\\AI\\projects\\mtf_projects\\dataset\\bert_model\\chinese-bert-wwm-ext'
tokenizer_zh = BertTokenizer.from_pretrained(model_name_zh)
# model_zh = BertModel.from_pretrained(model_name_zh)
ids_zh = tokenizer_zh('浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，', return_tensors='pt', padding=True)
print(ids_zh)
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
# print(ids_zh3)tags

#
# model_name_en = 'G:\\AI\\projects\\ResumeRobot\\dataset\\bert_model\\bert-base-uncased'
# tokenizer_en = BertTokenizer.from_pretrained(model_name_en)
# model_en = BertModel.from_pretrained(model_name_en)
# configer_en = BertConfig.from_pretrained(model_name_en)
# ids_en = tokenizer_en("This is Mike's book with value of $25.32.", return_tensors='pt')
# outs_en = model_en(**ids_en)
# tokens = tokenizer_en.tokenize("按顺序对应的，id是None，query12对应的是text，对应的处理是text_field对应的处理。 label对应的是。这个是我故意这么起名的，就是为了说明其实是按排序，而不是名字匹配的。text_field里包含了你需要对'query12'里数据的各种处理.")
# print(sys.getsizeof("按顺序对应的，id是None，query12对应的是text，对应的处理是text_field对应的处理。 label对应的是。这个是我故意这么起名的，就是为了说明其实是按排序，而不是名字匹配的。text_field里包含了你需要对'query12'里数据的各种处理."))
# print(sys.getsizeof(tokens))
#
#
# ids = tokenizer_en.convert_tokens_to_ids(tokens)
#
# ids2 = tokenizer_en.encode("This is Mike's book with value of $25.32.")
# text2 = tokenizer_en.convert_ids_to_tokens(ids2)
# # print(outs_en.last_hidden_state)
# # print(tokenizer_en.tokenize('this is your book with value of $25.85.'))
# # print(tokenize_en_bySplit("This is Mike's book with value of $25.32."))
# print(tokenizer_en.tokenize("This is Mike's book with value of $25.32."))
# print('OK.')

