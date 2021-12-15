from src.models.tokenizer.tokenizer import *

text_en = "that was Mike's text-book with value $256.25, 10% off-cut. He likes apple. pls contact me at erterjkjdf@abc.263.com"
text_zh = '这本参考书的价格是256.25元，便宜了10%，请给我发邮件erterjkjdf@abc.263.com'

# print(tokenize_en_bySpacy(text_en))
print(tokenize_en_byJieba(text_en))
print(count_token(text_en))
print(tokenize_en_bySplit(text_en))
# print(tokenize_zh_bySpacy(text_zh))
# print(tokenize_zh_byJieba(text_zh))
print(tokenize_zh_bySplit(text_zh))
print(tokenize_zh_byStrip(text_zh))
