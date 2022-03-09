import jieba, re
# import spacy
# tokenizer_spacy_en = spacy.load("en_core_web_sm")
# tokenizer_spacy_zh = spacy.load("zh_core_web_sm")



def tokenize_en_bySplit(text):
    return [word for word in en_data_clean(text).split()]
    # 英语标点符号集：!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # 其中：单引号'，连接符号-，下划线_等，进行保留，其他的左右两侧加空格，但是.两侧如果都紧跟数字或字母则不加空格（小数和email地址）。

# def tokenize_en_bySpacy(text):
#     # return [word.lemma_ for word in spacy_tokenizer(text)]
#     return [word for word in tokenizer_spacy_en(text)]

def count_token(text):
    text = re.sub(r'(?<=[^a-zA-Z0-9])|(?=[^a-zA-Z0-9])', " ", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\s]', '', text)
    return len(text)

def tokenize_en_byJieba(text):
    return [word for word in jieba.cut(text) if word != ' ']

def en_data_clean(text):
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"\n", " ", text)
    text = text.lower()
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"e-mail", "email", text)
    # text = re.sub(r'((?<=[&,/:;`~\#\<\=\>\!\*\+\.\"\$\^\?\(\)\[\\\]\{\|\}])|(?=[&,/:;`~\#\<\=\>\!\*\+\.\"\$\^\?\(\)\[\\\]\{\|\}]))(?<![0-9a-zA-Z]\.)(?!\.[0-9a-zA-Z])', " ", text)
    text = re.sub(
        r'(?<=[@,&,/:;`~\#\<\=\>\!\*\+\.\"\$\%\^\?\(\)\[\\\]\{\|\}\d])|(?=[@,&,/:;`~\#\<\=\>\!\*\+\.\"\$\%\^\?\(\)\[\\\]\{\|\}\d])',
        " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def zh_data_clean(text):
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r'(?<=[^a-zA-Z0-9\._@])|(?=[^a-zA-Z0-9\._@])', " ", text)
    return text


def tokenize_zh_byStrip(text):
    return [word for word in text.strip() if word != ' ']

def tokenize_zh_bySplit(text):
    return [word for word in zh_data_clean(text).split() if word != ' ']

# def tokenize_zh_bySpacy(text):
#     return [word.text for word in tokenizer_spacy_zh(text)]

def tokenize_zh_byJieba(text, stopwords=None):
    if stopwords is not None:
        return [word for word in jieba.cut(text) if (word != ' ') and (word not in stopwords)]
    else:
        return [word for word in jieba.cut(text) if word != ' ']
