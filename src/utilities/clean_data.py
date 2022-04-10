import re
from functools import partial
from string import punctuation as en_punc
from zhon.hanzi import punctuation as zh_punc


class DataCleaner():
    def __init__(self, clean_config, stopwords=None):
        if clean_config['replace_english']:
            self.english_token_replacement = clean_config['english_token_replacement']
        if clean_config['replace_digits']:
            self.digits_token_replacement = clean_config['digits_token_replacement']
        if clean_config['replace_specials']:
            self.replace_list = clean_config['specials_token_replacement']
        self.english_punctuation = en_punc
        self.chinese_punctuation = zh_punc
        self.stopwords = stopwords
        self.clean_pipeline = self.create_clean_pipeline(clean_config)

    def clean(self, data_list):
        data_set = list(map(self.clean_pipeline, data_list))
        return data_set

    def create_clean_pipeline(self, clean_config):
        pipeline = []
        if clean_config['replace_specials']:
            pipeline.append(self.replace_specials)
        if clean_config['replace_english']:
            pipeline.append(self.replace_english)
        if clean_config['replace_digits']:
            pipeline.append(self.replace_digits)
        if clean_config['remove_english_punctuation']:
            pipeline.append(self.remove_english_punctuation)
        if clean_config['remove_chinese_punctuation']:
            pipeline.append(self.remove_chinese_punctuation)
        if clean_config['remove_non_hanzi_english_digits']:
            pipeline.append(self.remove_chinese_punctuation)
        if clean_config['lower_case']:
            pipeline.append(self.lower_case)
        pipeline.append(self.shrank_blank)
        if len(pipeline) > 0:
            return partial(self.sequential_cleaning, pipeline)
        return None

    def sequential_cleaning(self, pipeline, text_seq):
        for p in pipeline:
            text_seq = p(text_seq)
        return text_seq

    def replace_specials(self, text):
        for r in self.replace_list:
            text = re.sub(r[0], r[1], text)
        return text

    def replace_english(self, text):
        text = re.sub('[\u0370-\u03ffa-zA-Z_-]+[0-9.]*', ' ' + self.english_token_replacement + ' ', text)   # \u0370-\u03ff 为希腊字母
        return text

    def replace_digits(self, text):
        text = re.sub('[0-9]+\.*[0-9]+', ' ' + self.digits_token_replacement + ' ', text)
        return text

    def remove_english_punctuation(self, text):
        text = re.sub('[' + self.english_punctuation + r'\\' + ']', ' ', text)
        # text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\\]', ' ', text)
        # text = ''.join([w for w in list(text) if w not in self.english_punctuation])
        return text

    def remove_chinese_punctuation(self, text):
        text = re.sub('[' + self.chinese_punctuation + ']', ' ', text)
        # text = re.sub('[＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]', ' ', text)
        return text

    def lower_case(self, text):
        text = text.lower()
        return text

    def shrank_blank(self, text):
        text = re.sub('\s+', ' ', text)
        return text

    def remove_non_hanzi_english_digits(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_.-]|([^\d]\.)|(\.[^\d])', ' ', text)
        return text

    def remove_stopwords(self, text):
        pass



if __name__ == '__main__':
    test1 = '这是一句测试�Θθj+1=θj∆·∂Loss/∂θ=θj∆·1/m·∑x·(hy)，  说hello，    值25.36. iphone10好用吗？this is a english_text-book 36, yes!\w[] 我说：“你干啥呢”。《人世间》很好看！'
    r_list = [('25', 'aa'), ('hello', 'hi')]
    cleaner = DataCleaner()
    print(test1)
    # print(cleaner.remove_digits(test1))
    # print(cleaner.remove_english(test1))
    # print(cleaner.replace_digits(test1))
    # print(cleaner.replace_english(test1))
    # print(cleaner.remove_english_punctuation(test1))
    # print(cleaner.remove_chinese_punctuation(test1))
    # print(cleaner.remove_non_hanzi_english_digits(test1))
    print(cleaner.replace_specified(r_list, test1))
    print(cleaner.shrank_blank(test1))