import xml.etree.ElementTree as xmltree
from random import choice
import re, gensim, numpy
import string, zhon.hanzi
# from src.utilities.compute_sentence_similarity import tfidf_vector_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
from src.modules.tokenizers.tokenizer import tokenize_zh_byJieba



class ChatRobotByTemplate():
    def __init__(self, vector_file, template_file):
        # template_file = r"/deploy/demo_app2/services/data\robot_template.xml"
        self.template = xmltree.parse(template_file)
        self.temp = self.template.findall("temp")   #加载问答信息
        # 加载个人属性
        self.robot_info = {}
        for i in self.template.find("robot_info"):
            self.robot_info[i.tag] = i.text
        # model_file = r"/deploy/demo_app2/services/data\news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
        # self.vector_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
        # vector_file = r"/deploy/demo_app2/services/data\wiki.Mode"
        self.vector_model = gensim.models.Word2Vec.load(vector_file)
        self.stop_words = self.get_stop_words()
        # self.corpus = []
        # for t in self.temp:
        #     for q in t.find("question").findall("q"):
        #         candidate_tokens = tokenize_zh_byJieba(q.text, self.stop_words)
        #         self.corpus.append(' '.join(candidate_tokens))
        # self.tfidf_vectorizer, self.vocab_tokens, self.vocab_vectors = self.create_vocab_vectors(self.corpus, self.vector_model)
        self.similarity_threshold = 0.6

    def get_stop_words(self):
        # with open("../../../../dataset/stopwords.txt", encoding='utf-8') as f:
        #     stopwords = list(map(lambda x: re.sub(r"\n", "", x), f.readlines()))
        stopwords = list(string.punctuation + zhon.hanzi.punctuation) + ['+.']
        return stopwords

    def answer_by_regular(self, question):
        q_found = False
        for t in self.temp:
            for q in t.find("question").findall("q"):
                q_found = re.search(q.text, question)
                if q_found:
                    break
            if q_found:
                return choice([a.text for a in t.find("answer").findall("a")]).format(**self.robot_info)
        return None


    def answer_by_similarity(self, question):
        vector_q = self.compute_sentence_vector(question)
        best_similarity = -1
        for t in self.temp:
            for q in t.find("question").findall("q"):
                vector_c = self.compute_sentence_vector(q.text)
                similarity = numpy.dot(vector_q, vector_c) / (numpy.linalg.norm(vector_q) * numpy.linalg.norm(vector_c))
                if similarity > best_similarity:
                    best_temp = t
                    best_similarity = similarity
        if best_similarity > self.similarity_threshold:
            return choice([a.text for a in best_temp.find("answer").findall("a")]).format(**self.robot_info)
        return None


    def compute_sentence_vector(self, sen):
        sen = tokenize_zh_byJieba(sen, self.stop_words)
        vector = numpy.zeros(self.vector_model.vector_size)
        n = 0
        for s in sen:
            try:
                vector += self.vector_model.wv[s]
                n += 1
            except KeyError:
                continue
        if n > 0:
            return vector / n
        else:
            return vector

    # def answer_by_similarity(self, question):
    #     question = ' '.join(tokenize_zh_byJieba(question, self.stop_words))
    #     best_similarity = -1
    #     for t in self.temp:
    #         for q in t.find("question").findall("q"):
    #             candidate_q = ' '.join(tokenize_zh_byJieba(q.text, self.stop_words))
    #             similarity = tfidf_vector_similarity(candidate_q, question, self.vocab_tokens, self.vocab_vectors, self.tfidf_vectorizer, self.vector_model)
    #             if similarity > best_similarity:
    #                 best_temp = t
    #                 best_similarity = similarity
    #     if best_similarity > self.similarity_threshold:
    #         return choice([a.text for a in best_temp.find("answer").findall("a")]).format(**self.robot_info)
    #     return None

    # def create_vocab_vectors(self, vocab_corpus, vector_model):
    #     tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" ")).fit(vocab_corpus)
    #     vocab_tokens = tfidf_vectorizer.get_feature_names()
    #     vocab_vectors = []
    #     for token in vocab_tokens:
    #         try:
    #             vocab_vectors.append(vector_model.wv[token])
    #         except KeyError:
    #             vocab_vectors.append(numpy.zeros(vector_model.vector_size))
    #     return tfidf_vectorizer, vocab_tokens, vocab_vectors

if __name__ == "__main__":
    robot = ChatRobotByTemplate()
    while 1:
        q = input("请提问：")
        if q == 'q':
            break
        answer = robot.answer_by_regular(q)
        if answer is not None:
            print(answer)
        else:
            answer = robot.answer_by_similarity(q)
            if answer is not None:
                print(answer)
            else:
                print("你说啥？没听懂欸。")