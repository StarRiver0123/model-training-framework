import re, gensim, numpy, jieba
from random import choice
# from src.utilities.compute_sentence_similarity import tfidf_vector_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
from src.modules.tokenizers.tokenizer import tokenize_zh_byJieba


class ChatRobotByCorpus():
    def __init__(self, vector_file, corpus_file):
        # corpus_file = r"/deploy/demo_app2/services/data\conversation_test.txt"
        self.list_q = []
        self.list_a = []
        with open(corpus_file, encoding='utf-8') as f:
            for i, text in enumerate(f):
                if i % 2 == 0:
                    self.list_q.append(text)
                else:
                    self.list_a.append(text)
        # model_file = r"/deploy/demo_app2/services/data\news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
        # self.vector_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
        # vector_file = r"/deploy/demo_app2/services/data\wiki.Mode"
        self.vector_model = gensim.models.Word2Vec.load(vector_file)
        # self.tfidf_vectorizer, self.vocab_tokens, self.vocab_vectors = self.create_vocab_vectors(self.list_q, self.vector_model)
        self.similarity_threshold = 0.6


    def answer(self, question):
        vector_q = self.compute_sentence_vector(question)
        best_similarity = -1
        best_index = 0
        for i, q in enumerate(self.list_q):
            vector_c = self.compute_sentence_vector(q)
            similarity = numpy.dot(vector_q, vector_c) / (numpy.linalg.norm(vector_q) * numpy.linalg.norm(vector_c))
            if similarity > best_similarity:
                best_index = i
                best_similarity = similarity
        if best_similarity > self.similarity_threshold:
            return self.list_a[best_index]
        return None

    def compute_sentence_vector(self, sen):
        sen = tokenize_zh_byJieba(sen)
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

    # def create_vocab_vectors(self, vocab_corpus, vector_model):
    #     tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" ")).fit(vocab_corpus)
    #     vocab_tokens = tfidf_vectorizer.get_feature_names()
    #     vocab_vectors = []
    #     zero_vector = numpy.zeros(vector_model.vector_size)
    #     for token in vocab_tokens:
    #         try:
    #             vocab_vectors.append(vector_model.wv[token])
    #         except KeyError:
    #             vocab_vectors.append(zero_vector)
    #     return tfidf_vectorizer, vocab_tokens, vocab_vectors

if __name__ == "__main__":
    robot = ChatRobotByCorpus()
    while 1:
        q = input("请提问：")
        if q == 'q':
            break
        answer = robot.answer(q)
        if answer is not None:
            print(answer)
        else:
            print("你说啥？没听懂欸。")