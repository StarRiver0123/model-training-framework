from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.modules.tokenizers.tokenizer import tokenize_zh_byJieba
import numpy as np


def tfidf_vector_similarity(corpus_s, input_s, vocab_tokens, vocab_vectors, tfidf_vectorizer, vector_model):
    corpus = [corpus_s, input_s]
    tfidfs = tfidf_vectorizer.transform(corpus).toarray()
    s1 = tfidf_sentence_vector(vocab_vectors, tfidfs[0])
    # 下面做了改进，如果词语不在字典里，进行补充计算，不能给对应到零向量。
    added_vocab = []
    added_tfidf = []
    for token in input_s.split():
        if token not in vocab_tokens:
            if token in vector_model:
                added_vocab.append(vector_model[token])
            else:
                added_vocab.append(np.zeros(vector_model.vector_size))
            added_tfidf.append(np.sum(tfidfs[1])/np.sum(tfidfs[1] != 0))
    s2 = tfidf_sentence_vector(vocab_vectors + added_vocab, list(tfidfs[1]) + added_tfidf)
    similarity = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
    return similarity


# def tfidf_vector_similarity(corpus_s, input_s, vocab_vectors, tfidf_vectorizer):
#     corpus = [corpus_s, input_s]
#     tfidfs = tfidf_vectorizer.transform(corpus).toarray()
#     s1 = tfidf_sentence_vector(vocab_vectors, tfidfs[0])     # 这里需要改进，如果词语不在字典里，进行补充计算，不能给对应到零向量。
#     s2 = tfidf_sentence_vector(vocab_vectors, tfidfs[1])
#     similarity = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
#     return similarity

def tfidf_sentence_vector(token_vectors, tfidf=None):
    vector = np.zeros(len(token_vectors[0]))
    if tfidf is None:
        tfidf = np.ones(len(token_vectors))
    else:
        assert len(token_vectors) == len(tfidf)
    for i, token_vector in enumerate(token_vectors):
        vector += token_vector * tfidf[i]
    vector /= np.sum(tfidf)
    return vector


if __name__ == "__main__":
    s1 = "你 是 哪一位"
    s2 = "你 是 谁"
    corpus = [s1, s2]
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))
    tfidfs = vectorizer.fit(corpus)
    print(vectorizer.vocabulary_)
    print(vectorizer.get_feature_names())
    # print(tfidfs.toarray())
    vectorizer.vocabulary_['我'] = 4
    vectorizer.vocabulary_['他'] = 5
    vectorizer.n_features = 6
    print(vectorizer.get_params())
    s3 = ["我 是 他"]
    tfidf = vectorizer.transform(s3)
    print(tfidf.toarray())
