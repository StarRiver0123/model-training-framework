import xml.etree.ElementTree as xmltree
from random import choice
import re, gensim
from src.utilities.compute_sentence_similarity import tfidf_vector_similarity
from src.modules.tokenizers.tokenizer import tokenize_zh_byJieba_remove_punctuation


class RobotPersonalities():
    def __init__(self, template_file):
        self.template = xmltree.parse(template_file)
        self.temp = self.template.findall("temp")   #加载问答信息
        # 加载个人属性
        self.robot_info = {}
        for i in self.template.find("robot_info"):
            self.robot_info[i.tag] = i.text
        model_file = "../../../../dataset/word_vector/news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
        self.similarity_threshold = 0.6
        self.vector_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    def answer_by_template(self, question):
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
        q_tokens = tokenize_zh_byJieba_remove_punctuation(question)
        best_similarity = -1
        for t in self.temp:
            for q in t.find("question").findall("q"):
                candidate_tokens = tokenize_zh_byJieba_remove_punctuation(q.text)
                similarity = tfidf_vector_similarity(candidate_tokens, q_tokens, self.vector_model)
                if similarity > best_similarity:
                    best_temp = t
                    best_similarity = similarity
        if best_similarity > self.similarity_threshold:
            return choice([a.text for a in best_temp.find("answer").findall("a")]).format(**self.robot_info)
        return None

    def create_vocab(self):
        for t in self.temp:
            for q in t.find("question").findall("q"):
                candidate_tokens = tokenize_zh_byJieba_remove_punctuation(q.text)
                similarity = tfidf_vector_similarity(candidate_tokens, q_tokens, self.vector_model)
                if similarity > best_similarity:
                    best_temp = t
                    best_similarity = similarity

if __name__ == "__main__":
    temp_file = "../../../../dataset/for_edu_dialogue_robot/robot_template.xml"
    robot = RobotPersonalities(temp_file)
    while 1:
        q = input("请提问：")
        if q == 'q':
            break
        answer = robot.answer_by_template(q)
        if answer is not None:
            print(answer)
        else:
            answer = robot.answer_by_similarity(q)
            if answer is not None:
                print(answer)
            else:
                print("你说啥？没听懂欸。")