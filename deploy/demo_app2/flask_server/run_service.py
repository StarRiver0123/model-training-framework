from flask import Flask, request
from flask_cors import cross_origin
import requests, json
import importlib
import sys, os
sys.path.append(os.getcwd())
from src.utilities.load_data import get_config
# 注意：os.getcwd()不是显示的代码所在文件的当前路径，而是调用者所在路径！！！！！！
# os.path.dirname(os.path.abspath(__file__)) 获取代码所在文件的路径

# sys.path.append(os.path.dirname(os.getcwd()))
running_app = os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[-2]
service_module_root = 'deploy.' + running_app + '.services.src.'
# exec("from " + service_module_root + "by_template import ChatRobotByTemplate")
# exec("from " + service_module_root + "by_corpus import ChatRobotByCorpus")
# exec("from " + service_module_root + "by_internet import ChatRobotByInternet")
#
# robot_template = ChatRobotByTemplate()
# robot_corpus = ChatRobotByCorpus()
# robot_internet = ChatRobotByInternet()
app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_config_file = app_root + os.path.sep + 'config_app.yaml'
config = get_config(app_config_file)

vector_file = app_root + os.path.sep + 'services/data/' + config['vector_file']
template_file = app_root + os.path.sep + 'services/data/' + config['template_file']
corpus_file = app_root + os.path.sep + 'services/data/' + config['corpus_file']
url_prefix = config['url_prefix']

robot_template = getattr(importlib.import_module(service_module_root + 'by_template'), 'ChatRobotByTemplate')(vector_file, template_file)
robot_corpus = getattr(importlib.import_module(service_module_root + 'by_corpus'), 'ChatRobotByCorpus')(vector_file, corpus_file)
robot_internet = getattr(importlib.import_module(service_module_root + 'by_internet'), 'ChatRobotByInternet')(url_prefix)

app = Flask(__name__)

@app.route('/chat_by_template', methods=['GET', 'POST'])
@cross_origin()
def chat_by_template():
    if request.method == "POST":
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        answer = robot_template.answer_by_regular(question)
        if answer is None or len(answer) == 0:
            answer = robot_template.answer_by_similarity(question)
            # if answer is None:
            #     answer ="你说啥？没听懂哦。"
        return json.dumps(answer, ensure_ascii=False)


@app.route('/chat_by_corpus', methods=['GET', 'POST'])
@cross_origin()
def chat_by_corpus():
    if request.method == "POST":
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        answer = robot_corpus.answer(question)
        # if answer is None:
        #     answer ="你说啥？没听懂哦。"
        return json.dumps(answer, ensure_ascii=False)


@app.route('/chat_by_internet', methods=['GET', 'POST'])
@cross_origin()
def chat_by_internet():
    if request.method == "POST":
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        answer = robot_internet.answer(question)
        # if answer is None:
        #     answer ="你说啥？没听懂哦。"
        return json.dumps(answer, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='localhost', port=8080, threaded=True)