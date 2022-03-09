import requests
import json

url_prefix = 'http://localhost:8080/'


while 1:
    q = input()
    if q == 'q':
        break
    question = json.dumps({'question': q})
    answer = requests.post(url=url_prefix+'chat_by_template', data=question).json()
    if answer is not None:
        print("by_template: ", answer)
        continue
    answer = requests.post(url=url_prefix+'chat_by_corpus', data=question).json()
    if answer is not None:
        print("by_corpus: ", answer)
        continue
    answer = requests.post(url=url_prefix+'chat_by_internet', data=question).json()
    if answer is not None:
        print("by_internet: ", answer)
        continue
    print("你说啥？没听懂欸。")
