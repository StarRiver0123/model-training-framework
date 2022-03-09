import requests


url_prefix = 'http://localhost:8080/'

# to django server:
while 1:
    q = input()
    if q == 'q':
        break
    question = {'question': q}
    answer = requests.post(url=url_prefix+'chat_by_template/', data=question).text
    if answer is not None and answer != 'None':
        print("by_template: ", answer)
        continue
    answer = requests.post(url=url_prefix+'chat_by_corpus/', data=question).text
    if answer is not None and answer != 'None':
        print("by_corpus: ", answer)
        continue
    answer = requests.post(url=url_prefix+'chat_by_internet/', data=question).text
    if answer is not None and answer != 'None':
        print("by_internet: ", answer)
        continue
    print("你说啥？没听懂欸。")

