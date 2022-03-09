# coding=utf-8

import requests
from lxml import etree
import html
import json
import os
import time
import random


def print_dict(d, n):
    if isinstance(d, dict):
        for k,v in d.items():
            print("*"*n, " ", k)
            print_dict(v, n+4)

root_path = "jobs"
if not os.path.exists(root_path):
    os.makedirs(root_path)

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36 Edg/98.0.1108.43"
}

keywords = ['nlp', 'java']
for search_keyword in keywords:
    path = root_path + '/' + search_keyword
    if not os.path.exists(path):
        os.makedirs(path)
    page = 1
    while True:
        url = "https://www.aaaaaa.com/wn/jobs?px=new&pn={}&fromSearch=true&kd={}".format(page, search_keyword)
        res = requests.get(url=url, headers=headers)
        res.encoding = 'utf-8'
        # print('page: ', page)
        # print('text:', res.text)   /html/body
        html_xpath = etree.HTML(res.text)
        # body = html_xpath.xpath('/html/body')
        # body_str = html.unescape(etree.tostring(body[0]).decode('utf-8', 'ignore'))
        if (res.text.find("密码登录") != -1) and (res.text.find("验证码登录") != -1) or (res.text.find("页面加载中") != -1):
            sleep_s = random.randint(120, 300)
            print("met login page, hang up for %d seconds..." % sleep_s)
            time.sleep(sleep_s)
            continue
        main_content = html_xpath.xpath('//*[@id="jobList"]/div')[0]
        main_content_str = html.unescape(etree.tostring(main_content).decode('utf-8'))
        if main_content_str.find("暂时没有符合该搜索条件的职位") != -1:
            print("met list end...")
            break
        print("processing page %d" % page)
        json_data = html_xpath.xpath('//*[@id="__NEXT_DATA__"]')[0]
        job_data = json.loads(json_data.text)
        job_list = job_data["props"]["pageProps"]["initData"]["content"]["positionResult"]["result"]
        for job in job_list:
            job_id = job["positionId"]
            job_title = job["positionName"]
            job_slabels = job["skillLables"]
            job_plabels = job["positionLables"]
            job_ilabels = job["industryLables"]
            job_detail = job["positionDetail"].replace("<br>", '').replace("<br />", "").replace("\n\n", "\n")
            # print(job_title, "\n", job_slabels, "\n", job_plabels, "\n", job_ilabels, "\n", job_detail, "\n")
            with open(path + '/' + str(job_id) + '.txt', "a", encoding='utf-8', errors='ignore') as f:
                f.write(job_title + "\n\n")
                f.write(" ".join(job_slabels) + "\n")
                f.write(" ".join(job_plabels) + "\n")
                f.write(" ".join(job_ilabels) + "\n\n")
                f.write(job_detail)
        page += 1
        time.sleep(random.randint(10, 30))
