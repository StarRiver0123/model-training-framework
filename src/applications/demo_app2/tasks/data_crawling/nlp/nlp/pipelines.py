# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import csv

from itemadapter import ItemAdapter
import re

class CleanData:
    def process_item(self, item, spider):
        new_item = {}
        new_item['title'] = item['title']
        new_item['summary'] = ''
        for s in item['summary']:
            new_item['summary'] += re.sub('\[\d+\]\xa0', '', s.replace('\n', '')) + '\n'
        new_item['attributes'] = list(map(lambda x: x.replace('\xa0',''), item['attributes']))
        values = list(map(lambda x: x.replace('\xa0','').replace('\n',''), item['values']))
        new_item['values'] = list(filter(None, values))
        return new_item


class SaveData:
    @classmethod
    def from_crawler(cls, crawler):
        return cls(save_file = crawler.settings.get('SAVE_FILE'))

    def __init__(self, save_file):
        self.save_file = save_file

    def open_spider(self, spider):
        self.file = open(self.save_file, "a", encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.file)

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.csv_writer.writerow([item['title'], '概要', item['summary']])
        if len(item['attributes']) != len(item['values']):
            return item
        for i in range(len(item['attributes'])):
            self.csv_writer.writerow([item['title'], item['attributes'][i], item['values'][i]])
        return item

