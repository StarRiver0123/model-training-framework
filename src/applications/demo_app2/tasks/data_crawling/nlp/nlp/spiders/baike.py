import scrapy
from src.applications.demo_app2.tasks.spider.nlp.nlp.items import NlpItem


class BaikeSpider(scrapy.Spider):
    name = 'baike'
    allowed_domains = ['baike.baidu.com']
    start_urls = ['https://baike.baidu.com/item/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/9180']
    url_count = 1
    def parse(self, response):
        item = NlpItem()
        item['title'] = response.xpath('//h1/text()').get()
        item['summary'] = [''.join(s.xpath('.//text()').getall()) for s in response.xpath('//div[@class="lemma-summary"]/div')]
        item['attributes'] = [''.join(a.xpath('.//text()').getall()) for a in response.xpath('//div[contains(@class,"basic-info")]//dt')]
        item['values'] = [''.join(a.xpath('.//text()').getall()) for a in response.xpath('//div[contains(@class,"basic-info")]//dd')]
        yield item

        item_urls = response.xpath('//div[@class="para"]//a[contains(@href,"/item/")]/@href').getall()
        for item_url in item_urls:
            if BaikeSpider.url_count < 1000:
                url = "https://baike.baidu.com" + item_url
                BaikeSpider.url_count += 1
                # print("Added url No. ", BaikeSpider.url_count)
                yield scrapy.Request(url)


