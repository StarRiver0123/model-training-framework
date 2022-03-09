# Scrapy settings for nlp project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'nlp'

SPIDER_MODULES = ['nlp.spiders']
NEWSPIDER_MODULE = 'nlp.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36 Edg/98.0.1108.50'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
COOKIES_ENABLED = True

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
    'Cookie': 'zhishiTopicRequestTime=1645022361294; BAIKE_SHITONG={"data":"f56cd1ebc111d2e990b75ad23425ea85eb9000448b88a28b57e3fb90dc9656ae3012b2fddf4df7ecf8bafb22269f4043a88edae5600e11d390a6ae03b2494e1e739dbf45599c3043263d4a2c8deddaa5770c4b50f38b115872cf3e0a885c080a","key_id":"10","sign":"82fb1d81"}; BIDUPSID=EFBB27DA639B8D1C1AA550F20B98F969; PSTM=1627384578; __yjs_duid=1_6a1310b44965628bc93a00640b4ebc831627385017672; BDUSS=EIxYU1DdGhRV3dKRVFHN3VmOGJLRElCZ1RwUDRvfmRneWE2MUQ0bllvcEJnenhoSVFBQUFBJCQAAAAAAAAAAAEAAAAZAlUNaGVuZHVvd2VuemhhbmcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEH2FGFB9hRhV; BDUSS_BFESS=EIxYU1DdGhRV3dKRVFHN3VmOGJLRElCZ1RwUDRvfmRneWE2MUQ0bllvcEJnenhoSVFBQUFBJCQAAAAAAAAAAAEAAAAZAlUNaGVuZHVvd2VuemhhbmcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEH2FGFB9hRhV; BK_SEARCHLOG={"key":["桀","涿鹿","仿射变换","向量的范数和模","向量的范数","马尔可夫过程","马尔可夫假设","马尔可夫链","条件随机场"]}; BAIDUID=13A3DE8A931F7E37B336E9F2DF518CB3:FG=1; zhishiTopicRequestTime=1644978955601; baikeVisitId=36861b9e-9306-4422-b118-dffd390746ae; Hm_lvt_55b574651fcae74b0a9f1cf9c8d7c93a=1643169099,1644237441,1644978970; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDSFRCVID=s9kOJeC62uJiB9QDBBeYjiIwBm8ayeQTH6aok5EM-M7i-G7kZU1WEG0PbM8g0Kub12KkogKK0gOTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF=tRAOoCIbtDvbeJrc5DTD-tFO5eT22-usMjTr2hcH0KLKEp3J2f6xQ5t1DUKHtlLjLIQA0qj2afb1MRjvXx5fDRLdhb3H5qbD0CORhq5TtUthSDnTDMRhqqJXX-7yKMnitIj9-pnG2hQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuD6_MejvbeatsKC62atoLBRjOMJnqD6rnhPF3KTFfXP6-35KHbmQ05DTl3poIefTFMfrbL4rb-qr-Ql37JD6yBK5NB66YMnIzQTJOWlQL3PoxJpOaBRbMopvaKDDKOnvvbURvD--g3-AqBM5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-j5JIE3-oJqC8ahI_x3f; BDSFRCVID_BFESS=s9kOJeC62uJiB9QDBBeYjiIwBm8ayeQTH6aok5EM-M7i-G7kZU1WEG0PbM8g0Kub12KkogKK0gOTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF_BFESS=tRAOoCIbtDvbeJrc5DTD-tFO5eT22-usMjTr2hcH0KLKEp3J2f6xQ5t1DUKHtlLjLIQA0qj2afb1MRjvXx5fDRLdhb3H5qbD0CORhq5TtUthSDnTDMRhqqJXX-7yKMnitIj9-pnG2hQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuD6_MejvbeatsKC62atoLBRjOMJnqD6rnhPF3KTFfXP6-35KHbmQ05DTl3poIefTFMfrbL4rb-qr-Ql37JD6yBK5NB66YMnIzQTJOWlQL3PoxJpOaBRbMopvaKDDKOnvvbURvD--g3-AqBM5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-j5JIE3-oJqC8ahI_x3f; delPer=0; PSINO=1; BAIDUID_BFESS=B4164D523653E260619088B7D48741C1:FG=1; BDRCVFR[feWj1Vr5u3D]=mk3SLVN4HKm; H_PS_PSSID=35836_35104_31253_35766_35488_34584_35490_35871_35797_35325_26350_35881_35877_35746; Hm_lpvt_55b574651fcae74b0a9f1cf9c8d7c93a=1645022361; ab_sr=1.0.1_NDY3M2NmZTNlOTBkMWY5ZWRmZTFhNGM1Yjg1ZjJhN2QyMjFjN2U3NTFiZmVjNzQyNmE3YjI3YzQwYjZiNWFmZjVmYzliZjNmM2ZjNjU5NWNiODdiNWEyN2FjNDhkYzNhNTIxOGNlOTNiMGFhMmM5ZDA2NGMxODIwYTcxZTA2MWJkYThjNTM1ODUzY2Q1YjgyMjQzMjI1M2Q5YTUwOWViYjZkMWYzNjE4MzFmZWE0NjEzNzg1MjNhMDllNjQ4MjYx'
}

# Enable or disable data_crawling middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'nlp.middlewares.NlpSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'nlp.middlewares.NlpDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'nlp.pipelines.CleanData': 300,
    'nlp.pipelines.SaveData': 301,
}

SAVE_FILE = 'baike.csv'
# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
