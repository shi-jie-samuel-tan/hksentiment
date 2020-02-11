import scrapy
from scrapy import Request
from items import *
import datetime
import re


class WeiboSpiderSpider(scrapy.Spider):
    name = 'weibo_spider'
    allowed_domains = ['weibo.cn']
    # start_urls = ['http://weibo.cn/']
    base_url = "https://weibo.cn"

    def start_requests(self):
        url_format = "https://weibo.cn/search/mblog?hideSearchFrame=&keyword={}&advancedfilter=1&starttime={}&endtime={}&sort=time"

        # 搜索的关键词，可以修改

        # keyword = "阴阳师"
        keyword = "高考"

        # 搜索的起始日期，自行修改   微博的创建日期是2009-08-16 也就是说不要采用这个日期更前面的日期了
        date_start = datetime.datetime.strptime("2019-05-20", '%Y-%m-%d')
        # 搜索的结束日期，自行修改
        date_end = datetime.datetime.strptime("2019-06-20", '%Y-%m-%d')

        time_spread = datetime.timedelta(days=1)
        while date_start < date_end:
            next_time = date_start + time_spread
            url = url_format.format(keyword, date_start.strftime("%Y%m%d"), next_time.strftime("%Y%m%d"))
            date_start = next_time
            print("happy")
            yield Request(url, callback=self.parse_tweet, dont_filter=True)