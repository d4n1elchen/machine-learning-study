from PttWebCrawler.crawler import *

c = PttWebCrawler(as_lib=True)
for i in range(10):
    c.parse_articles(i*100+1, (i+1)*100, 'Gossiping')
