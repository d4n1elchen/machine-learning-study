import scrapy

BASE_URL = "https://www.ptt.cc"

class BoardSpider(scrapy.Spider):
    name = "PTT_BBS_list_crawler"
    start_urls = ["https://www.ptt.cc/cls/1"]

    def parse(self, response):
        cls = response.meta["cls"] if "cls" in response.meta.keys() else []
        for ent in response.css(".b-ent"):
            path = ent.css("a.board ::attr(href)").extract_first()
            if "cls" in path:
                yield scrapy.Request(BASE_URL+path, callback = self.parse, meta = {"cls": cls + [ent.css(".board-name ::text").extract_first()]})
            elif "bbs" in path:
                yield {
                    "cls": ">".join(cls),
                    "board": ent.css(".board-name ::text").extract_first()
                }

