import json
import csv
import re

def chinese_filter(line):
    rule = re.compile(r"[^\u4e00-\u9fff]")
    line = rule.sub('',line)
    return line

push = []
for i in range(10):
    with open(f"Gossiping-{i*100+1}-{(i+1)*100}.json") as json_file:
        articles = json.load(json_file)["articles"]
        for article in articles:
            if "messages" in article.keys():
                pushes = article["messages"]
                for p in pushes:
                    tag = 2 if p["push_tag"] == "推" else 1 if p["push_tag"] == "→" else 0
                    content = p["push_content"]
                    content = chinese_filter(content)
                    if content == "":
                        continue
                    push.append((tag, content))

print(len(push))
with open("push_list.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(("tag","content"))
    for p in push:
        writer.writerow(p)
