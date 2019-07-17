PTT 看板列表爬蟲
===

Lastest update: 2019/07/18

Files
---
- `scraper.py`: 爬蟲主程式
- `tree.py`: 讀取 `board_list.csv` 並以樹狀結構印出
- `board_list.csv`: 所有看板列表，一共兩個欄位，第一個欄位為看板分類，多層分類以 `>` 分隔，第二個欄位為看板名稱
- `board_tree.txt`: 由 `tree.py` 所印出的樹狀看板列表

Prerequisite
---
```
pip install scrapy
```

Run
---
```
$ scrapy runspider scraper.py -o board_list.csv -t csv
$ python tree.py
```

個人常用看板
---
```
Gossiping
WomenTalk
C_Chat
Boy-Girl
sex
HatePolitics
Tainan
studyabroad
Tech_Job
Oversea_Job
```
