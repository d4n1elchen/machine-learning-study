#/bin/bash

wget http://data.ncku.edu.tw/storage/f/2018-02-02T03%3A28%3A20.393Z/activity-record.csv
iconv -f big5 -t utf-8 activity-record.csv -o activity-record-utf8.csv
