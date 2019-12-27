# -*- coding: utf-8 -*-
from __future__ import print_function

import datetime
import os
import sys
from operator import add

import jieba
import jieba.posseg as pseg
from pyspark import SparkConf, SparkContext, SparkFiles
from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# if len(sys.argv) < 1:
#     print("Usage: wordcount <file>", file=sys.stderr)
#     sys.exit(-1)
print("-----------------------------------------",len(sys.argv))
spark = SparkSession\
    .builder\
    .appName("PythonWordCount")\
    .getOrCreate()
context = spark.sparkContext

stop_word_rdd = context.textFile('file:///home/zhengping/wordPro/stopWords.txt')
stop_words = set(stop_word_rdd.collect())
def lineCut(line):
    return set(jieba.cut(line, cut_all=False)) - stop_words

begin = datetime.datetime.now()
inputFileRdd0 = context.textFile('file:///home/zhengping/wordPro/data.txt')
print(inputFileRdd0)
inputFileRdd1 = inputFileRdd0.flatMap(lambda line:lineCut(line))
print(inputFileRdd1)
inputFileRdd2 = inputFileRdd1.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y).collect()
inputFileRdd2.sort(key=lambda x: x[1],reverse = True)
for i in inputFileRdd2[:30]:
    print(i[0],i[1])
end = datetime.datetime.now()
print((end - begin).seconds)
spark.stop()

words_dict = dict()
for i in inputFileRdd2[:10]:
    words_dict[i[0]] = i[1]

wc = WordCloud(
    font_path="font.ttf",
    background_color='white',
    max_words=10,
    min_font_size=10,
    max_font_size=40,
    collocations=False,
    random_state=42
).fit_words(words_dict)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wc) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
wc.to_file('ans.png')
