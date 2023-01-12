#!/usr/bin/python
# --*-- coding: utf-8 --*--
# @Author  : znn 
# @Email   : m18917010972@163.com
# @Time    : 2022/8/25 17:18
# @File    : nltk_pos.py
# @Software: PyCharm
import os
import re
import nltk
from jieba_cutword_pos import stopwordslist

base_dir = os.getcwd()
english_stop_path = os.path.join(base_dir, "data", "stop_word", "english_stop_words.txt")


def english_pos(sentences_list):
    english_spo_list = []
    # 分词后需要进行使用去停止词
    stop_words_list = stopwordslist(english_stop_path)  # 这里加载停用词的路径
    for sentence in sentences_list:
        all_work_tokenize = nltk.word_tokenize(re.sub('[^\w ]', '', sentence[0]))
        filtered_content_tokenize = [i for i in all_work_tokenize if i not in stop_words_list]
        # 对分完词的结果进行词性标注
        enlish_pos_result = [word + "/" + pos for (word, pos) in nltk.pos_tag(filtered_content_tokenize)]
        english_spo_list.append({"ID": sentence[1], "sentence": sentence[0], "pos_result": enlish_pos_result})
    return english_spo_list
