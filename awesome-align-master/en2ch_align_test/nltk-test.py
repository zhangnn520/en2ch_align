#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：awesome-align-master 
@File    ：nltk-calc_word_vector.py
@Author  ：znn
@Date    ：2022/9/15 9:57 
"""
import os
import re
import nltk
from data.fast_align_data_processor import read_text

base_dir = os.getcwd()
# 直接将模型预测的结果写入文件中，方便后续进行数据写入数据库中
result_data_path = os.path.join(base_dir, "ch2en_align_result.json")
new_result_data_path = os.path.join(base_dir, "new_ch2en_align_result.json")
en2ch_file_path = os.path.join(base_dir, "小于20数据集.txt")


def get_cut_word_result():
    en2ch_content_list = read_text(en2ch_file_path)
    for en2ch_content in en2ch_content_list:
        english_content = en2ch_content.split("|||")[0]
        chinese_content = en2ch_content.split("|||")[1]
        string = re.sub(r'[^\w ]', '', english_content)
        print(english_content)

        print(nltk.word_tokenize(string))
        print("&&&&&&&&&&&&&&&&&&777777777777777777777777777")
        # print(nltk.pos_tag(nltk.word_tokenize(string)))  # 对分完词的结果进行词性标注


if __name__ == "__main__":
    get_cut_word_result()
