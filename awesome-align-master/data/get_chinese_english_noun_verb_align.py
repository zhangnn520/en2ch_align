#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza 
@File    ：get_chinese_english_noun_verb_align.py
@Author  ：znn
@Date    ：2022/9/1 8:47 
"""
# 该工程文件用于将source和target内容进行拼接成中间数据
import os
from tools.tools import read_text, write_text, fast_align_data

base_dir = os.path.join(os.getcwd(), "word_align")
source_file_path = os.path.join(base_dir, "source.txt")
target_file_path = os.path.join(base_dir, "target.txt")
en2ch_result_path = os.path.join(base_dir, "en2ch.txt")


def create_result():
    source_content = read_text(source_file_path)
    target_content = read_text(target_file_path)
    source2target_data = fast_align_data(source_content, target_content)
    write_text(en2ch_result_path, source2target_data)


if __name__ == "__main__":
    create_result()
