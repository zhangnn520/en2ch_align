#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza 
@File    ：get_align_data.py
@Author  ：znn
@Date    ：2022/8/31 10:21 
"""
import os
from tools.tools import *
base_dir = os.path.join(os.getcwd(), "data")
english_file_path = os.path.join(base_dir, "new_data", "english.txt")
chinese_file_path = os.path.join(base_dir, "new_data", "chinese.txt")
en2ch_file_path = os.path.join(base_dir, "new_data", "en2ch.txt")
chinese_word_cut_list = read_text(chinese_file_path)
english_word_cut_list = read_text(english_file_path)
fast_align_en2ch_list = fast_align_data([i.replace("\n", "") for i in english_word_cut_list],
                                        [j.replace("\n", "") for j in chinese_word_cut_list])
write_text(en2ch_file_path, fast_align_en2ch_list)



