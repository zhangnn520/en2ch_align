#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：awesome-align-master 
@File    ：data_transfor.py
@Author  ：znn
@Date    ：2022/9/15 10:53 
"""
import os
from tools.tools import read_text,write_text,fast_align_data
base_dir = os.getcwd()
english_list,chinese_list = [],[]
file_path = os.path.join(base_dir,"小于20数据集.txt")
new_file_path = os.path.join(base_dir,"ch2en小于20数据集.txt")
content_list = read_text(file_path)
for content in content_list:
    english_content = content.split(" ||| ")[0]
    chinese_content = content.split(" ||| ")[1]
    english_list.append(english_content)
    chinese_list.append(chinese_content)

ch2en_list,_ = fast_align_data(chinese_list,english_list)
write_text(new_file_path,ch2en_list)