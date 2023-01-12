#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：awesome-align-master 
@File    ：新能源汽车重新统计.py
@Author  ：znn
@Date    ：2022/9/8 9:12 
"""
import os
from tools.tools import read_json, write_text
base_dir = os.path.join(os.getcwd(), "word_align")
result_data_path = os.path.join(base_dir, "ch2en_align_result.json")
result_list = read_json(result_data_path)
