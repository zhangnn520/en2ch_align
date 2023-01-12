#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza 
@File    ：维修手册对齐代码.py
@Author  ：znn
@Date    ：2022/9/1 11:14 
"""
import os
import re
from tqdm import tqdm
from tools.tools import *

base_dir = os.path.join(os.getcwd(), "data")
en2ch_result, ch2ch_result = os.path.join(base_dir, "e2z.A3.final"), os.path.join(base_dir, "z2e.A3.final")
en2ch_filename, ch2en_filename = os.path.join(base_dir, "en2.txt"), os.path.join(base_dir, "ch2.txt")


def parse_alignments(alignments_line):
    word_alignments_regex = r"(\S+)\s\(\{([\s\d]*)\}\)"
    alignments = re.findall(word_alignments_regex, alignments_line)
    return alignments


def get_en2ch_unidirectional(filename):
    content_result = []
    with open(filename, encoding="utf-8") as reader:
        for line in reader:
            result = parse_alignments(line)
            if result: content_result.append(result)
    return content_result


if __name__ == '__main__':
    word_align_path = os.path.join(base_dir, "Modle3-中英文官方文档-GIAZ-en2ch单向对齐结果.json")
    result_list = []
    en2ch_result_list = get_en2ch_unidirectional(en2ch_result)
    ch2en_list = read_text(ch2en_filename)
    en2ch_content_list = [i.replace("\n", "") for i in ch2en_list]
    en2ch_list = [i.replace("\n", "").split(" ") for i in ch2en_list]
    for content_num, en in tqdm(enumerate(en2ch_list)):
        temp_list = []
        temp_dict = dict()
        try:
            for num, word in enumerate(en):
                temp_dict[' '] = ' '
                temp_dict[str(num + 1)] = word
            for (en, ch_id) in en2ch_result_list[content_num]:
                element_list = list(set(ch_id.split(" ")))
                element_list.remove("")
                if len(element_list) == 1:

                    if en != 'NULL' and temp_dict[element_list[0]]:
                        temp_list.append((en, temp_dict[element_list[0]]))
                else:
                    multiple_element_list = list(set(ch_id.split(" ")))
                    multiple_element_list.remove("")
                    if en != 'NULL' and [temp_dict[num] for num in multiple_element_list]:
                        temp_list.append((en, [temp_dict[num] for num in multiple_element_list]))
            result_list.append(temp_list)
        except Exception as e:
            print(e)
    assert len(en2ch_content_list) == len(result_list)
    word_align_result = dict(zip(en2ch_content_list, result_list))
    write_line_json(word_align_path, word_align_result)
