#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza 
@File    ：fast_align_data_processor.py
@Author  ：znn
@Date    ：2022/8/31 16:12 
"""
import os
import copy
# import random
from tqdm import tqdm
from en2ch_align_test.test import model_eval_fuc
from data.fast_align_data_processor import read_text
from tools.tools import write_json, delete_duplicate_elements

base_dir = os.getcwd()
# 直接将模型预测的结果写入文件中，方便后续进行数据写入数据库中
result_data_path = os.path.join(base_dir, "ch2en_align_result.json")
new_result_data_path = os.path.join(base_dir, "new_ch2en_align_result.json")
# en2ch_file_path = os.path.join(base_dir, "小于20数据集.txt")
en2ch_file_path = os.path.join(base_dir, "test.txt")


def get_test_result():
    en2ch_content_list = read_text(en2ch_file_path)
    # random.shuffle(en2ch_content_list)
    result_list = list()
    for en2ch_content in tqdm(en2ch_content_list):
        temp_dict = dict()
        english_content = en2ch_content.split("|||")[0]
        chinese_content = en2ch_content.split("|||")[1]
        result_data = model_eval_fuc(english_content, chinese_content)
        temp_dict["chinese_content"] = english_content.replace("\n", "")
        temp_dict["english_content"] = chinese_content.replace("\n", "")
        temp_dict["ch2en_word_align"] = result_data
        result_list.append(temp_dict)
    write_json(result_data_path, result_list)
    return result_list


if __name__ == "__main__":
    result_lists = get_test_result()
    for result in tqdm(result_lists):
        temp_list = []
        data_list = []
        new_result_list = []
        ch2en_aligin = result['ch2en_word_align']
        en2ch_align = copy.deepcopy(ch2en_aligin)
        for num in range(len(ch2en_aligin) - 1):
            first_thing = ch2en_aligin[num]
            second_thing = ch2en_aligin[num + 1]
            if first_thing[1] == second_thing[1] and first_thing[0] != second_thing[0]:
                data_list.append([ch2en_aligin[num][0], ch2en_aligin[num][1]])
                temp_list.append([ch2en_aligin[num][0] + " " + ch2en_aligin[num + 1][0], ch2en_aligin[num + 1][1]])
            elif first_thing[1] == second_thing[1] and first_thing[0] == second_thing[0]:
                data_list.append([ch2en_aligin[num][0], ch2en_aligin[num][1]])
            elif first_thing[1] != second_thing[1] and first_thing[0] == second_thing[0]:
                data_list.append([ch2en_aligin[num][0], ch2en_aligin[num][1]])
                temp_list.append([ch2en_aligin[num][0], ch2en_aligin[num][1] + " " + ch2en_aligin[num + 1][1]])
        if data_list:
            for data in data_list:
                en2ch_align.remove(data)
            en2ch_align = en2ch_align + temp_list
            en2ch_align = delete_duplicate_elements(en2ch_align)
            new_result_list.append(en2ch_align)
            new_result_list = delete_duplicate_elements(new_result_list)
            result['ch2en_word_align'] = new_result_list
    write_json(new_result_data_path, result_lists)
