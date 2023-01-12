#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza 
@File    ：fast_align_data_processor.py
@Author  ：znn
@Date    ：2022/8/31 16:12 
"""
import os
import random
from tqdm import tqdm

base_dir = os.path.join(os.getcwd(), "word_align")


def write_text(write_path, content_line, types="\n"):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            if content != content_line[-1] and types != "\n":
                writer.write(content)
            else:
                writer.write(content)


def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return reader.readlines()


def get_word_dict(content_list):
    word_list = list()
    for content in tqdm(content_list):
        word_list = word_list + content.split(" ")
    word_list = list(set(word_list))
    word_list.remove("|||")
    print("詞典個數大小:", len(word_list))
    return word_list


if __name__ == "__main__":
    train_data_path = os.path.join(base_dir, "../model_data", "train.txt")
    dev_data_path = os.path.join(base_dir, "../model_data", "test.txt")
    test_data_path = os.path.join(base_dir, "../model_data", "test.txt")
    # word_dict_path = os.path.join(base_dir, "../model_data", "dict.txt")
    en2ch_file_path = os.path.join(base_dir, "小于20数据集.txt")
    en2ch_content_list = read_text(en2ch_file_path)
    # random.shuffle(en2ch_content_list)
    # random.shuffle(en2ch_content_list)
    # 生成字典形式
    # word_list = get_word_dict(en2ch_content_list)
    random.shuffle(en2ch_content_list)
    content_num = len(en2ch_content_list)
    train_data = en2ch_content_list[:int(content_num * 0.9)]
    dev_data = en2ch_content_list[int(content_num * 0.9):]
    # test_data = en2ch_content_list[int(content_num * 0.9):]
    # write_text(word_dict_path, word_list, "\n")
    write_text(train_data_path, train_data, "\t")
    write_text(dev_data_path, dev_data, "\t")
    # write_text(test_data_path, test_data, "\t")
