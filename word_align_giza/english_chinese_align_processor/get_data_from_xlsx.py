#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza 
@File    ：get_data_from_xlsx.py
@Author  ：znn
@Date    ：2022/8/31 11:02 
"""
import os
import xlrd
import jieba
import paddle
from tqdm import tqdm
from tools.tools import sub_word, write_text,word_tokenize

user_defined_word_path = r"E:\ENOCH-2022\NLP_MODEL\07_machine_transformer\word_align_giza\data\new_dict\汽车维修词典.txt"
jieba.load_userdict(user_defined_word_path)
base_dir = os.path.join(os.getcwd(), "data")


def word_cut(content, lan_type="ch"):

    if lan_type == "ch":
        paddle.enable_static()
        seg_list = jieba.cut(content, use_paddle=True)  # 使用paddle模式进行分词
    else:
        seg_list = word_tokenize(content)
    return ' '.join(seg_list)


def get_data_from_xlsx(path):
    chinese_sentence_list = []
    enlish_sentence_list = []
    # 打开excel
    wb = xlrd.open_workbook(path)
    # 按工作簿定位工作表
    sh = wb.sheet_by_name('维修手册详细步骤')
    # print(sh.nrows)  # 有效数据行数
    # print(sh.ncols)  # 有效数据列数
    # print(sh.cell(0, 0).value)  # 输出第一行第一列的值
    # print(sh.row_values(0))  # 输出第一行的所有值
    for i in range(sh.nrows):
        # 将数据和标题组合成字典
        xlsx_dict = dict(zip(sh.row_values(0), sh.row_values(i)))
        # 遍历excel，打印所有数据
        chinese_maintenance = xlsx_dict['维修内容（中文）']
        chinese_sentence_list.append(chinese_maintenance)
        english_maintenance = xlsx_dict['维修内容（英文）']
        enlish_sentence_list.append(english_maintenance)

    return chinese_sentence_list[1:], enlish_sentence_list[1:]


# def get_data_from_xlsx(path):
#     chinese_sentence_list = []
#     enlish_sentence_list = []
#     # 打开excel
#     wb = xlrd.open_workbook(path)
#     # 按工作簿定位工作表
#     sh = wb.sheet_by_name('维修手册详细步骤')
#     # print(sh.nrows)  # 有效数据行数
#     # print(sh.ncols)  # 有效数据列数
#     # print(sh.cell(0, 0).value)  # 输出第一行第一列的值
#     # print(sh.row_values(0))  # 输出第一行的所有值
#     for i in range(sh.nrows):
#         # 将数据和标题组合成字典
#         xlsx_dict = dict(zip(sh.row_values(0), sh.row_values(i)))
#         # 遍历excel，打印所有数据
#         content_id = str(xlsx_dict['FRT_CODE']).replace("更正代码", "") + "-" + str(xlsx_dict['序号'])
#         chinese_maintenance = xlsx_dict['维修内容（中文）']
#         chinese_sentence_list.append(chinese_maintenance)
#         english_maintenance = xlsx_dict['维修内容（英文）']
#         enlish_sentence_list.append(english_maintenance)
#
#     return chinese_sentence_list[1:], enlish_sentence_list[1:]

# if __name__ == "__main__":
#
#     file_path = os.path.join(base_dir, "Model3维修手册.xls")
#     english_file_path = os.path.join(base_dir, "en.txt")
#     chinese_file_path = os.path.join(base_dir,"ch.txt")
#     chinese_maintain_data, english_maintain_data = get_data_from_xlsx(file_path)
#     chinese_word_cut_list = [word_cut(sentence.replace("\n",""), "ch") for sentence in tqdm(chinese_maintain_data)]
#     english_word_cut_list = [word_cut(sentence.replace("\n",""), "en") for sentence in tqdm(english_maintain_data)]
#     write_text(chinese_file_path, chinese_word_cut_list)
#     write_text(english_file_path, english_word_cut_list)

# todo 使用fast-align方法进行单词对齐，效果不佳。有时间在研究
# en2ch_file_path = os.path.join(base_dir, "en2ch")
# fast_align_en2ch_list = fast_align_data(english_word_cut_list,chinese_word_cut_list)
# write_text(en2ch_file_path, fast_align_en2ch_list)

if __name__ == "__main__":
    file_path = os.path.join(base_dir, "Modle3-中英文官方文档.xlsx")
    english_file_path = os.path.join(base_dir, "source.txt")
    chinese_file_path = os.path.join(base_dir, "target.txt")
    awesome_align_data_file_path = os.path.join(base_dir, "awesome_align_data.txt")
    chinese_maintain_data, english_maintain_data = get_data_from_xlsx(file_path)
    chinese_list = [sub_word(ch_content) for ch_content in chinese_maintain_data]
    english_list = [sub_word(en_content) for en_content in english_maintain_data]

    chinese_word_cut_list = [word_cut(sentence.replace("\n", ""), "ch").lower() for sentence in tqdm(chinese_list)]
    english_word_cut_list = [word_cut(sentence.replace("\n", ""), "en").lower() for sentence in tqdm(english_list)]



    assert len(chinese_word_cut_list) == len(english_word_cut_list)
    write_text(chinese_file_path, chinese_word_cut_list)
    write_text(english_file_path, english_word_cut_list)
    write_text(awesome_align_data_file_path, english_word_cut_list)
