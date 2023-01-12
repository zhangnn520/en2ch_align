# coding=utf-8
import os
import re
import json
import jieba
import paddle
import pymysql
from tqdm import tqdm
import xlsxwriter
from nltk.tokenize import word_tokenize

base_dir = os.getcwd()
# 用户自定义词典



def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return reader.readlines()


def read_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.loads(reader.read())


def get_content(object_list, content_name="CONTENT"):
    return [content[f'{content_name}'] for content in object_list]


def write_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            if content != content_line[-1]:
                writer.write(content.replace("  "," ") + "\n")
            else:
                writer.write(content)


def write_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        writer.writelines(json.dumps(content_line, ensure_ascii=False))


def write_line_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for line in content_line:
            writer.write(json.dumps({line: content_line[line]}, ensure_ascii=False))
            writer.write("\n")


class MysqlSearch:
    def __init__(self):
        try:
            self.conn = pymysql.connect(host="192.168.1.181",
                                        user="bigData",
                                        password="P@ssw0rd654321",
                                        db="bigData",
                                        charset="utf8")

            self.cursor = self.conn.cursor()
        except pymysql.err as e:
            print("error" % e)

    # def get_one(self):
    #     sql = "select * from api_test where method = %s order by id desc"
    #     self.cursor.execute(sql, ('post',))
    #     # print(dir(self.cursor))
    #     # print(self.cursor.description)
    #     # 获取第一条查询结果，结果是元组
    #     rest = self.cursor.fetchone()
    #     # 处理查询结果，将元组转换为字典
    #     result = dict(zip([k[0] for k in self.cursor.description], rest))
    #     self.cursor.close()
    #     return result

    def get_all(self, tabel_name):
        sql = f"SELECT * FROM {tabel_name} ORDER BY TITLE"
        self.cursor.execute(sql)
        # 获取第一条查询结果，结果是元组
        rest = self.cursor.fetchall()
        # 处理查询结果，将元组转换为字典
        result = [dict(zip([k[0] for k in self.cursor.description], row)) for row in rest]
        return result

    def db_close(self):
        # 关闭游标
        self.cursor.close()
        # 关闭库链接
        self.conn.close()


def get_maintain_clear_data(english_maintain, chinese_maintain):
    english_clear_data_list, chinses_clear_data_list = list(), list()
    for en in tqdm(english_maintain):
        for ch in chinese_maintain:
            if en['STATISTICS'] == ch['STATISTICS'] and en['HTML'].split("/")[-1] == ch['HTML'].split("/")[-1]:
                english_clear_data_list.append(en)
                chinses_clear_data_list.append(ch)
    try:
        assert len(english_clear_data_list) == len(chinses_clear_data_list)
    except Exception as e:
        print(e)
    return english_clear_data_list, chinses_clear_data_list


# def word_cut(content, lan_type="ch"):
#     user_defined_word_path = r"E:\ENOCH-2022\NLP_MODEL\07_machine_transformer\word_align_giza\data\new_dict\汽车维修词典.txt"
#     jieba.load_userdict(user_defined_word_path)
#     if lan_type == "ch":
#         paddle.enable_static()
#         seg_list = jieba.cut(content, use_paddle=True)  # 使用paddle模式进行分词
#     else:
#         seg_list = word_tokenize(content)
#     return ' '.join(seg_list)


def fast_align_data(en_list, ch_list):
    en2ch_list = list()
    assert len(en_list) == len(ch_list)
    for num, ch_content in enumerate(ch_list):
        en_content = en_list[num]
        new_content = en_content + ' ||| ' + ch_content
        en2ch_list.append(new_content)
    return en2ch_list


def get_fast_align_data(en2chcontent, en2ch_align_content, word_align_result_path):
    """
    :param en2chcontent: 对齐原始预料
    :param en2ch_align_content: 对齐后产生的映射字典
    :param word_align_result_path: 写入对齐映射的文件目录
    :return: null
    """
    result_list = []
    for num, content in enumerate(en2chcontent):
        ens, chs = content.split("|||")
        id2en = {str(num): en for num, en in enumerate(ens.split(" "))}
        id2ch = {str(num): ch for num, ch in enumerate(chs.split(" "))}
        content_align = en2ch_align_content[num]
        en_word_mapping = [id2en[i.split("-")[0]] for i in content_align.replace("\n", "").split(" ")]
        ch_word_mapping = [id2ch[i.split("-")[1]] for i in content_align.replace("\n", "").split(" ")]
        result = dict(zip(en_word_mapping, ch_word_mapping))
        result_list.append(result)
    write_json(word_align_result_path, result_list)


def write_sheet(path, new_table, datas):
    """
    向表中写入操作
    new_table: 要写入内容的新的工作表
    datas: 要写入的内容列表
    """
    '''创建excel文件'''
    xl = xlsxwriter.Workbook(path)
    '''添加工作表'''
    sheet = xl.add_worksheet(new_table)
    '''向单元格cell中添加数据，写入索引（标题）'''
    for num, data in enumerate(datas):
        sheet.write_string("A", json.dumps(data, ensure_ascii=False))


def sub_word(string):
    pattern_one = u"\\(.*?\\)|\\（.*?）|\\[.*?]|-|“|”|``|''"
    result = re.sub(pattern_one, "", string)
    pattern_two = u" +"
    result = re.sub(pattern_two, " ", result)

    return result
