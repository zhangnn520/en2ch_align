#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import argparse
from tqdm import tqdm
from tools.tools import read_text, write_json, delete_duplicate_elements

base_dir = os.getcwd()


def parser_conf():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--test_file_path",
        default=os.path.join(base_dir, "../data/model_data/test.txt"),
        type=str,
        help="测试集地址文件"
    )
    parser.add_argument(
        "--align_result_file_path",
        type=str,
        default=os.path.join(base_dir, "align_result.txt"),
        help="对齐映射文件"
    )
    parser.add_argument(
        "--output_prob_file_path",
        default=os.path.join(base_dir, "output_prob_file.txt"),
        type=str, help='模型预测后中英文对齐的概率文件'
    )
    parser.add_argument(
        "--output_word_file_path",
        default=os.path.join(base_dir, "output_word_file.txt"),
        type=str, help='中英文对齐文件'
    )
    parser.add_argument(
        "--result_file_path",
        default=os.path.join(base_dir, "result_file_path.json"),
        type=str, help='中英文对齐文件'
    )
    args = parser.parse_args()
    return args


class GetAlignFile(object):
    def __init__(self, args):
        self.test_file_path = args.test_file_path
        self.align_result_file_path = args.align_result_file_path
        self.output_prob_file_path = args.output_prob_file_path
        self.output_word_file_path = args.output_word_file_path
        self.result_file_path = args.result_file_path

    def read_file(self):
        test_data = read_text(self.test_file_path)
        align_result = read_text(self.align_result_file_path)
        output_prob_file_data = read_text(self.output_prob_file_path)
        output_word_data = read_text(self.output_word_file_path)
        assert len(test_data) == len(align_result)
        assert len(align_result) == len(output_prob_file_data) == len(output_word_data)
        return test_data, align_result, output_prob_file_data, output_word_data

    def write_file(self):
        write_data = self.statistical_merger()
        write_json(self.result_file_path, write_data)

    @staticmethod
    def get_align_collocation(data_content, split_symbol):
        data_content = [i.replace("\n","") for i in data_content.strip().split(split_symbol)]
        return data_content

    @staticmethod
    def get_same_index(data_list, align_en):
        list_same = []
        for i in data_list:
            address_index = [align_en[x] for x in range(len(data_list)) if data_list[x] == i]
            list_same.append([i, address_index])
        list_same = delete_duplicate_elements(list_same)
        result_dict = {i[0]: " ".join(i[1]) for i in list_same}
        return result_dict

    @staticmethod
    def merger_words(ch_list, en_list, align_list):
        # 将中文词语对应的英文单词进行合并
        align_ch = [ch_list[int(X.split("-")[0])] for X in align_list]
        align_en = [en_list[int(Y.split("-")[1])] for Y in align_list]
        result_dict = GetAlignFile.get_same_index(align_ch, align_en)

        return result_dict

    def statistical_merger(self):
        # 对三个模型输出的数据进行处理，并生成可读性较好的文件
        data_list = []
        test_data_list, align_result_list, prob_data_list, word_data_list = self.read_file()
        for num, test_data in tqdm(enumerate(test_data_list)):
            try:
                data_dict = dict()
                ch_content, en_content = test_data.split(' ||| ')
                data_dict['chinese'] = ch_content.replace("\n", "")
                data_dict['english'] = en_content.replace("\n", "")
                data_dict['origin_result'] = str(align_result_list[num]).strip()
                align_result = GetAlignFile.get_align_collocation(align_result_list[num], " ")
                prob_data = GetAlignFile.get_align_collocation(prob_data_list[num], " ")
                word_data = GetAlignFile.get_align_collocation(word_data_list[num], " ")
                data_dict['align_result'] = align_result
                data_dict['orgin_align_result'] = dict(zip(word_data, prob_data))
                result_dict = GetAlignFile.merger_words(ch_content.replace("\n", "").split(" "),
                                                        en_content.replace("\n", "").split(" "),
                                                        align_result)
                data_dict['align_result'] = result_dict
                data_list.append(data_dict)
            except Exception as e:
                print(e)
                print("存在特殊字符的数据：\n", test_data)
        return data_list


if __name__ == "__main__":
    arg = parser_conf()
    get_align = GetAlignFile(arg)
    get_align.write_file()
