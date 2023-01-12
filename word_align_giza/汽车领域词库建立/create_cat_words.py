# coding=utf-8

import os
import json

base_dir = os.getcwd()


def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return reader.read()


def write_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            if content:
                writer.write(content + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.loads(reader.read())


# {
# "ID": "898bd1ad12de11ed87c40242ac110005",
# "ENG_NAME": "Cab-over-engine truck",
# "CN_NAME": null,
# "ABBREVIATION": null,
# "ALIAS": null,
# "TYPE": null,
# "CREATED_TIME": "2022-08-03 11:44:55",
# "UPDATED_TIME": "2022-08-03 11:44:56"
# }


if __name__ == "__main__":
    terminology_words_path = os.path.join(base_dir, "data", "术语词汇.json")
    terminology_word_list = read_json(terminology_words_path)
    stop_word_list = ['；', "，", "（", "）", "(", ")"]
    abandon_words_list = []
    all_word_list = []
    for word in terminology_word_list:
        all_word_list.append(word['CN_NAME'])
        for stop_word in stop_word_list:
            if stop_word in word['CN_NAME']:
                abandon_words_list.append(word['CN_NAME'])
    result_word_list = [i for i in all_word_list if i not in abandon_words_list]
    cat_component_path = os.path.join(base_dir, "data", "new_dict", "汽车维修词典.txt")
    path1 = os.path.join(base_dir, "data", "origin_data", "car_dict.txt")
    data1 = [x1 for x1 in read_text(path1).split('\n')]
    path2 = os.path.join(base_dir, "data", "origin_data", "特斯拉系列零件词库.txt")
    data2 = [x2 for x2 in read_text(path2).split('\n')]
    path3 = os.path.join(base_dir, "data", "origin_data", "THUOCL_car.txt")
    data3 = read_text(path3)
    new2 = [i.split('\t')[0] for i in data3.split('\n')]
    path4 = os.path.join(base_dir, "data", "origin_data", "汽车行业.txt")
    data4 = read_text(path4)
    new3 = [j.split('\t')[0] for j in data4.split('\n')]
    newlist = data1 + data2 + new2 + new3 + result_word_list
    new = [x for x in list(set(newlist)) if ";" not in x and " " not in x]

    print("汽车维修领域初版词库表，具有词语{0}个".format(len(list(set(newlist)))))
    write_text(cat_component_path, new)
