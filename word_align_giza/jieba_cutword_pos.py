#!/usr/bin/python
# --*-- coding: utf-8 --*--
import os
import re
import jieba
import jieba.posseg as psg
from tqdm import tqdm
import nltk
nltk.download()


base_dir = os.getcwd()
chinese_stop_path = os.path.join(base_dir, "data", "stop_word", "chinese_stop_words.txt")
jieba_user_defined_word_path = os.path.join(base_dir, "data", "new_dict", "汽车维修词典.txt")
# jieba.load_userdict(jieba_user_defined_word_path)

# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# # 对句子进行分词
def seg_sentence(sentence, use_paddle=False):
    sentence_pos_token_list = []
    jieba.load_userdict(jieba_user_defined_word_path)
    # 启动paddle模式。
    # paddle.enable_static()
    # jieba.enable_paddle()
    sentence_seged = psg.cut(sentence[0].strip(), use_paddle=use_paddle)
    stop_words_list = stopwordslist(chinese_stop_path)  # 这里加载停用词的路径
    for word, pos in sentence_seged:
        if word not in stop_words_list and chinese_character_judgment(word):
            sentence_pos_token_list.append("{0}/{1}".format(word, pos))
    return {"ID": sentence[1], "sentence": sentence[0], "pos_result": sentence_pos_token_list}


def re_han_internal(sentence):
    pattern = re.compile(u"([\u4e00-\u9fa5]+)")
    return pattern.match(sentence)


def chinese_character_judgment(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def jieba_pos(sentense_list):
    sentence_pos_list = []
    for sentence in tqdm(sentense_list):
        sentence_pos_list.append(seg_sentence(sentence))
    return sentence_pos_list
