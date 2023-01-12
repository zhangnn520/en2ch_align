#!/usr/bin/python
# --*-- coding: utf-8 --*--
# @Author  : znn 
# @Email   : m18917010972@163.com
# @Time    : 2022/8/25 18:37
# @File    : nltk_test.py
# @Software: PyCharm
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
text=word_tokenize('And now for something completely different')
print(pos_tag(text))

