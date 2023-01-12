#!/usr/bin/python
# --*-- coding: utf-8 --*--
# @Author  : znn 
# @Email   : m18917010972@163.com
# @Time    : 2022/8/26 17:33
# @File    : ltp_test.py
# @Software: PyCharm

from ltp import LTP

model = LTP(path='pretrained_model/GSDSimp+CRF')
# 默认加载 small 模型
# model = LTP(path="small")
# path 可以为下载下来的包含ltp.model和vocab.txt的模型文件夹
# 也可以接受一些已注册可自动下载的模型名：
# base/base1/base2/small/tiny/GSD/GSD+CRF/GSDSimp/GSDSimp+CRF
sent_list = ['俄罗斯总统普京决定在顿巴斯地区开展特别军事行动。']

# 中文分词
seg, hidden = model.seg(sent_list)
print("中文分词",seg)
# 词性标注
pos = model.pos(hidden)
print("词性标注",pos)
# 命名实体识别
ner = model.ner(hidden)
print("词性标注",ner)
# 语义角色标注
srl = model.srl(hidden)
print("语义角色标注",srl)
# 依存句法分析
dep = model.dep(hidden)
print("依存句法分析",dep)
# 语义依存分析
sdp = model.sdp(hidden)
print("语义依存分析",dep)