#!/usr/bin/python
# --*-- coding: utf-8 --*--
# @Author  : znn 
# @Email   : m18917010972@163.com
# @Time    : 2022/8/26 17:57
# @File    : 维修手册对齐代码.py
# @Software: PyCharm
from ltp import LTP
string = "松开将低压线束连接到顶楼HV电池线束支架的充电端口的夹子。"
ltp = LTP("LTP/small")  # 默认加载 Small 模型
output = ltp.pipeline([string], tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
# 使用字典格式作为返回结果
# print(output.cws)  # print(output[0]) / print(output['cws']) # 也可以使用下标访问
# print(output.pos)
# print(output.sdp)
print(output.ner)

# 使用感知机算法实现的分词、词性和命名实体识别，速度比较快，但是精度略低
ltp = LTP("LTP/legacy")
# cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "ner"]).to_tuple() # error: NER 需要 词性标注任务的结果
cws, pos, ner = ltp.pipeline([string], tasks=["cws", "pos", "ner"]).to_tuple()  # to tuple 可以自动转换为元组格式
# 使用元组格式作为返回结果
print(ner)