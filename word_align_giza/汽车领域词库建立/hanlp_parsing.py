#!/usr/bin/python
# --*-- coding: utf-8 --*--
# @Author  : znn 
# @Email   : m18917010972@163.com
# @Time    : 2022/8/26 9:34
# @File    : hanlp_parsing.py
# @Software: PyCharm
import hanlp
hanlp.pretrained.mtl.ALL
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
doc = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
print(doc)