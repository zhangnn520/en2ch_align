#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：word_align_giza
@File    ：get_mysql_data.py
@Author  ：znn
@Date    ：2022/8/31 9:45
"""
import os
from tools.tools import *

base_dir = os.path.join(os.getcwd(), "data")
mysql_search = MysqlSearch()
# 从数据库中导出中英操作维修步骤

model3_maintain_web_english = mysql_search.get_all(tabel_name="MODEL3_MAINTAIN_WEB")

# 获取mysql中数据指定字段的数据
# english_maintain_content = get_content(model3_maintain_web_english)
# chinese_maintain_content = get_content(model3_maintain_web_chinese)

model3_maintain_web_chinese = mysql_search.get_all(tabel_name="MODEL3_MAINTAIN_WEB_CHINESE")

mysql_search.db_close()
maintain_english, maintain_chinese = get_maintain_clear_data(model3_maintain_web_english, model3_maintain_web_chinese)
write_text(os.path.join(base_dir, "new_data", "junjie_english.txt"), [maintain['CONTENT'] for maintain in maintain_english])
write_text(os.path.join(base_dir, "new_data", "junjie_chinese.txt"), [maintain['CONTENT'] for maintain in maintain_chinese])
write_json(os.path.join(base_dir, "new_data", "english_maintain_content.json"), maintain_english)
write_json(os.path.join(base_dir, "new_data", "chinese_maintain_content.json"), maintain_chinese)
