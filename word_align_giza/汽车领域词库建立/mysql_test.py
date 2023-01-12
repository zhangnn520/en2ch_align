# coding=utf-8
import re
import json
import pymysql
from datetime import date, datetime


def write_text(content_line, write_path):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.write(content + "\n")


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)


def write_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        writer.writelines(json.dumps(content_line, ensure_ascii=False, indent=4, cls=ComplexEncoder))


class MysqlSearch:
    def __init__(self,table_name,db_name):
        self.table_name = table_name
        self.db_name = db_name
        try:
            self.conn = pymysql.connect(host="192.168.1.181",
                                        user="bigData",
                                        password="P@ssw0rd654321",
                                        # db="warehouse",
                                        db=self.db_name,
                                        charset="utf8")

            self.cursor = self.conn.cursor()
        except pymysql.err as e:
            print("error" % e)

    def get_one(self):
        sql = "select * from api_test where method = %s order by id desc"
        self.cursor.execute(sql, ('post',))
        # print(dir(self.cursor))
        # print(self.cursor.description)
        # 获取第一条查询结果，结果是元组
        rest = self.cursor.fetchone()
        # 处理查询结果，将元组转换为字典
        result = dict(zip([k[0] for k in self.cursor.description], rest))
        self.cursor.close()
        return result

    def get_all(self):
        sql = f"select * from {self.table_name}"
        self.cursor.execute(sql)
        # 获取第一条查询结果，结果是元组
        rest = self.cursor.fetchall()
        # 处理查询结果，将元组转换为字典
        result = [dict(zip([k[0] for k in self.cursor.description], row)) for row in rest]
        self.cursor.close()
        return result

    def db_close(self):
        self.conn.close()


# if __name__ == '__main__':
#     obj = MysqlSearch()
#     reslts = obj.get_all()
#     write_json(r"H:\汽车领域词库建立\data\术语词汇.json", reslts)
