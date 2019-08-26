#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/8/19 15:40
@Author  : Hou hailun
@File    : chapter6_src.py
"""

print(__doc__)
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
"""
章节内容：数据加载、存储与文件格式
"""


def data_io_txt():
    # 读写文本格式的数据
    # read_csv(): 从文件、URL、文件型对象中加载数据，默认分隔符是逗号
    # read_table():从文件、URL、文件型对象中加载数据，默认分隔符是制表符\t
    # read_fwf(), read_clipboard():
    # 从文本数据转换为df过程中，以上函数可以分为：
    # 索引：将一个或多个列当作df，是否从文件、用户获取列明
    # 类型推断和数据转换： 包括用户自定义值的转换、缺失值标记列表
    # 日期解析：
    # 迭代：支持对大文件进行逐块迭代
    # 不规整数据问题：跳过一些行、页脚、注释等
    df = pd.read_csv('ex1.csv')
    # print(df)
    # print(pd.read_table('ex1.csv', sep=','))

    # print(pd.read_csv('ex2.csv'))  # 没有列名
    # print(pd.read_csv('ex2.csv', header=None))   # pd分配默认的列名
    # print(pd.read_csv('ex2.csv', names=['a', 'b', 'c', 'd', 'message']))  # 自定义列名

    names = ['a', 'b', 'c', 'd', 'message']
    # print(pd.read_csv('ex2.csv', names=names, index_col='message'))  # 指定列名，设置message为索引
    # print(pd.read_csv('csv_mindex.csv', index_col=['key1', 'key2']))  # 多层索引

    # result = pd.read_table('ex3.txt', sep='\s+')  # \s匹配任何空白字符，包括空格、制表符、换页符等等
    # print(result)

    # 逐块读取文本文件
    # result = pd.read_csv('ex6.csv')
    # print(pd.read_csv('ex6.csv', nrows=10))  # 读取10行
    # chunker = pd.read_csv('ex6.csv', chunksize=1024)  # 逐块读取
    # for piece in chunker:
    #     print('-'*20)

    # 将数据写到文本文件
    # print(df)
    # df.to_csv('out.csv')
    # df.to_csv(sys.stdout, sep='|')  # 指定分隔符
    # df.to_csv(sys.stdout, na_rep='NULL')  # 替换缺失值
    # df.to_csv(sys.stdout, index=False, header=False)  # 禁止写行/列标签
    # df.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])  # 选择部分列写文件
    #
    # dates = pd.date_range('1/1/2000', periods=7)
    # ts = Series(np.arange(7), index=dates)
    # ts.to_csv('tsereis.csv')
    # print(Series.from_csv('tsereis.csv', parse_dates=True))

    # 手工处理分隔符格式，针对畸形文件read_table出错的情况
    import csv
    # f = open('ex7.csv')
    # reader = csv.reader(f)
    # for line in reader:
    #     print(line)
    lines = list(csv.reader(open('ex7.csv')))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, values)}


if __name__ == "__main__":
    data_io_txt()