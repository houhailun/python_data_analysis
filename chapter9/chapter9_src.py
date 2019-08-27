#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/8/27 15:50
@Author  : Hou hailun
@File    : chapter9_src.py
"""

print(__doc__)
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# 数据聚和与分组运算
# 根据一个或多个键拆分pandas对象
# 计算分组摘要统计，如计数、平均值、标准差、用户自定义函数
# 对df的列应用各种函数
# 应用组内转换或其他运算，如规格化、线性回归、排名、选取子集等
# 计算透视表或交叉表
# 执行分位数分析以及其他分组分析


def pandas_groupby_part1():
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})

    # 按key1分组，计算data1列的平均值
    grouped = df['data1'].groupby(df['key1'])  # 访问data1，并根据key1调用groupby
    grouped2 = df.groupby(['key1'])['data1']   # 根据key1调用groupby，在访问data1
    # print(grouped.mean())  # 2中方式等价
    # print(grouped2.mean())  # 得到新的series，索引为key1的唯一值

    means = df['data1'].groupby([df['key1'], df['key2']]).mean()
    means2 = df.groupby(['key1', 'key2'])['data1'].mean()
    # print(means)  # 对2个键分组，得到一个层次化索引
    # print(means2)
    # print(means.unstack())  # 转换为df

    # 分组键为任何长度的数组
    states = np.array(['bj', 'cd', 'cd', 'bj', 'bj'])
    years = np.array([2005, 2005, 2006, 2005, 2006])
    # print(df['data1'].groupby([states, years]).mean())

    # 把列名作为分组键
    print(df.groupby('key1').mean())
    print(df.groupby(['key1', 'key2']).mean())

    print(df.groupby(['key1', 'key2']).size())  # size() 返回含有分组大小的series
    """
    首先通过groupby得到DataFrameGroupBy对象, 比如data.groupby('race')
    然后选择需要研究的列, 比如['age'], 这样我们就得到了一个SeriesGroupby, 它代表每一个组都有一个Series
    对SeriesGroupby进行操作, 比如.mean(), 相当于对每个组的Series求均值
    注: 如果不选列, 那么第三步的操作会遍历所有列, pandas会对能成功操作的列进行操作, 最后返回的一个由操作成功的列组成的DataFrame
    """

def pandas_groupby_part2():
    # 对分组进行迭代
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
    # groupBy对象支持迭代，产生一组二元元组
    for name, group in df.groupby('key1'):
        print(name)   # key1的唯一值
        print(group)  # 对应的分组

    print('-'*30)
    for (k1, k2), group in df.groupby(['key1', 'key2']):
        print(k1, k2)  # 多个key分组
        print(group)

    print('-' * 30)
    pieces = dict(list(df.groupby('key1')))
    print(pieces)

    print('-' * 30)
    # NOTE: groupby默认是在axis=0上进行分组的(行),也可以指定在其他轴上分组
    grouped = df.groupby(df.dtypes, axis=1)
    print(dict(list(grouped)))


def pandas_groupby_part3():
    # 选取一个或一组列
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})

    print(df.groupby(['key1', 'key2'])['data2'].mean())

    # 先用一个或一组列名进行索引，然后再选取部分列进行聚合


def pandas_groupby_part4():
    # 通过字典或series进行分组
    people = DataFrame(np.random.randn(5, 5),
                       columns=['a', 'b', 'c', 'd', 'e'],
                       index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])

    people.ix[2:3, ['b', 'c']] = np.nan
    print(people)

    mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
               'd': 'blue', 'e': 'red', 'f': 'orange'}

    print('-'*30)
    by_column = people.groupby(mapping, axis=1)
    print(by_column.sum())

    # series可以被看作固定大小的映射
    print('-' * 30)
    map_series = Series(mapping)
    print(people.groupby(map_series, axis=1).count())


def pandas_groupby_part5():
    # 通过函数进行分组
    people = DataFrame(np.random.randn(5, 5),
                       columns=['a', 'b', 'c', 'd', 'e'],
                       index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
    print(people.groupby(len).sum())


def pandas_groupby_part6():
    # 根据索引级别分组
    columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                         [1, 3, 5, 1, 3]], names=['city', 'tenor'])
    hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
    print(hier_df)
    print('-'*30)
    print(hier_df.groupby(level='city', axis=1))


def pandas_groupby():
    # # groupby技术
    # pandas_groupby_part1()
    # pandas_groupby_part2()
    # pandas_groupby_part3()
    # pandas_groupby_part4()
    # pandas_groupby_part5()
    pandas_groupby_part6()


def pandas_agg_part1():
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})

    grouped = df.groupby('key1')
    print(grouped['data1'].quantile(0.9))  # quantile分位数

    # 自定义聚合函数
    def peak_to_peak(arr):
        return arr.max() - arr.min()

    print(grouped.agg(peak_to_peak))

    print(grouped.describe().T)


def pandas_agg_part2():
    # 面向列的多函数应用
    tips = pd.read_csv('tips.csv')
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    print(tips[:6])

    grouped = tips.groupby(['sex', 'smoker'])
    grouped_pct = grouped['tip_pct']
    print(grouped_pct.agg('mean'))  # 可以把函数名以字符串形式传入
    print(grouped_pct.agg(['mean', 'std']))  # 传入一组函数，得到的df的列会以函数命名
    print(grouped_pct.agg([('foo', 'mean'), ('bar', np.std)]))  # 指定最后df的列为foo，bar

    # df,定义一组应用于全部列的函数
    functions = ['count', 'mean', 'max']
    result = grouped['tip_pct', 'total_bill'].agg(functions)
    print(result)  # 层次化列的df

    # 不同的列应用不同函数
    print(grouped.agg({'tip': np.max, 'size': 'sum'}))


def pandas_agg():
    # 数据聚合：从数组产生标量值的数据转换过程
    # pandas_agg_part1()
    pandas_agg_part2()


if __name__ == "__main__":
    # pandas_groupby()
    pandas_agg()