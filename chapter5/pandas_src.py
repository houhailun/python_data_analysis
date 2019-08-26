#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/8/16 10:30
@Author  : Hou hailun
@File    : pandas_src.py
"""

print(__doc__)
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def pandas_data_struct():
    # pandas的数据结构介绍：Series, DataFrame
    # PART1：Series：类似于一维数组的对象，由一组数据和一组与之相关的索引组成
    # 创建方式1：数组创建
    obj = Series([4, 7, -5, 3])
    # print(obj)  # 索引在左边，数据在右边；没有指定索引，会自动创建一个0~N-1的整数索引
    # print(obj.values)
    # print(obj.index)

    obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])  # 创建带索引的series
    # print(obj2)
    # print(obj2.values)
    # print(obj2.index)

    # print(obj2['a'])  # 与普通numpy数组相比，可以通过索引的方式选取数据
    # print(obj2[['c', 'd', 'a']])

    # numpy数组运算会保留索引和值
    # print(obj2 > 0)
    # print(obj2[obj2 > 0])
    # print(obj2 * 2)

    # 可以把series看作一个定长的有序是字典：它是索引到数据的映射
    # print('b' in obj2)

    # 创建方式2：利用字典创建
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    obj3 = Series(sdata)
    # print(obj3)
    # print(obj3.values)
    # print(obj3.index)

    states = ['California', 'Ohio', 'Oregon', 'Texas']
    obj4 = Series(sdata, index=states)  # 如果重新指定index，那么会在sdata中查找索引，如果匹配不上该索引对应值为Nan
    # print(obj4)  # California在sdata中找不到，则其值为Nan
    # print(pd.isnull(obj4))   # 检测每条数据是否缺失
    # print(pd.notnull(obj4))  # 非缺失
    # print(obj4.isnull())

    # Series重要功能：算数运算中会自动对齐不同索引数据
    # print(obj3 + obj4)

    # series对象本身及其索引都有name属性
    obj4.name = 'population'
    obj4.index.name = 'state'
    # print(obj4)

    # 索引赋值
    obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
    # print(obj)

    # PART2: DataFrame
    # 创建方式1：字典
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
    frame = DataFrame(data)
    # print(frame)  # 3列: state, year, pop; 自动添加索引；列自动有序排列
    # print(DataFrame(data, columns=['year', 'state', 'pop']))  # 指定列顺序

    frame2 = DataFrame(data,
                       columns=['year', 'state', 'pop', 'debt'],
                       index=['ones', 'two', 'three', 'four', 'five'])
    # print(frame2)  # 传入的列找不到数据，产生NaN

    # print(frame2['state'])  # 获取df的指定列--字典方式
    # print(frame2.state)     # --属性方式
    # print(frame2.ix['three'])  # ix来获取指定行

    # frame2['debt'] = 16.5
    # print(frame2)  # 列赋值

    frame2['eatern'] = frame2.state == 'Ohio'
    # print(frame2)  # 为不存在的列赋值会创建一个新列
    # del frame2['eatern']  # del删除列
    # print(frame2.columns)

    # 创建方式2: 字典嵌套
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
           'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    frame3 = DataFrame(pop)
    # print(frame3)  # 外层字典的键作为列,内层字典的键作为行索引

    frame3.index.name = 'year'
    frame3.columns.name = 'state'
    # print(frame3)  # 设置name属性

    # print(frame3.values)  # 二维series形式

    # PART3: 索引对象: 负责管理轴标签和其他元数据(如轴名称等)
    obj = Series(range(3), index=['a', 'b', 'c'])
    print(obj.index)
    # obj.index[1] = 'b'  # Index对象是不可修改的

    # index的不可修改性,保证index对象在多个数据结构之间安全共享
    index = pd.Index(np.arange(3))
    obj2 = Series([1, 2, 3], index=index)
    print(obj2)
    print(index is obj2.index)


def panda_basci_function():
    # 基本功能
    # PART1: 重新索引 reindex
    obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
    obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])  # reindex会根据新索引进行重排,某个索引值不存在就引入Nan
    obj3 = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
    # print(obj2)
    # print(obj3)

    obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
    # print(obj3)
    # print(obj3.reindex(range(6), method='ffill'))
    # print(obj3.reindex(range(6), method='bfill'))
    # method: ffill/pad --前向填充/搬运  bfill/backfill --后向填充/搬运

    # DF,reindex可以修改行索引,列,或者两个都修改;如果仅传入一个序列,则会重新索引行
    frame = DataFrame(np.arange(9).reshape((3, 3)),
                      columns=['Ohio', 'Texas', 'California'],
                      index=['a', 'c', 'd'])
    frame2 = frame.reindex(['a', 'b', 'c', 'd'])  # 仅仅重新索引行
    # print(frame2)

    states = ['Texas', 'Utah', 'California']
    # print(frame.reindex(columns=states))  # 对列重新索引

    # print(frame.reindex(index=['a', 'b', 'c', 'd'], columns=states))  # 同时重新索引行列
    # print(frame.ix[['a', 'b', 'c', 'd'], states])

    # PART2: 丢弃指定轴上的项
    obj = Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
    new_obj = obj.drop('c')  # 删除索引c
    # print(new_obj)
    # print(obj.drop(['c', 'd']))

    data = DataFrame(np.arange(16).reshape((4, 4)),
                     index=['Ohio', 'Colorado', 'Utah', 'New York'],
                     columns=['one', 'two', 'three', 'four'])
    # print(data.drop(['Colorado', 'Ohio']))  # 删除行索引
    # print(data.drop('two', axis=1))  # axis删除列上的索引
    # print(data.drop(['two', 'four'], axis=1))

    # PART3: 索引,选取和过滤
    obj = Series(np.arange(4), index=['a', 'b', 'c', 'd'])
    # print(obj['b'])  # 标签索引
    # print(obj[1])    # 数字索引
    # print(obj[2:4])
    # print(obj[['b', 'c']])

    # print(obj['b': 'c'])  # 标签切片,末端是包含的

    # print(data['two'])  # 对DF索引其实就是获取一个或多个列
    # print(data[['three', 'one']])
    # print(data[:2])     # 切片选取行
    # print(data[data['three'] > 5])  # 布尔数组选取行
    # print(data.ix[:2])
    # print(data.ix['Colorado'])  # 行标签选择指定行
    # print(data.ix['Colorado', ['two', 'three']])  # 行标签 & 列索引
    # print(data.ix[:, 'two'])  # two列
    """
    obj[val]: 选取df的一个或一组列
    obj.ix[val]: 选取dc的单个行或一组行
    obj.ix[:, val]: 选取单个列或列子集
    obj.ix[val1, val2]: 同时选取行和列
    loc：通过行标签索引数据
    iloc：通过行号索引行数据
    ix：通过行标签或行号索引数据（基于loc和iloc的混合） 
        df.ix[0] -- 行号索引
        dc.ix['a'] -- 行标签索引
    """

    # PART4: 算数运算和数据对齐
    s1 = Series([1, 2, 3, 4], index=['a', 'c', 'd', 'e'])
    s2 = Series([5, 6, 7, 8, 9], index=['a', 'c', 'e', 'f', 'g'])
    # print(s1 + s2)  # 相加时结果为索引对的并集,自动对齐使得在不重复的索引处引入Na

    # df1 = DataFrame(np.arange(9).reshape((3, 3)),
    #                 columns=list('bcd'),
    #                 index=['Ohio', 'Texas', 'Colorado'])
    # df2 = DataFrame(np.arange(12).reshape((4, 3)),
    #                 columns=list('bde'),
    #                 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    # print(df1 + df2)  # df对齐会同时发生在行和列上

    # 在算数方法中填充值: add(),sub(),div(),mul()
    df1 = DataFrame(np.arange(12).reshape((3, 4)), columns=list('abcd'))
    df2 = DataFrame(np.arange(20).reshape((4, 5)), columns=list('abcde'))
    # print(df1.add(df2, fill_value=0))  # 调用df的add()
    # print(df1.reindex(columns=df2.columns, fill_value=0))  # 重新索引

    # DF和Series之间的运算
    arr = np.arange(12).reshape((3, 4))
    # print(arr - arr[0])  # 广播

    frame = DataFrame(np.arange(12).reshape((4, 3)),
                      columns=list('bde'),
                      index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    # series = frame.ix[0]
    # print(frame - series)  # series的索引匹配到df的列,然后沿着行一直向下广播

    # 列相减,即匹配行且在列上广播
    series = frame['d']
    # print(frame.sub(series, axis=0))  # axis就是希望匹配的轴,行为轴0,列为轴1

    # PART5: 函数应用和映射
    frame = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    # print(np.abs(frame))  # numpy的元素级函数可以直接使用
    f = lambda x: x.max() - x.min()
    # print(frame.apply(f))  # 使用apply将函数应用到各行或列上
    # print(frame.apply(f, axis=1))
    format = lambda x: '%.2f' % x
    # print(frame.applymap(format))  # 元素级的python函数通过applymap()
    # print(frame[0].map(format))

    # PART6: 排序和排名
    obj = Series(range(4), index=['d', 'a', 'b', 'c'])
    # print(obj.sort_index())   # 对行索引进行排序
    # print(obj.sort_values())  # 对值进行排序

    frame = DataFrame(np.arange(8).reshape((2, 4)),
                      index=['three', 'one'],
                      columns=['d', 'a', 'b', 'c'])
    # print(frame.sort_index())  # 对行轴进行排序,即一行中的数据是由小到大的
    # print(frame.sort_index(axis=1))  # 对列轴进行排序.即一列中的数据是由小到大的
    # print(frame.sort_index(by='b'))  # 对某个列进行排序

    # 排名ranking和排序关系密切,会增加一个排名值(从1开始,一直到数据中有效数据的数量),和numpy的argsort()类似,只不过
    # ranking可以根据某种规则破坏评级关系
    obj = Series([1, 2, 3, 4])
    print(obj.rank())  # 现在值表示:在原来obj这个序列中，0-3这4个索引所对应的每一个值分别在序列里排名第几。

    obj = Series([1,1,2,2,3,4])
    print(obj.rank())
    # 索引0和索引1对应的值均为1，按照上面的说法，调用rank()方法后，他们的排名分别是第1位，和第2位，那么究竟是索引0对应的值是第1，还是索引1对应的值是第1呢？
    # rank函数的默认处理是当出现重复值的情况下，默认取他们排名次序值（这里的第1名、第2名）的平均值。也就是说索引0和索引1对应的值1统一排名为（1+2）/2 = 1.5。
    # method      说明
    # average     默认:在相等分组中,为各个值分配平均排名
    # min         使用整个分组的最小排名(两人并列第 1 名，下一个人是第 3 名。 )
    # max         使用整个分组的最大排名(两人并列第 2 名，下一个人是第 3 名)
    # first       按值在原始数据中的出现顺序分配排名

    # PART7: 带重复值的轴索引
    obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
    print(obj)
    print(obj.index.is_unique)  # 索引是否唯一
    print(obj['a'])  # 索引a对应2个值，返回series
    print(obj['c'])  # 索引c对应1个值，返回标量


def pandas_groupby_describe():
    # 汇总和描述性统计
    df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
                   index=['a', 'b', 'c', 'd'],
                   columns=['one', 'two'])
    # print(df)
    # print(df.sum())
    # print(df.sum(axis=1))
    # print(df.mean(axis=1, skipna=False))
    # 简约方法的选项
    # axis: 约简的轴，df的行用0，列用1
    # skipna: 排除缺失值，默认为True
    # level: 如果轴是层次化索引的，则根据level而分组简约

    # print(df.idxmax())  # 最大值的索引
    # print(df.idxmin())

    # print(df.cumsum())    # 累计型
    # print(df.describe())  # 描述性统计
    # print(df.head(2))
    # print(df.count())
    # print(df.max())
    # 描述和汇总统计
    # count: 非NA的数量   describe: 针对series或df列计算汇总统计    min\max: 计算最大值和最小值
    # argmin\argmax: 最小值/最大值的索引位置   idxmax/idxmin: 最大值/最小值的索引值
    # quantile: 样本的分位数  sum: 求和     mean: 平均值   median: 中位数
    # mad: 根据平均值计算平均绝对离差    var: 方差     std: 标准差   cumsum: 累计和  diff: 一阶差分

    # PART2:相关系数与协方差
    # PART3: 唯一值、值计算以及成员资格
    obj = Series(['c', 'a', 'd', 'a', 'b', 'c'])
    # print(obj.unique())  # 得到唯一值数组，类似去重
    # print(obj.value_counts())  # 计算series中各值出现的频率
    # print(obj.isin(['b', 'c']))  # 判断矢量化集合的成员资格，obj中各个元素是否是['b','c']的成员

    data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                      'Qu2': [2, 3, 1, 2, 3],
                      'Qu3': [1, 5, 2, 4, 4]})
    # print(data)
    # print(data.apply(pd.value_counts)).fillna(0)


def pandas_nan():
    # 处理缺失数据
    string_data = Series(['a', 'b', np.nan, 'c'])
    # print(string_data)
    # print(string_data.isnull())
    # string_data[0] = None  # None也会被当作NA
    # print(string_data.isnull())
    # NA处理方式
    # dropna: 删除NA  fillna: 填充  isnull:检查是否为NA  notnull: isnull的否定式
    # PART3_1: 过滤缺失数据
    from numpy import nan as NA
    data = Series([1, NA, 3.5, NA, 7])
    # print(data.dropna())         # 删除NA
    # print(data[data.notnull()])  # 布尔型数组实现删除NA

    # df的过滤涉及到行、列、全部
    data = DataFrame([[1, 6.5, 3], [1, NA, NA], [NA, NA, NA], [NA, 6.5, 3]])
    # print(data)
    # cleaned = data.dropna()  # 默认删除任何含有缺失值的行
    # print(cleaned)
    # cleaned_2 = data.dropna(how='all')  # how='all'表示只丢弃全为NA的行
    # print(cleaned_2)
    # cleaned_3 = data.dropna(axis=1, how='all')  # axis=1表示针对列丢弃
    # print(cleaned_3)

    # 时间序列的过滤
    df = DataFrame(np.random.randn(7, 3))
    df.ix[:4, 1] = NA
    df.ix[:2, 2] = NA
    # print(df.dropna(thresh=3))  # thresh=3,保留至少有3个非空值的数据行

    # PART3_2：填充缺失数据fillna()
    print(df.fillna(0.0))  # 使用常数替换NA
    print(df.fillna({1: 0.5, 2: -1}))  # 字典形式实现对不同列填充不同值
    _ = df.fillna(0, inplace=True)
    print(df)  # inplace默认返回新对象，通过inplace=True可以就地修改
    # fillna()的参数:
    # value: 用户填充缺失值的标量或字典对象   method: 插值方式，默认ffill
    # axis: 待填充的轴，默认为0              inplace: 修改调用者，不返回副本
    # limit: 限制填充数量


def pandas_index():
    # 层次化索引，一个轴上有多个索引级别
    data = Series(np.random.randn(10),
                  index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                         [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
    # print(data)
    # print(data.index)
    # print(data['b'])  # 外层索引
    # print(data['b': 'c'])
    # print(data[:, 2])  # 内层索引
    # print(data.unstack())  # 转换为df
    # print(data.unstack().stack())

    # df, 每条轴可以有分层索引
    frame = DataFrame(np.arange(12).reshape((4, 3)),
                      index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                      columns=[['Ohio', 'Ohio', 'BJ'],
                               ['Green', 'Red', 'Green']])
    frame.index.names = ['key1', 'key2']
    frame.columns.name = ['state', 'color']
    # print(frame)
    # print(frame['Ohio'])

    # 重排分级顺序
    # print(frame.swaplevel('key1', 'key2'))  # 交换行索引的顺序
    # print(frame.sortlevel(1))


if __name__ == "__main__":
    # pandas_data_struct()
    # panda_basci_function()
    # pandas_groupby_describe()
    # pandas_nan()
    pandas_index()