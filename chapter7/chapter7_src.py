#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/8/20 14:04
@Author  : Hou hailun
@File    : chapter7_src.py
"""

print(__doc__)
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

"""
章节内容：
    数据规整化：清理、转换、合并、重塑
"""


def pandas_union_data_part1():
    # 数据库风格的df合并
    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                     'data1': range(7)})
    df2 = DataFrame({'key': ['a', 'b', 'd'],
                     'data2': range(3)})
    # print(df1)
    # print(df2)
    # print(pd.merge(df1, df2))  # 如果没有指定连接的列，那么会把所有重叠列作为键
    # print(pd.merge(df1, df2, on='key'))

    df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                     'data1': range(7)})
    df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                     'data2': range(3)})
    # print(pd.merge(df3, df4, left_on='lkey', right_on='rkey'))  # 如果df没有相同的列名，也可以分别指定
    # print(pd.merge(df1, df2, how='outer'))  # outer取的是键的并集

    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                     'data1': range(6)})
    df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                     'data2': range(5)})
    # print(pd.merge(df1, df2, on='key', how='left'))  # 多对多连接产生的是行的笛卡尔集，即左右有3个’b'行，右边有2个‘b’行，最后有6个‘b’行

    left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                      'key2': ['one', 'two', 'one'],
                      'lval': [1, 2, 3]})
    right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                       'key2': ['one', 'one', 'one', 'two'],
                       'rval': [4, 5, 6, 7]})
    # print(pd.merge(left, right, on=['key1', 'key2'], how='outer'))  # 多个键合并，传入列表；多个键形成一系列元组，并将其当作单个连接键
    # print(pd.merge(left, right, on='key1'))  # 含有重复列名
    # print(pd.merge(left, right, on='key1', suffixes=('_left', '_right')))  # suffixes指定附加到左右df的重复列名上的字符串
    """
    merge()的参数说明
    参数          说明
    left        参与合并的左侧df
    right       参与合并的右侧df
    how         inner,outer,left,right;默认为inner
    on          用于连接的列名。必须存在于左右df对象中；未指定则以left和right列名的交集作为连接键
    left_on     左侧df用作连接键的列
    right_on    右侧df用作连接键的列
    left_index  将左侧的行索引用作其连接键
    right_index 类似left_index
    sort        对合并后的数据进行排序，大数据集时禁用该选项可以得到更好的性能
    suffixes    字符串值元组，用于追加到重叠列名的末尾，默认为('_x', '_y')
    copy        False: 避免将数据复制到结果数据结构中；默认总是复制 
    """


def pandas_union_data_part2():
    # 索引上的合并
    # df的连接键位于索引上，指定left_index=True或right_index=True(或2个都指定)来说明索引被用作连接键
    left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
    right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
    # print(pd.merge(left1, right1, left_on='key', right_index=True))  # left1以key作为连接键，right1以索引作为连接键
    # print(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))

    left2 = DataFrame([[1, 2], [3, 4], [5, 6]], index=['a', 'c', 'e'], columns=['Ohio', 'Nevada'])
    right2 = DataFrame([[7, 8], [9, 10], [11, 12], [13, 14]], index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
    # print(pd.merge(left2, right2, left_index=True, right_index=True))  # 同时使用2个df的索引
    # print(pd.merge(left2, right2, left_index=True, right_index=True, how='outer'))

    # join实例方法: 更方便的实现按索引合并
    # print(left2.join(right2, how='outer'))


def pandas_union_data_part3():
    # 轴向连接：连接、绑定、堆叠
    arr = np.arange(12).reshape((3, 4))
    # print(np.concatenate([arr, arr]))           # 行堆叠
    # print(np.concatenate([arr, arr], axis=1))   # 列堆叠

    s1 = Series([0, 1], index=['a', 'b'])
    s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
    s3 = Series([5, 6], index=['f', 'g'])
    # print(pd.concat([s1, s2, s3]))  # 增加行
    # print(pd.concat([s1, s2, s3], axis=1))  # 增加行列，构成df

    s4 = pd.concat([s1*5, s3])
    # print(pd.concat([s1, s4], axis=1))  # 另一条轴上没有重叠，从索引的有序并集
    # print(pd.concat([s1, s4], axis=1, join='inner'))
    # print(pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']]))  # 指定使用的索引

    # 参与连接的数据在结果中的连续的，无法区分，如果想在连接轴上创建层次化索引，使用key参数即可
    result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
    result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'], axis=1)  # keys变为df的列头
    # print(result)

    # unstack()
    # print(result.unstack())

    df1 = DataFrame(np.arange(6).reshape((3, 2)), index=['a', 'b', 'c'], columns=['one', 'two'])
    df2 = DataFrame(5 + np.arange(4).reshape((2, 2)), index=['a', 'b'], columns=['three', 'four'])
    # print(pd.concat([df1, df2]))
    # print(pd.concat([df1, df2], axis=1))
    # print(pd.concat([df1, df2], axis=1, keys=['level1', 'level2']))
    # print(pd.concat({'level1': df1, 'level2': df2}, axis=1))  # 字典的键当作keys选项

    df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
    df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
    print(pd.concat([df1, df2], ignore_index=True))
    print(pd.concat([df1, df2], ignore_index=False))
    """
    concat()参数说明：
    objs: 参与连接的pd对象的列表或字典，唯一必须的参数   axis: 指明连接的轴向，默认为0
    join: inner/outer，默认为inner。指明其他轴向上的索引是按交集inner还是并集outer进行合并
    ...
    """


def pandas_union_data_part4():
    # 合并重叠数据
    a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
    b = Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
    b[-1] = np.nan
    # print(np.where(pd.isnull(a), b, a))  # a为nan则取b，不为nan则取a

    print(b[:-2].combine_first(a[2:]))  # 用a的前2个元素去填补b的后2个数据；类似于where，只有b为nan，a不为nan的数据才会真正填补


def pandas_union_data():
    """
    合并数据集
        merge(): 可以根据一个或多个键将df中的行连接起来，类似于sql的join
        concat(): 可以沿着一条轴将多个对象堆叠到一起
        combine_first：可以将重复数据编接在一起，用一个对象中的值来填充另一个对象中的缺失值
    :return:
    """
    # pandas_union_data_part1()
    # pandas_union_data_part2()
    # pandas_union_data_part3()
    pandas_union_data_part4()


def pandas_reshape_pivot_part1():
    # 重塑层次化索引
    # stack: 将数据的列"旋转"为行； unstack: 将数据的行"旋转"为列
    data = DataFrame(np.arange(6).reshape((2, 3)),
                     index=pd.Index(['Ohio', 'Colorado'], name='state'),
                     columns=pd.Index(['one', 'two', 'three'], name='number'))

    # stack()是将原来的列索引转成了最内层的行索引，把df转换为series
    result = data.stack()
    # print(result)

    # unstack()最内层的行索引还原成了列索引,把series转换为df
    # print(result.unstack())

    # 默认stack(),unstack()操作的是最内层,也可以对指定分层级别进行操作
    # print(result.unstack(0))  # 最外层编号0，依次增加
    # print(result.unstack('state'))

    s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
    s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
    # data2 = pd.concat([s1, s2], keys=['one', 'two'])
    # print(data2.unstack())  # 对于级别值在分组中找不到数据，则引入Nan

    df = DataFrame({'left': result, 'right': result + 5},
                   columns=pd.Index(['left', 'right'], names='side'))
    print(df)


def pandas_reshape_pivot_part2():
    # 将”长格式“旋转为”宽格式”
    ldata = pd.read_csv('data.csv')
    # param1：行索引 param2：列索引 param3：value
    pivoted = ldata.pivot('date', 'item', 'value')
    print(pivoted)

    # pivot用来对表进行重塑，index是重塑的新表的索引名称是什么，第二个columns是重塑的新表的列名称是什么，
    # 一般来说就是被统计列的分组，第三个values就是生成新列的值应该是多少


def pandas_reshape_pivot():
    """
    重塑reshape, 轴向旋转pivot:用于重新排列表格型数据
    :return:
    """
    pandas_reshape_pivot_part1()
    pandas_reshape_pivot_part2()


def pandas_data_astype_part1():
    # 移除重复数据
    data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                      'k2': [1, 1, 2, 3, 3, 4, 5]})
    print(data)  # 部分重复
    print(data.duplicated())  # duplicated()表示各行是否是重复行，返回布尔型series
    print(data[~data.duplicated()])  # 过滤重复行
    print(data.drop_duplicates())
    print(data.drop_duplicates(['k1']))  # 根据k1列过滤重复项

    # 查看drop_duplicates的参数


def pandas_data_astype_part2():
    # 利用函数或映射进行数据转换
    data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'pastrami', 'corned beef',
                               'bacon', 'pastrami', 'honey ham', 'nova lox'],
                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
    # want: 添加一列表示该肉类食物来源的动物类型
    # step1: 编写肉类到动物的映射
    meat_to_animal = {
        'bacon': 'pig',
        'pulled pork': 'pig',
        'pastrami': 'pig',
        'corned beef': 'cow',
        'honey ham': 'pig',
        'nova lox': 'salmon'
    }

    # series的map方法可以接收一个函数或者映射关系的字典型对象
    data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
    print(data)

    print(data['food'].map(lambda x: meat_to_animal[x.lower()]))
    # 利用map是一种实现元素级转换以及其他数据清洗工作的便捷方式


def pandas_data_astype_part3():
    # 替换值
    data = Series([1, -999, 2, -999, -1000, 3])
    print(data.replace(to_replace=-999, value=np.nan))  # 把-999替换为np.nan
    print(data.replace(to_replace=[-999, -1000], value=np.nan))  # 使用列表来实现一次替换多个值
    print(data.replace(to_replace=[-999, -1000], value=[np.nan, 0]))  # 对不同的值进行不同的替换
    print(data.replace(to_replace={-999: np.nan, -1000: 0}))  # 对不同的值进行不同的替换


def pandas_data_astype_part4():
    # 重命名轴索引
    data = DataFrame(np.arange(12).reshape((3, 4)),
                     index=['Ohio', 'Colorado', 'New York'],
                     columns=['one', 'two', 'three', 'four'])
    # 利用map函数实现索引重命名
    data.index = data.index.map(str.upper)
    # print(data)

    # rename()
    print(data.rename(index=str.title, columns=str.upper))
    print(data.rename(index={'Ohio': 'bj'}, columns={'three': 'Zero'}))  # 利用字典实现对部分轴标签更新


def pandas_data_astype_part5():
    # 离散化和面元划分:cut(), qcut()
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins = [18, 25, 35, 60, 100]
    cats = pd.cut(x=ages, bins=bins)  # x: 待离散化的序列  bins: 离散化规则
    # print(cats)
    # print(cats.labels)
    # print(pd.value_counts(cats))

    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    # print(pd.cut(x=ages, bins=bins, labels=group_names))  # labels：设置面元名称

    # 如果传入的是面元的数量，那么会根据数据的最小值和最大值计算等长面元;
    # gap = (max - min) / bins, 划分后面元: (min, min+gap], (min+gap, min+gap+gap], ..., (max-gap, max]
    data = np.random.randn(1000)
    cats = pd.cut(x=data, bins=4, precision=2)
    print(pd.value_counts(cats))

    # qcut(): 可以根据样本分位数对样本进行切分
    # cut存在可能无法使各个面元中具有相同数量的数据点；qcut使用样本分位数，因此得到大小相等的面元
    data = np.random.randn(1000)
    cats = pd.qcut(x=data, q=4)  # 4分位q数切分
    # print(pd.value_counts(cats))
    print(pd.qcut(x=data, q=[0, 0.1, 0.5, 0.9, 1.0]))  # 同cut，设置自定义的分位数


def pandas_data_astype_part6():
    # 检测和过滤异常值
    np.random.seed(12345)
    data = DataFrame(np.random.randn(1000, 4))
    # print(data.describe())

    col = data[3]
    # print(col[np.abs(col) > 3])  # 第3列中绝对值大于3的数
    print(data[(np.abs(data) > 3).any(1)])  # 全部含有"绝对值大于3"的行


def pandas_data_astype_part7():
    # 排列和随机采样
    df = DataFrame(np.arange(5 * 4).reshape((5, 4)))

    sampler = np.random.permutation(5)  # 输入整数，则对np.arange(5)进行随机重排序
    # print(sampler)
    # print(np.random.permutation([1,2,3,4,5]))  # 输入一维数组，则对该数组进行随机排序

    # print(df.take(sampler))  # 对第一维索引重排序（行索引）

    # 随机采样
    # 方法1：sample和take
    bag = np.array([1, 2, 3, 4, 5])
    sampler = np.random.randint(0, len(bag), size=10)
    # print(sampler)
    # draws = bag.take(sampler)
    # print(draws)

    # 方法2：sample
    df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
    print(df.sample(n=3))  # n指定采样的个数
    print(df.sample(frac=0.1, replace=True))  # frac指定采样占原始数据的比例，replace=True表示有放回采样


def pandas_data_astype_part8():
    # 计算指标/哑变量：将分类变量转换为"哑变量矩阵"或"指标矩阵"
    # 具体含义：若df的某一列具有k个不同的值，则可以派生出一个k列矩阵或df，其值全为0或1
    df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
    print(pd.get_dummies(data=df['key']))
    dummies = pd.get_dummies(data=df['key'], prefix='key')  # prefix添加前缀
    df_with_dummy = df[['data1']].join(dummies)
    print(df_with_dummy)


def pandas_data_astype():
    # 数据转换
    # pandas_data_astype_part1()
    # pandas_data_astype_part2()
    # pandas_data_astype_part3()
    # pandas_data_astype_part4()
    # pandas_data_astype_part5()
    # pandas_data_astype_part6()
    # pandas_data_astype_part7()
    pandas_data_astype_part8()


def pandas_data_str_part1():
    # 字符串对象方法
    val = 'a,b,  guido'
    print(val.split(sep=','))  # split()切分字符串
    pieces = [x.strip() for x in val.split(',')]  # strip()去除空格
    first, second, third = pieces
    print(first + ':' + second + ':' + third)  # 利用‘+’拼接字符串
    print(':'.join(pieces))
    print('guido' in val)
    print(val.index(','))  # index找不到字符串会引发异常
    print(val.find(':'))   # find找不到字符串会返回-1
    print(val.count(','))
    print(val.replace(',', '::'))
    print(val.replace(',', ''))


def pandas_data_str_part2():
    # 正则表达式regex
    # re 模块的函数分为: 模式匹配、替换、拆分
    import re
    text = "foo   bar\t baz   \tqux"
    # print(re.split('\s+', text))  # \s：匹配任何非空白字符，包括空格、制表符、换页符等
    regex = re.compile('\s+')
    # print(regex.split(text))
    # print(regex.findall(text))  # 查找所有匹配项

    # findall(): 返回字符串中所有匹配项    search(): 返回第一个匹配项   match(): 只匹配字符串的首部
    text = """Dave dave@google.com
    Steve steve@goole.com
    Rob rob@google.com
    Ryan ryan@google.com"""


def pandas_data_str_part3():
    # Pandas中矢量化的字符串函数
    data = {'Dave': 'dave@google.com', 'Steve': 'steve@gamil.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
    data = Series(data)
    # print(data)
    # print(data.isnull())
    # print(data.str.contains('gmail'))



def pandas_data_str():
    # 字符串操作
    # pandas_data_str_part1()
    # pandas_data_str_part2()
    pandas_data_str_part3()


def usda_project():
    # usda食品数据库案例
    import json

    f = open('foods-2011-10-03.json')
    db = json.loads(f.read())
    # print(len(db))
    # print(db[0].keys())
    # print(db[0]['nutrients'])

    nutrients = DataFrame(db[0]['nutrients'])
    # print(nutrients[:7])

    info_keys = ['description', 'group', 'id', 'manufacturer']
    info = DataFrame(db, columns=info_keys)
    # print(info[:5])
    print(info.describe())
    print(pd.value_counts(info.group)[:10])  # 查看事物类别分布情况


if __name__ == "__main__":
    # pandas_union_data()
    # pandas_reshape_pivot()
    # pandas_data_astype()
    # pandas_data_str()
    usda_project()