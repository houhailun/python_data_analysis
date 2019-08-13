#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/8/12 14:00
@Author  : Hou hailun
@File    : numpy.py
"""
print(__doc__)

import numpy as np

"""
numpy练习：numpy是python的科学计算和数据处理的包，主要功能：
    ndarray：多维数据，可进行矢量运算
    快速操作处理数组/矩阵的函数
    线性代数、随机数、傅里叶变换等功能
    集成其他语言的代码
"""


def numpy_learn_create_ndarray():
    # PART1: 创建ndarray的方式
    # 1、array(序列型对象)
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])

    # 2、全0，全1,empty数组，单位矩阵(对角线为1，其余为0)
    print(np.zeros(10))
    print(np.zeros((2, 3)))
    print(np.ones(10))
    print(np.ones((2, 3)))
    print(np.empty(10))  # empty返回的是未初始化的垃圾值
    print(np.empty((2, 3)))
    print(np.eye(10))

    # 3、arange(): range()的数组版本
    print(np.arange(10))


def numpy_learn_ndarray_type():
    arr1 = np.array([1, 2, 3], dtype=np.float64)
    arr2 = np.array([1, 2, 3], dtype=np.int32)

    print(arr1.dtype)
    print(arr2.dtype)

    # 使用astype()转换ndarray的dtype
    # astype()会创建一个新的数组，两者互不影响
    arr = np.array([1, 2, 3])
    print(arr.dtype)
    float_arr = arr.astype(np.float64)
    print(float_arr.dtype)


def numpy_learn_ndarray_math():
    arr = np.array([1, 2, 3])
    print(arr * arr)
    print(arr - arr)
    print(arr ** 0.5)
    print(1 / arr)


def numpy_learn_index_split():
    # 一维数组
    arr = np.arange(10)
    # print(arr[5])
    # print(arr[5: 8])

    # 多维数组
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(arr2d[2])  # [7, 8, 9]
    # print(arr2d[0][2])  # 等价的
    # print(arr2d[0, 2])

    arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # print(arr3d)
    # print(arr3d[0])
    # print(arr3d[1][0])

    # 切片索引
    # print(arr[1:6])
    # print(arr2d[:2])  # 前2行
    # print(arr2d[:2, :1])  # 前2行 && 前1列

    # 布尔型索引
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    data = np.random.randn(7, 4)  # 正态分布的随机数
    print(names == 'Bob')  # 布尔型数组 [True,False,False,True,False,False,False]
    print(data[names == 'Bob'])  # 布尔型数组用于数组索引,只取True对应的数据
    print(data[names == 'Bob', 2:])
    print(names != 'Bob')
    print(data[names != 'Bob'])
    print(data[~(names == 'Bob')])

    mask = (names == 'Bob') | (names == 'Will')  # 组合布尔条件
    print(mask)
    print(data[mask])

    data[data < 0] = 0
    print(data)

    data[names != 'Joe'] = 7  # 设置某行/列值
    print(data)

    # 花式索引：利用整数数组进行索引
    arr = np.empty((8, 4))
    for i in range(8):
        arr[i] = i
    # print(arr)

    # print(arr[[4, 3, 0, 6]])  # 根据指定顺序的列表来选取行子集

    arr = np.arange(32).reshape((8, 4))
    print(arr[[1, 5, 7, 2]])
    print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])  # 先选取行1，5，7，2；在根据列0，3，1，2调整顺序
    print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])


def numpy_learn_transpose():
    arr = np.arange(15).reshape((3, 5))
    # print(arr)
    # print(arr.T)

    arr = np.random.randn(6, 3)
    print(np.dot(arr.T, arr))  # dot()计算内积


def numpy_learn():
    # Numpy的ndarray: N-Dim-Array, 多维数组
    # a = np.array([1, 2, 3])
    #
    # # 数组属性
    # print(type(a))
    # print(a.dtype)  # 元素类型
    # print(a.size)   # 元素个数
    # print(a.shape)  # 元素维数
    # print(a.itemsize)  # 单个元素所占内存字节数
    # print(a.ndim)      # 维度
    # print(a.nbytes)    # 所占内存字节数 = a.size * a.itemsize

    # PART1: 创建ndarray
    # numpy_learn_create_ndarray()

    # PART2: ndarray的数据类型
    # numpy_learn_ndarray_type()

    # PART3: 数组和标量之间的运算
    # 矢量化处理：
    #   大小相等的数组的算数运算会把运算应用到元素级
    #   数组于标量的算数运算也会把标量值传播到各个元素
    # numpy_learn_ndarray_math()

    # PART4: 索引和切片
    # numpy_learn_index_split()

    # PART5: 数组转置和轴对换
    # numpy_learn_transpose()
    pass


def numpy_ufunc():
    # 通用函数：快速的元素级数组函数
    arr = np.arange(10)
    # print(np.sqrt(arr))  # 一元函数
    # print(np.exp(arr))

    x = np.random.randn(8)
    y = np.random.randn(8)
    # print(np.maximum(x, y))  # 二元函数，元素级最大值


def numpy_data_process():
    # 利用数组进行数据处理
    # 使用数组表达式代替循环 -- 矢量化
    points = np.arange(-5, 5, 0.01)
    xs, ys = np.meshgrid(points, points)
    # print(ys.shape)
    # import matplotlib.pyplot as plt
    # z = np.sqrt(xs**2 + ys**2)
    # plt.imshow(z, cmap=plt.cm.gray)
    # plt.colorbar()
    # plt.show()

    # PART2：将条件逻辑表述为数组运算
    xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    cond = np.array([True, False, True, True, False])
    result = [(x if c else y) for x,y,c in zip(xarr, yarr, cond)]
    # print(result)

    # 纯python速度慢且无法用于多维数组
    result = np.where(cond, xarr, yarr)  # 第一个参数为条件，真则选第2个参数；假则选第3个参数
    # print(result)

    arr = np.random.randn(4, 4)
    arr = np.where(arr > 0, 2, -2)
    # print(arr)

    # PART3：数学和统计方法
    arr = np.random.randn(5, 4)
    # print(arr)
    # print(arr.mean())   # 数组的实例方法
    # print(np.mean(arr))  # np的顶级方法
    # print(arr.sum())
    # print(arr.mean(axis=1))  # 按行求和，即把一行中的所有列求均值，最后数据个数和行数相等
    # print(arr.sum(0))

    # 用于布尔型数组的方法
    arr = np.random.randn(100)
    # print(np.sum(arr>0))

    bools = np.array([False, False, True, False])
    # print(bools.any())   # 测试数组中是否存在一个或多个True
    # print(bools.all())   # 测试数组中所有值是否都是True

    # 排序: 就地排序
    arr = np.random.randn(8)
    # print(arr)
    arr.sort()
    # print(arr)

    arr = np.random.randn(5, 3)
    # arr.sort()
    # print(arr)
    # arr.sort(0)
    # print(arr)
    # axis=0，沿着纵轴操作；axis=1，沿着横轴操作

    # 唯一化以及其他的集合逻辑
    names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
    print(np.unique(names))  # 唯一化并返回已排序的结果，等于sorted(set(names))

    values = np.array([6, 0, 0, 3, 2, 5, 6])
    print(np.in1d(values, [2, 3, 6]))  # 测试数组中的值是否在另一个数组中
    """
    unique(x): 计算x中的唯一值，并返回有序结果
    intersect1d(x, y): 计算x和y中的公共元素，并返回有序结果
    union1d(x, y): 计算x和y的并集，并返回有序结果
    in1d(x, y): 得到一个表示”x的元素是否包含于y“的布尔型数组
    setdiff1d(x,y): 集合的查，即元素在x中且不在y中
    setxor1d(x,y): 集合的对称差，即存在于一个数组中单不同时存在于两个数组中的元素
    """

    xarr = np.array([1, 2, 3, 4])
    yarr = np.array([3, 4, 5, 6])
    print(np.intersect1d(xarr, yarr))
    print(np.union1d(xarr, yarr))
    print(np.setdiff1d(xarr, yarr))
    print(np.setxor1d(xarr, yarr))


def numpy_input_output():
    # 用于数组的文件输入输出
    # PART1: 将数组以二进制格式保存到磁盘；np.load(), np.save()
    arr = np.arange(10)
    np.save('some_array', arr)  # 二进制保存

    arr_load = np.load('some_array.npy')  # 加载文件
    # print(arr_load)

    np.savez('array_archive.npz', a=arr, b=arr)  # 将多个数组保存到压缩文件
    arch = np.load('array_archive.npz')
    # print(arch['a'])
    # print(arch['b'])
    #
    # PART2: 存取文本文件
    arr = np.loadtxt('array_ex.txt', delimiter=',')  # 加载文本文件到二维数组
    # print(arr)
    np.savetxt('array_ex.txt', arr, delimiter=',')  # 把数组保存到文本文件


def numpy_linear_algebra():
    # 线性代数: 矩阵乘法、矩阵分解、行列式等
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[6, 23], [-1, 7], [8, 9]])

    # print(np.dot(x, y))  # 矩阵乘法,点击

    from numpy.linalg import det, svd, inv
    print(svd(x))
    """
    dot: 矩阵乘法
    trace: 对角线元素的和
    det: 计算矩阵行列式
    eig: 计算方阵的特征值和特征向量
    inv: 计算方阵的逆
    qr: QR分解
    svd: svd分解
    lstsq: 计算Ax=b的最小二乘解
    """


def numpy_random_data():
    # 随机数生成
    samples = np.random.normal(size=(4, 4))  # 标准正态分布

    """
    numpy.random常用函数
    seed: 随机数生成器种子
    permutation: 返回序列的随机排序
    shuffle: 就地随机打乱序列
    rand: 产生均匀分布的样本值
    randint: 从给定的上下限范围内随机选取整数
    randn: 正态分布
    binomial: 二项分布
    normal: 正态-高斯分布
    beta: beta分布
    chisquare: 卡方分布
    gamma: gamma分布
    uniform: [0,1)均匀分布
    """
    arr = np.array([1,2,3,4,5])
    np.random.shuffle(arr)
    print(arr)

    arr_new = np.random.permutation(arr)
    print(arr_new)

    print(np.random.uniform(size=(4,4)))
    print(np.random.rand(4, 3))


def numpy_random_walk():
    # 范例：随机漫步
    nsteps = 1000
    draws = np.random.randint(0, 2, size=nsteps)
    steps = np.where(draws > 0, 1, -1)
    walk = np.cumsum(steps)
    print(walk.min())
    print(walk.max())


if __name__ == "__main__":
    # numpy_learn()
    # numpy_ufunc()
    # numpy_data_process()
    # numpy_input_output()
    # numpy_linear_algebra()
    # numpy_random_data()
    numpy_random_walk()