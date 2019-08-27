#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/8/26 16:56
@Author  : Hou hailun
@File    : chapter8_src.py
"""

print(__doc__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
"""
绘图和可视化
"""
def matplotlib_part1():
    # Figure,subplot
    fig = plt.figure()  # 创建新的figure对象
    ax1 = fig.add_subplot(2, 2, 1)  # 不能通过空figure绘图，需要add_subplot创建一个或多个subplot
    ax2 = fig.add_subplot(2, 2, 2)  # 图像是2X2，当前选中第2个
    ax3 = fig.add_subplot(2, 2, 3)

    from numpy.random import randn
    _ = ax1.hist(randn(20), bins=20, color='k', alpha=0.3)
    ax2.scatter(np.arange(30), np.arange(30)+3*randn(30))
    plt.plot(randn(50).cumsum(), 'k--')  # k--： 黑色虚线
    # plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=3)
    # print(axes)

    # 调整subplot周围的间距：subplots_adjust(left, bottom, right, top, wspace, hspace)
    # wsapce/hspace用于控制宽度/高度的百分比，可以用作subplot之间的间距
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show(10)


def matplotlib_part2():
    # 颜色、标记和线型
    from numpy.random import randn

    # plt.plot(randn(30).cumsum(), 'ko--')  # 和下面的一样效果
    plt.plot(randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
    plt.show()


def matplotlib_part3():
    # 刻度、标签、图例
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # # ax.plot(np.random.randn(1000).cumsum())
    # ax.plot(np.random.randn(1000).cumsum(), color='green', linestyle='dashed')
    #
    # # 设置标题、轴标签、刻度以及刻度标签
    # ax.set_xticks([0, 250, 500, 750, 1000])  # set_xticks: 设置X轴刻度值
    # # ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')  # 设置任何值作为标签
    # ax.set_title('My first plot')  # 设置标题
    # ax.set_xlabel('Stage')  # 设置X轴标签
    # plt.show()

    # 添加图例
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')  # label指定
    ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
    ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
    ax.legend(loc='best')  # 自动创建图例
    plt.show()


def matplotlib_part4():
    # 注释以及在subplot上绘图
    from datetime import datetime
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    data = pd.read_csv('spx.csv', index_col=0, parse_dates=True)
    spx = data['SPX']

    spx.plot(ax=ax, style='k-')
    # plt.show(10)

    crisis_data = [(datetime(2007, 10, 11), 'Peak of bull market'),
                   (datetime(2008, 3, 12), 'Bear Stearns Fails'),
                   (datetime(2008, 9, 15), 'Lehman Bankruptcy')]

    for date, label in crisis_data:
        # annotate：用于在图形上给数据添加文本注解，而且支持带箭头的划线工具，方便我们在合适的位置添加描述信息
        # 参数说明:
        #   s: 注释文本的内容
        #   xy: 被注释的坐标点，二维元组(x,y)
        # xytext: 注释文本的坐标点，也是二维元组，默认与xy相同
        # arrowprops：箭头的样式，dict（字典）型数据，如果该属性非空，则会在注释文本和被注释点之间画一个箭头
        ax.annotate(s=label,
                    xy=(date, spx.asof(date) + 50),  # 假如我有一组数据，某个点的时候这个值是NaN，那就求这个值之前最近一个不是NaN的值是多少
                    xytext=(date, spx.asof(date) + 200),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='left',
                    verticalalignment='top')

    ax.set_xlim(['1/1/2007', '1/1/2011'])  # 设置X轴数据范数
    ax.set_ylim([600, 1800])
    ax.set_title("Import dates in 2008-2009 financial crisis")
    # plt.show()

    # 图形绘制: 图像称之为块patch
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rect = plt.Rectangle(xy=(0.2, 0.75), width=0.4, height=0.15)
    circ = plt.Circle(xy=(0.7, 0.2), radius=0.15)
    ax.add_patch(rect)
    ax.add_patch(circ)
    plt.show()


def matplotlib_part5():
    # 将图标保存至文件 plt.savefig('figpath.svg')
    # savefig参数说明
    # fname: 含有文件路径的字符串或python的文件型对象
    # dpi: 图像分辨率，默认为100
    # facecolor,edgecolor: 图像的背景色，默认为'w'
    # format: 显示设置文件格式
    # bbox_inches: 图标需要保存的部分。tight则剪除图标周围的空白部分
    pass


def matplotlib_code():
    # matplotlib练习
    # matplotlib_part1()
    # matplotlib_part2()
    # matplotlib_part3()
    # matplotlib_part4()
    matplotlib_part5()


def pandas_draw_line():
    """
    线型图
    series, df都有一个用于生成各类图标的plot方法，默认生成线型图
    :return:
    """
    # series的索引传给matplotlib，用来绘制X轴，use_index=False可以禁用该功能
    s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
    # s.plot()
    # plt.show()

    # df的plot会在subplot中为各列绘制一条线，并自动创建图例
    df = DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
    df.plot()
    plt.show()


def pandas_draw_bar():
    # 柱状图 kind='bar'(垂直柱状图) 或者 kind='barh'(水平柱状图)
    # fig, axes = plt.subplots(2, 1)
    # data = Series(np.random.randn(5), index=list('abcde'))
    # data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)   # 索引当作X轴
    # data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)  # 索引当作Y轴
    # plt.show()

    # df柱状图会把每一行的值分为一组；把index当作X轴；
    df = DataFrame(np.random.rand(6, 4),
                   index=['one', 'two', 'three', 'four', 'five', 'six'],
                   columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
    df.plot(kind='bar')
    plt.show()


def pandas_draw_hist_kde():
    # 直方图：可以对值频率进行离散化显示的柱状图
    # 数据点被拆分到离散的、间隔均匀的面元中，绘制的是个面元中数据点的数量
    data = Series(np.random.randn(1000))
    # data.hist(bins=50)
    # plt.show()

    # 密度图：通过计算“可能会产生观测数据的连续概率分布的估计”而产生
    # data.plot(kind='kde')
    # plt.show()

    # 把直方图和密度图绘制到一起
    comp1 = np.random.normal(0, 1, size=200)
    comp2 = np.random.normal(10, 2, size=200)
    values = Series(np.concatenate([comp1, comp2]))  # 上下拼接
    values.hist(bins=100, alpha=0.3, color='k', normed=True)
    values.plot(kind='kde', style='k--')
    plt.show()


def pandas_draw_scatter():
    # 散步图scatter
    macro = pd.read_csv('macrodata.csv')
    data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
    trans_data = np.log(data).diff().dropna()
    print(trans_data[:5])

    # plt.scatter()
    # plt.scatter(trans_data['m1'], trans_data['unemp'])
    # plt.show()

    # pd.scatter_matrix()散布图矩阵
    pd.scatter_matrix(trans_data, diagonal='kde', color='k')
    plt.show()


def pandas_draw():
    # pandas中的绘图函数
    # pandas_draw_line()
    # pandas_draw_bar()
    # pandas_draw_hist_kde()
    pandas_draw_scatter()


if __name__ == "__main__":
    # matplotlib_code()
    pandas_draw()