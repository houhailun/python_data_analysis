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
                    xy=(date, spx.asof(date) + 50),
                    xytext=(date, spx.asof(date) + 200),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='left',
                    verticalalignment='top')

    ax.set_xlim(['1/1/2007', '1/1/2011'])  # 设置X轴数据范数
    ax.set_ylim([600, 1800])
    ax.set_title("Import dates in 2008-2009 financial crisis")
    plt.show()


if __name__ == "__main__":
    # matplotlib_part1()
    # matplotlib_part2()
    # matplotlib_part3()
    matplotlib_part4()