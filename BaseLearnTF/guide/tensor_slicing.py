# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: tensor_slicing
# CreateTime: 2023/6/19 18:07
# Summary: 张量切片


import numpy as np
import tensorflow as tf

# region 提取张量切片

t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])

# 两种切片都可以
print(tf.slice(t1, begin=[2], size=[4]))  # tf.Tensor([2 3 4 5], shape=(4,), dtype=int32)
print(t1[1: 4])  # tf.Tensor([1 2 3], shape=(3,), dtype=int32)

# 二维张量
t2 = tf.constant([[0, 1, 2, 3, 4],
                  [5, 6, 7, 8, 9],
                  [10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19]])
print(t2[:-1, 1:3])

# 高维
t3 = tf.constant([[[1, 3, 5, 7],
                   [9, 11, 13, 15]],
                  [[17, 19, 21, 23],
                   [25, 27, 29, 31]]
                  ])
print(tf.slice(t3,
               begin=[1, 1, 0],
               size=[1, 1, 2]))  # [[[25 27]]]

# 还可以使用 tf.strided_slice 通过在张量维度上“跨步”来提取张量切片
# 使用 tf.gather 从张量的单个轴中提取特定索引; 不要求索引均匀分布
print(tf.gather(t1, indices=[0, 3, 6]))

# 从张量的多个轴中提取切片，使用 tf.gather_nd
t4 = tf.constant([[0, 5],
                  [1, 6],
                  [2, 7],
                  [3, 8],
                  [4, 9]])
print(tf.gather_nd(t4, indices=[[2], [3], [0]]))
# tf.Tensor(
# [[2 7]
#  [3 8]
#  [0 5]], shape=(3, 2), dtype=int32)

t5 = np.reshape(np.arange(18), [2, 3, 3])
print(tf.gather_nd(t5, indices=[[0, 0, 0], [1, 2, 1]]))
# tf.Tensor([ 0 16], shape=(2,), dtype=int32)

# endregion


# region 将数据插入张量

# 使用 tf.scatter_nd 在张量的特定切片/索引处插入数据，将值插入的张量是用零初始化的
t6 = tf.constant([10])
indices = tf.constant([[1], [3], [5], [7], [9]])
data = tf.constant([2, 4, 6, 8, 10])
print(tf.scatter_nd(indices=indices, updates=data, shape=t6))

new_indices = tf.constant([[0, 2], [2, 1], [3, 3]])
t7 = tf.gather_nd(t2, indices=new_indices)  # [ 2, 11, 18]

# 使用 tf.gather_nd 和 tf.scatter_nd 来模拟稀疏张量运算的行为
t8 = tf.scatter_nd(indices=new_indices, updates=t7, shape=tf.constant([4, 5]))
# 类似于
t9 = tf.SparseTensor(indices=[[0, 2], [2, 1], [3, 3]],
                     values=[2, 11, 18],
                     dense_shape=[4, 5])

t10 = tf.sparse.to_dense(t9)
print(t10)

# 将数据插入到具有既有值的张量中，请使用 tf.tensor_scatter_nd_add
t11 = tf.constant([[2, 7, 0],
                   [9, 0, 1],
                   [0, 3, 8]])
t12 = tf.tensor_scatter_nd_add(t11,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[6, 5, 4])
# 减去值
t13 = tf.tensor_scatter_nd_sub(t11,
                               indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]],
                               updates=[1, 7, 9, -1, 1, 3, 7])

# tf.tensor_scatter_nd_min 将逐元素最小值从一个张量复制到另一个
t14 = tf.constant([[-2, -7, 0],
                   [-9, 0, 1],
                   [0, -3, -8]])
t15 = tf.tensor_scatter_nd_min(t14,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[-6, -5, -4])
# tf.tensor_scatter_nd_max 将逐元素最大值从一个张量复制到另一个
t16 = tf.tensor_scatter_nd_max(t14,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[6, 5, 4])

# endregion
