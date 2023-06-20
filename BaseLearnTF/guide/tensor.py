# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: tensor
# CreateTime: 2023/6/9 11:21
# Summary: 张量


import numpy as np
import tensorflow as tf

# 張量：多维数组 dtype 不可变的，只能创建一个新的张量

rank_0_tensor = tf.constant(4)  # 标量（或称 0秩 张量）；标量包含单个值，没有“轴”
print(rank_0_tensor)

rank_1_tensor = tf.constant([2.0, 3.0, 4.0])  # 向量 （或称 1秩 张量），有 1 个 轴
print(rank_1_tensor)

rank_2_tensor = tf.constant([[2, 3],
                             [4, 5],
                             [6, 7],
                             ], dtype=tf.float64)  # 矩阵 （或称 2秩 张量），有 2 个 轴
print(rank_2_tensor)

# np.array tensor.numpy 相互转换
np.array(rank_2_tensor)
rank_2_tensor.numpy()

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[2, 2],
                 [2, 2]])
tf.add(a, b)  # 加法  == a + b
tf.multiply(a, b)  # 逐个元素乘法  == a * b
tf.matmul(a, b)  # 矩阵乘法  == a @ b

tf.reduce_max(a)  # a 中最大元素
tf.math.argmax(a)  # a 中最大元素位置的索引

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
tf.nn.softmax(c)  # 计算softmax

rank_4_tensor = tf.zeros([3, 2, 4, 5])  # 4秩张量 形状[3, 2, 4, 5]
print('Type of every element: ', rank_4_tensor.dtype)  # 每个元素的类型
print('Number of axes: ', rank_4_tensor.ndim)  # 4  轴  不返回 Tensor 对象
print('Shape of tensor: ', rank_4_tensor.shape)  # [3, 2, 4, 5]
print('Elements along axis 0 of tensor: ', rank_4_tensor.shape[0])  # 3
print('Elements along the last axis of tensor: ', rank_4_tensor.shape[-1])  # 5
print('Total number of elements (3*2*4*5): ', tf.size(rank_4_tensor).numpy())  # 形状的大小（张量的总项数，即形状矢量元素的乘积）

# 索引
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())  # [ 0  1  1  2  3  5  8 13 21 34]
rank_1_tensor[4].numpy()  # 3

# 多轴索引
rank_2_tensor[2, 1].numpy()

rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])

rank_3_tensor[:, :, 4].numpy()  # 输出最后一列

# 操作形状
x = tf.constant([[1], [2], [3]])
reshaped = tf.reshape(x, [1, 3])
print(x.shape)
print(reshaped.shape)  # [1, 3]

tf.reshape(rank_3_tensor, [-1])  # 展平成 1轴

tf.reshape(rank_3_tensor, [3 * 2, 5])  # 不容易混淆
tf.reshape(rank_3_tensor, [3, -1])
tf.reshape(rank_3_tensor, [3, 2 * 5])  # 不容易混淆

# DTypes
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)  # 指定 float64 ，默认32
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)  # 64 转换为 16
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)  # 转换为无符整型

# 广播，对小张量进行扩展；不会在内存中具体化扩展的张量
x = tf.constant([1, 2, 3])  # 1行
y = tf.constant(2)
z = tf.constant([2, 2, 2])

print(tf.multiply(x, 2))  # 扩展
print(x * y)
print(x * z)

x = tf.reshape(x, [3, 1])  # 3 * 1 的矩阵
y = tf.range(1, 5)  # 4个元素
print(x * y)  # 3*4 的矩阵；广播

# 不使用广播的同一运算
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])
print(x_stretch * y_stretch)

# broadcast_to 不会节省内存。具体化张量
tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3])

# 不规则张量
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9],
]
try:
    tensor = tf.constant(ragged_list)  # 不规则的张量 会报错
except Exception as e:
    print(f"{type(e).__name__}: {e}")

ragged_tensor = tf.ragged.constant(ragged_list)

# 字符串张量
scalar_string_tensor = tf.constant('sam hello')
tensor_of_strings = tf.constant(['sam hello',
                                 'world',
                                 'hi!'])
tf.constant("🥳👍")

tf.strings.split(scalar_string_tensor, sep=" ")  # 字符串的操作
tf.strings.split(tensor_of_strings)  # 变为不规则的张量

text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))  # 字符串 分割后 转为 数字型

# 字符串 转换 为 数值
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)  # 68 117 99 107  ==> D u c k

# 稀疏张量
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[5, 7],
                                       dense_shape=[3, 4])
# [0,0]位置的值为5，[1, 2]位置的值为7，的 3 * 4 的矩阵; 其他位置自动填充0

print(sparse_tensor, "\n")
print(tf.sparse.to_dense(sparse_tensor))
