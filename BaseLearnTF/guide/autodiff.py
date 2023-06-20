# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: autodiff
# CreateTime: 2023/6/10 9:05
# Summary: 自动微分和梯度


import math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 计算梯度
# 要实现自动微分，tf 需要记住 前向传递 过程中 那些运算以何种顺序发生
# 之后，在 后向传递 期间，tf 以相反的顺序遍历此运算列表 来计算梯度


# region 梯度带
# GradientTape 提供了自动微分的api；

# x = tf.Variable(2/math.pi)

x = tf.Variable(3.)

with tf.GradientTape() as tape:  # 梯度带上下文
    y = x ** 2  # x的平方；记录运算顺序；
    # y = tf.math.cos(x)

dy_dx = tape.gradient(y, x)  # x的平方；微分后 2x；结果为6.0； y = cos(x)  结果为-1.0
print(dy_dx.numpy())

w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')  # tf.Variable(tf.zeros((3,2), dtype=tf.float32), name='b')  # 3 * 2  0矩阵
x = [[1., 2., 3.]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])

print(w.shape)  # 相对于每个源的梯度具有源的形状
print(dl_dw.shape)

# endregion


# region 相对于模型的梯度

layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[1., 2., 3.]])

with tf.GradientTape() as tape:
    y = layer(x)  # 将 tf.Variables 收集到 tf.Model中 (或子类中：layers.Layer，keras.Model) ，用于设置 检查点 和导出
    loss = tf.reduce_mean(y ** 2)

# 计算每个可训练变量的梯度
grad = tape.gradient(loss, layer.trainable_variables)

for var, g in zip(layer.trainable_variables, grad):
    print(f'{var.name}, shape: {g.shape}')

# endregion


# region 控制梯度带监视的内容

#  条带需要 前向传递 中记录哪些运算，以计算 后向传递 梯度
#  梯度带包含对中间输出的引用，避免记录不必要的操作
#  常用于 计算损失相对于模型的所有可变训练变量的梯度


x0 = tf.Variable(3.0, name='x0')  # 可训练变量
# 不可训练变量
x1 = tf.Variable(3.0, name='x1', trainable=False)
# 不可训练变量: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name='x2') + 1.
# 不可训练变量
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape:
    y = (x0 ** 2) + (x1 ** 2) + (x2 ** 2)  # 梯度带包含对中间输出的引用，避免记录不必要的操作

grad = tape.gradient(y, [x0, x1, x2, x3])

for g in grad:
    print(g)

[var.name for var in tape.watched_variables()]  # ['x0:0']；列出 梯度带 正在监视的变量

# 控制监视内容
x = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(x)  # 控制监控内容
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())

# 停用控制监视所有默认行为
x0 = tf.Variable(0., name='x0')
x1 = tf.Variable(10., name='x1')

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)  # 只监视一个可训练变量
    y0 = tf.math.sin(x0)
    y1 = tf.nn.softplus(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)

# [var.name for var in tape.watched_variables()]  # ['x1:0']

grad = tape.gradient(ys, {'x0': x0, 'x1': x1, })  # x0 为 None，因为未监视

# endregion


# region 中间结果

# GradientTape 上下文计算的 中间值 的 梯度

x = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
    z = y * y

# dz_dy = 2 * y   (对y求导)
# y = x ** 2 = 9  (不是对x求导)
dz_dy = tape.gradient(z, y).numpy()  # 中间变量 y 的梯度， dz_dy = 2 * y

# 计算多个梯度：persistent=True
x = tf.constant([1, 3.])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x * x
    z = y * y  # z = x**2 * x**2 = x**4

tape.gradient(z, y).numpy()  # dz_dy = 2y = 2x**2
tape.gradient(y, x).numpy()  # dy_dx = 2x
tape.gradient(z, x).numpy()  # 梯度  dz_dx = 4x**3

# endregion


# region 非标量目标的梯度

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y0 = x ** 2
    y1 = 1 / x

tape.gradient({'y0': y0, 'y1': y1}, x).numpy()  # 梯度总和， dy0_dx + dy1_dx = 4 + (-1/4)

x = tf.Variable(2.)
with tf.GradientTape() as tape:
    y = x * [3., 4.]

print(tape.gradient(y, x).numpy())  # dy_dx = [3., 4.] * 1 = 7

# 总和的梯度给出 每个元素相对于其输入元素的导数
x = tf.linspace(-10.0, 10.0, 200 + 1)  # 201个 元素
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.nn.sigmoid(x)

dy_dx = tape.gradient(y, x)  # 输出每个元素的导数

plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
_ = plt.xlabel('x')
plt.show()

# endregion


# region 控制流

x = tf.constant(1.)

v0 = tf.Variable(2.)
v1 = tf.Variable(2.)

# 梯度带会处理控制流 if
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    if x > 0.:
        result = v0
    else:
        result = v1 ** 2

dv0, dv1 = tape.gradient(result, [v0, v1])
# 控制语句本身不可微分，因此对 基于梯度的优化器 不可见
dx = tape.gradient(result, x)  # None

# endregion


# region gradient 返回 None 的情况

#  1. 当目标未连接到源，gradient返回None
x = tf.Variable(2.)
y = tf.Variable(3.)
with tf.GradientTape() as tape:
    z = y * y
tape.gradient(z, x)  # 返回None

#  2. 使用张量替换变量
x = tf.Variable(2.)

for epoch in range(2):
    with tf.GradientTape() as tape:
        y = x + 1
    print(tape.gradient(y, x))
    x = x + 1  # 变成了tensor 张量；不可训练变量，梯度带也不会监视

#  3. 在tf之外进行了计算
x = tf.Variable([[1., 2.],
                 [3., 4.]], dtype=tf.float32)

with tf.GradientTape() as tape:
    x2 = x ** 2
    y = np.mean(x2, axis=0)

    y = tf.reduce_mean(y, axis=0)  # reduce_mean 将 y 转变为了张量
    # y = tf.Variable(y) # 需要将 tensor 转换为 variable

print(tape.gradient(y, x))

#  4. 通过整数或字符串获取梯度
x = tf.constant(10)  # 整数和字符串不可微分
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
print(tape.gradient(y, x))

#  5. 通过有状态对象获取梯度
# 状态会停止梯度 （梯度带只能观察当前状态，不能观察导致该状态的历史记录）
# tensor 不可变；张量创建后不能更改，有一个值，但无状态
# variable 具有内部状态，也有值，使用变量会读取状态；变量的状态会组织梯度计算进一步向后移动，例：
x0 = tf.Variable(3.)
x1 = tf.Variable(0.)
with tf.GradientTape() as tape:
    x1.assign_add(x0)  # 更新 x1 = x1 + x0
    y = x1 ** 2  # y = (x1 + x2) ** 2
print(tape.gradient(y, x0))

# tf.data.Dataset 迭代器 和 tf.queue 也有状态，会停止经过它们的张量上的所有梯度


# endregion


# region 未注册梯度

# endregion

# region 零而不是None
#
x = tf.Variable([2., 2.])
y = tf.Variable(3.)
with tf.GradientTape() as tape:
    z = y ** 2
# print(tape.gradient(z, x))  # None
print(tape.gradient(z, x, unconnected_gradients=tf.UnconnectedGradients.ZERO))  # 未连接的梯度，设置返回0
# endregion
