# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: variable
# CreateTime: 2023/6/9 17:49
# Summary: 变量


import tensorflow as tf

# 查看变量位于哪个设备上
# tf.debugging.set_log_device_placement(True)

# 创建变量
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # tf.Variable 与 初始值的 dtype 相同
my_variable = tf.Variable(my_tensor)

#  变量无法重构形状

tf.convert_to_tensor(my_variable)  # 转换为 张量
tf.math.argmax(my_variable)  # 最大值的索引位置

#
a = tf.Variable([2., 3.])
a.assign([1, 2])  # 重新分配张量，不会分配新张量，而是 重用 现有 张量的内存

# a.assign([3., 4., 5.])  # 抛异常

a = tf.Variable([2., 3.])
b = tf.Variable(a)
a.assign([1, 2])
print(a.numpy())
print(b.numpy())  # a, b 不共享同一内存

a = tf.Variable(my_tensor, name='SAM')
b = tf.Variable(my_tensor + 1, name='SAM')
print(a == b)

# trainable 可关闭梯度，训练计步器是一个不需要梯度的变量； 不需要微分就关闭梯度
step_counter = tf.Variable(1, trainable=False)

# 使用多个设备，具体查看分布式训练
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
    k = a * b

print(k)
