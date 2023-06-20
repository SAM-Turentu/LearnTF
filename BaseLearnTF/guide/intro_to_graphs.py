# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: intro_to_graphs
# CreateTime: 2023/6/11 15:23
# Summary: 计算图和函数 tf.function


import timeit

import tensorflow as tf


# 计算图（包含一组 tf.Operation 对象和 tf.Tensor 对象的 数据结构）
# 计算图在 tf.Graph 上下文中定义
# 优点：
#  折叠常量节点来静态推断张量的值（常量折叠）
#  分离独立的计算子部分，并在线程 或 设备 之间进行拆分
#  消除通用子表达式简化算数运算


# region 利用计算图
def a_regular_function(x, y, b):
    x = tf.matmul(x, y)  # == x @ y
    x = x + b
    return x


# tf.function 可直接调用，要么作为装饰器
a_function_that_uses_a_graph = tf.function(a_regular_function)

x1 = tf.constant([[1., 2.]])
y1 = tf.constant([[2.], [3.]])
b1 = tf.constant(4.)

orig_value = a_regular_function(x1, y1, b1).numpy()

tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()  # 以python方式调用 这个 `Function`
assert (orig_value == tf_function_value)


# tf.function 适用于 一个函数极其调用的所有其他函数

def inner_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


# 外部函数添加一个 Function 的装饰器
@tf.function
def outer_function(x):
    y = tf.constant([[2.], [3.]])
    b = tf.constant(4.)

    return inner_function(x, y, b)


# 可调用对象将创建一个包含`inner_function`和`outer_function`的图
outer_function(tf.constant([[1., 2.]])).numpy()


# *********************************************************
# 将 Python 函数转换为计算图

def simple_relu(x):
    if tf.greater(x, 0):
        return x
    else:
        return 0


tf_simple_relu = tf.function(simple_relu)
# function 使用 AutoGraph 将python 代码转换为计算图生成代码
print('First branch, with graph:', tf_simple_relu(tf.constant(1)).numpy())
print('Second branch, with graph:', tf_simple_relu(tf.constant(-1)).numpy())


# *********************************************************
# 多态性：一个Function，多个计算图

@tf.function
def my_relu(x):
    return tf.maximum(0., x)  # 返回 0 x 中最大值


# 由于由多个计算图提供支持，因此 Function 是【多态的】，可以接受更多输入类型
# 这些输入 称为 【输入签名】 或 【签名】
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))
print(my_relu(tf.constant([3., -3.])))

print(my_relu(tf.constant(-2.5)))  # 已经使用了该签名调用了 Function，则 Function 不会创建新的 tf.Graph
print(my_relu(tf.constant([-1., 1])))

# 查看签名类型 和 返回值的类型和形状
print(my_relu.pretty_printed_concrete_signatures())


# endregion


# region 使用 tf.function

# Function 既能以 Eager 模式执行，也能作为计算图执行；默认作为计算图执行
@tf.function
def get_MSE(y_true, y_pred):
    sq_diff = tf.pow(y_true - y_pred, 2)  # 相减后 在平方
    return tf.reduce_mean(sq_diff)


y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print(y_true)
print(y_pred)

get_MSE(y_true, y_pred)


# # Function 以 Eager 模式执行 开关
# tf.config.run_functions_eagerly(True)
# # 开启 Eager 验证两种方式运行结构是否相同
# get_MSE(y_true, y_pred)


# Function 在计算图执行和 Eager Execution 下的行为可能有所不同
@tf.function
def get_MSE_v2(y_true, y_pred):
    print('Calculating MSE!')
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)


error = get_MSE_v2(y_true, y_pred)
error = get_MSE_v2(y_true, y_pred)
error = get_MSE_v2(y_true, y_pred)


# 只打印了一次 'Calculating MSE!'

# 关闭计算图模式，打开 Eager 模式运算 Function
# tf.config.run_functions_eagerly(True)


# 打印了3次 'Calculating MSE!'

# 原因是 print python 代码没有别计算图捕获

# ******************************************************
# 非严格执行
#  函数的返回值
#  输入输出运算 （tf.print）
#  调试运算，（tf.debugging中的断言函数）
#  tf.Variable的突变

# @tf.function
def unused_return_eager(x):
    tf.gather(x, [1])
    return x


# 计算图 执行期间 跳过了 不必要的运算 tf.gather
# 不会像 Eager 中那样报错

try:
    print(unused_return_eager(tf.constant([0.])))
except tf.errors.InvalidArgumentError as e:
    print(f'{type(e).__name__}: {e}')

# ******************************************************
# tf.function 最佳做法
#  经常使用 tf.config.run_functions_eagerly(True) 查看 Eager 和 计算图之间是否何时出现分歧
#  在python函数外部创建 Variable 并在内部修改它们。
#  避免依赖于外部 python变量的函数，除了 Variable 和 Keras 对象
#  尽可能编写以 张量和其他 tf 类型作为输入的函数
#  在tf.function 包含尽可能多的计算，提高性能。（例：装饰整个训练步骤或整个训练循环）

# endregion


# region 见证加速
# 小型计算以调用计算图为主

x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)


def power(x, y):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(y):
        result = tf.matmul(x, result)
    return result


# 很慢
print('Eager execution: ', timeit.timeit(lambda: power(x, 100), number=1000), 'seconds')

# 很快
power_as_graph = tf.function(power)
print('Eager execution: ', timeit.timeit(lambda: power_as_graph(x, 100), number=1000), 'seconds')


# tf.function(jit_compile=True) 以获得更显著的性能提升；
#  特别是代码非常依赖于 tf控制流并且有很多小张量


# endregion

# 某些函数，计算图的创建比计算图的执行话费更长时间
# 由于跟踪的原因，任何大模型训练前几步可能较慢


# region Function 何时跟踪
@tf.function
def a_function_with_python_side_effect(x):
    print('Tracing!')
    return x * x + tf.constant(2)


# Tracing! 打印一次
print(a_function_with_python_side_effect(tf.constant(2)))
print(a_function_with_python_side_effect(tf.constant(3)))

# Tracing! 打印2次; 新的参数总是会触发 计算图的创建，需要额外的跟踪
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))
# endregion
