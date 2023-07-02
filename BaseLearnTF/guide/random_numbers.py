# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: random_numbers
# CreateTime: 2023/7/1 14:28
# Summary: 生成随机数


import tensorflow as tf

# 创建一些虚拟设备(cpu:0, cpu:1等)用于使用分发策略
physical_devices = tf.config.list_physical_devices('CPU')
tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration(),
    ]
)

# region tf.random.Generator 类


g1 = tf.random.Generator.from_seed(1)  # 生成器
print(g1.normal(shape=[2, 3]))
g2 = tf.random.get_global_generator()  # 获得默认全局生成器
print(g2.normal(shape=[2, 3]))

g1 = tf.random.Generator.from_seed(1, alg='philox')
print(g1.normal(shape=[2, 3]))

# 以这种方式创建的生成器首先会处于非确定状态，具体取决于时间和操作系统等因素
g = tf.random.Generator.from_non_deterministic_state()
print(g.normal(shape=[2, 3]))

g = tf.random.Generator.from_seed(1)
print(g.normal([]))
print(g.normal([]))
g.reset_from_seed(1)
print(g.normal([]))

# 创建独立的随机数流
g = tf.random.Generator.from_seed(1)
print(g.normal([]))
new_gs = g.split(3)  # 创建多个独立的随机数流（不能相互重叠，不能有统计学上可检测到的相关性）
for new_g in new_gs:
    print(new_g.normal([]))
print(g.normal([]))

# region 与 tf.function 交互

# 在 tf.function 的外部创建生成器
g = tf.random.Generator.from_seed(1)


@tf.function
def foo_1():
    return g.normal([])


print(foo_1())

# 在 tf.function 的内部创建生成器
g = None


@tf.function
def foo_2():
    global g
    if g is None:
        g = tf.random.Generator.from_seed(1)
    return g.normal([])


print(foo_2())
print(foo_2())

# 将生成器作为参数传递给 tf.function
num_traces = 0


# 当用作 tf.function 的参数时，不同的生成器对象将导致 tf.function 的回溯
@tf.function
def foo_3(g):
    global num_traces
    num_traces += 1
    return g.normal([])


foo_3(tf.random.Generator.from_seed(1))
foo_3(tf.random.Generator.from_seed(2))
print(num_traces)

# 此回溯行为与 tf.Variable 一致s
num_traces = 0


@tf.function
def foo(v):
    global num_traces
    num_traces += 1
    return v.read_value()


foo(tf.Variable(1))
foo(tf.Variable(2))
print(num_traces)

# endregion


# region 与分布策略交互

# 在分布策略的外部创建生成器

#   在策略作用域的外部创建的生成器，则会序列化访问此生成器的所有副本
#   每一个副本都会得到不同的随机数
# 这种使用方法可能产生性能问题
g = tf.random.Generator.from_seed(1)
start = tf.distribute.MirroredStrategy(devices=['cpu:0', 'cpu:1'])
with start.scope():
    def f():
        print(g.normal([]))


    result = start.run(f)

# 在分布策略的内部创建生成器

#   在策略作用域内创建生成器，则每个副本将获得不同且独立的随机数流
start = tf.distribute.MirroredStrategy(devices=['cpu:0', 'cpu:1'])
with start.scope():
    g = tf.random.Generator.from_seed(1)
    print(start.run(lambda: g.normal([])))
    print(start.run(lambda: g.normal([])))

# 在 Strategy.run 内创建 tf.random.Generator
start = tf.distribute.MirroredStrategy(devices=['cpu:0', 'cpu:1'])
with start.scope():
    def f():
        g = tf.random.Generator.from_seed(1)
        a = g.normal([])
        b = g.normal([])
        return tf.stack([a, b])


    print(start.run(f))
    print(start.run(f))

# endregion


# region 保存生成器

# TF 中有两种序列化机制：检查点和 SavedModel

# 检查点
#   可以使用 tf.train.Checkpoint 自由保存和恢复生成器
filename = "./random_numbers/checkpoint"
g = tf.random.Generator.from_seed(1)
cp = tf.train.Checkpoint(generator=g)
print(g.normal([]))

cp.write(filename)  # 写入
print("RNG stream from saving point:")
print(g.normal([]))
print(g.normal([]))

cp.restore(filename)  # 恢复
print("RNG stream from restoring point:")
print(g.normal([]))
print(g.normal([]))

# 在分发策略中保存和恢复
filename = "./random_numbers/checkpoint"
strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
with strat.scope():
    g = tf.random.Generator.from_seed(1)
    cp = tf.train.Checkpoint(my_generator=g)
    print(strat.run(lambda: g.normal([])))

with strat.scope():
    cp.write(filename)
    print("RNG stream from saving point:")
    print(strat.run(lambda: g.normal([])))
    print(strat.run(lambda: g.normal([])))

with strat.scope():
    cp.restore(filename)
    print("RNG stream from restoring point:")
    print(strat.run(lambda: g.normal([])))
    print(strat.run(lambda: g.normal([])))

# endregion


# endregion
