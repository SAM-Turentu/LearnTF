# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: data_performance
# CreateTime: 2023/7/6 9:31
# Summary: 使用 tf.data API 提升性能


import itertools
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import time


# region 设置

# 数据集
class ArtificialDataset(tf.data.Dataset):

    def _generator(num_samples):
        time.sleep(0.03)  # 模拟打开文件之前休眠一段时间
        for sample_idx in range(num_samples):
            time.sleep(0.015)  # 在生成每项之前休眠一段时间，以模拟从文件读取数据
            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )


# 训练循环
def benchmark(dataset, num_epochs=2):
    # 编写一个虚拟的训练循环，以测量迭代数据集所用的时间。训练时间是模拟的
    start_time = time.perf_counter()  # 项目启动的时间 秒
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.01)
    tf.print('Execution time: ', time.perf_counter() - start_time)


# endregion


# region 优化性能(优化 ArtificialDataset)
benchmark(ArtificialDataset())

# region 预提取
# 预提取会与训练步骤的预处理和模型执行重叠
#   在模型执行第 s 步训练的同时，输入流水线会读取第 s+1 步的数据
#   能够最大程度减少训练的单步用时（而非总和），并减少提取数据所需的时间

benchmark(
    ArtificialDataset()
        .prefetch(tf.data.experimental.AUTOTUNE)
)

# endregion


# region 并行数据提取
# 使用 tf.data.Dataset.interleave 转换来并行化数据加载步骤
#   从而交错其他数据集（如数据文件读取器）的内容

# 可以通过 cycle_length 参数指定想要重叠的数据集数量
# 并通过 num_parallel_calls 参数指定并行度级别
# 与 prefetch 转换类似，interleave 转换也支持 tf.data.AUTOTUNE
#   它将让 tf.data 运行时决定要使用的并行度级别

# 顺序交错
#   Dataset.interleave 转换的默认参数会使其按顺序交错两个数据集中的单个样本
benchmark(tf.data.Dataset.range(2).interleave(ArtificialDataset))

# 并行交错
#   num_parallel_calls 可以并行加载多个数据集，从而减少等待打开文件的时间
benchmark(
    tf.data.Dataset.range(2)
        .interleave(ArtificialDataset,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )
)


# endregion


# region 并行数据转换

def mapped_function(s):
    # 做一些预处理
    tf.py_function(lambda: time.sleep(0.03), [], ())
    return s


# 顺序映射（首先使用不具有并行度的 map 转换作为基准示例）
benchmark(ArtificialDataset().map(mapped_function))

# 并行映射（将其并行应用于多个样本）
benchmark(
    ArtificialDataset()
        .map(
        mapped_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
)

# endregion


# region 缓存
# Dataset.cache 转换可以在内存中或本地存储空间中缓存数据集
#   可以避免一些运算（如文件打开和数据读取）在每个周期都被执行

# 缓存数据集时，仅在第一个周期执行一次 cache 之前的转换（如文件打开和数据读取）
#   后续周期将重用通过 cache 转换缓存的数据
benchmark(
    ArtificialDataset().map(  # 在缓存之前应用耗时的操作
        mapped_function
    ).cache(), 5
)

# endregion


# region 向量化映射
# 调用传递给 map 转换的用户定义函数会产生与调度和执行用户定义函数相关的开销
#   对用户定义函数进行向量化处理（即，让它一次运算一批输入）并在 map 转换之前应用 batch 转换


fast_dataset = tf.data.Dataset.range(10000)


def fast_benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for _ in tf.data.Dataset.range(num_epochs):
        for _ in dataset:
            ...
    tf.print('Execution time: ', time.perf_counter() - start_time)


def increment(x):
    return x + 1


# 标量映射
fast_benchmark(
    fast_dataset
        .map(increment)
        .batch(256)
)

# 向量化映射
fast_benchmark(
    fast_dataset
        .batch(256)
        .map(increment)
)


# endregion


# region 减少内存占用
# 建议在 map 转换后缓存数据集，除非此转换会使数据过大而不适合放在内存中
#  如果映射函数可以分成两个部分，则能实现折衷：一个耗时的部分和一个消耗内存的部分
#  在这种情况下，您可以按如下方式将转换链接起来

# dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)

# endregion


# endregion

# 总结
#  1. 使用 prefetch 转换使生产者和使用者的工作重叠。
#  2. 使用 interleave 转换实现并行数据读取转换
#  3. 通过设置 num_parallel_calls 参数实现并行 map 转换
#  4. 第一个周期使用 cache 转换将数据缓存在内存中。
#  5. 向量化传递给 map 转换的用户定义函数
#  6. 应用 interleave、prefetch 和 shuffle 转换时减少内存使用量


# region 重现图表

# region 数据集
# 构建一个返回每步用时的数据集
class TimeMeasuredDataset(tf.data.Dataset):
    OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
    OUTPUT_SHAPES = ((2, 1), (2, 2), (2, 3))

    _INSTANCES_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)

    def _generator(instance_idx, num_samples):
        epoch_idx = next(TimeMeasuredDataset._EPOCHS_COUNTER[instance_idx])

        open_enter = time.perf_counter()
        time.sleep(0.03)
        open_elapsed = time.perf_counter() - open_enter

        for sample_idx in range(num_samples):
            read_enter = time.perf_counter()
            time.sleep(0.015)
            read_elapsed = time.perf_counter() - read_enter

            yield (
                [('Open',), ('Read',)],  # Open 和 Read 是步骤标识符
                [(open_enter, open_elapsed), (read_enter, read_elapsed)],
                # enter: 相应步骤开始时的时间戳; elapsed: 在相应步骤中花费的时间
                [(instance_idx, epoch_idx, -1), (instance_idx, epoch_idx, sample_idx)]
                # instance_idx: 实例索引; epoch_idx: 周期索引（数据集被迭代的次数）; sample_idx: 样本索引
            )
            open_enter, open_elapsed = -1., -1.

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES,
            args=(next(cls._INSTANCES_COUNTER), num_samples)
        )


# endregion


# region 迭代循环
# 使迭代循环稍微复杂一点，以汇总所有计时（仅适用于生成上述样本的数据集）
def timelined_benchmark(dataset, num_epochs=2):
    # 初始化累加器
    steps_acc = tf.zeros([0, 1], tf.dtypes.string)
    times_acc = tf.zeros([0, 2], tf.dtypes.float32)
    values_acc = tf.zeros([0, 3], tf.dtypes.int32)

    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        epoch_enter = time.perf_counter()
        for (steps, times, values) in dataset:
            # 记录数据集准备信息
            steps_acc = tf.concat((steps_acc, steps), axis=0)
            times_acc = tf.concat((times_acc, times), axis=0)
            values_acc = tf.concat((values_acc, values), axis=0)

            # 模拟训练时间
            train_enter = time.perf_counter()
            time.sleep(0.01)
            train_elapsed = time.perf_counter() - train_enter

            # 记录训练信息
            steps_acc = tf.concat((steps_acc, [['Train']]), axis=0)
            times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)
            values_acc = tf.concat((values_acc, [values[-1]]), axis=0)

        # 记录epoch信息
        epoch_elapsed = time.perf_counter() - epoch_enter
        steps_acc = tf.concat((steps_acc, [['Epoch']]), axis=0)
        times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)
        values_acc = tf.concat((values_acc, [[-1, epoch_num, -1]]), axis=0)
        time.sleep(0.001)

    tf.print('Execution time: ', time.perf_counter() - start_time)
    return {'steps': steps_acc, 'times': times_acc, 'values': values_acc}


# endregion


# region 绘图方法
# 定义一个函数，根据 timelined_benchmark 函数返回的值绘制时间线
def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
    # 从时间轴中删除无效条目(负数时间，或空steps)
    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:, 0]
    steps = timeline['steps'][invalid_mask].numpy()
    times = timeline['times'][invalid_mask].numpy()
    values = timeline['values'][invalid_mask].numpy()

    # 获取一组不同的步骤，按照第一次遇到它们的时间进行排序
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # 将起始时间移为0并计算最大时间值
    min_time = times[:, 0].min()
    times[:, 0] = (times[:, 0] - min_time)
    end = max(width, (times[:, 0] + times[:, 1]).max() + 0.01)

    cmap = mpl.cm.get_cmap('plasma')
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(17.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('time(s)')
        ax.set_xticklabels([])
        ax.grid(which='both', axis='x', color='k', linestyle=':')

        # 获取给定步骤的时间和注释
        entries_mark = np.squeeze(steps == step)
        serie = np.unique(times[entries_mark], axis=0)
        annotations = values[entries_mark]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = '\n'.join([f"{l}: {v}" for l, v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(' ', '_')) + '.svg')
    plt.show()


# endregion


# region 对映射函数使用封装容器
def map_decorator(func):
    def wrapper(steps, times, values):
        # 使用tf.py_function来防止auto-graph编译该方法
        return tf.py_function(
            func,
            inp=(steps, times, values),
            Tout=(steps.dtype, times.dtype, values.dtype)
        )

    return wrapper


# endregion


# region 流水线对比
_batch_map_num_items = 50


def dataset_generator_fun(*args):
    return TimeMeasuredDataset(num_samples=_batch_map_num_items)


# 朴素流水线
@map_decorator
def naive_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001)
    time.sleep(0.0001)
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, [['Map']]), axis=0),
        tf.concat((times, [[map_enter, map_elapsed]]), axis=0),
        tf.concat((values, [values[-1]]), axis=0),
    )


# 耗时半分多钟
naive_timeline = timelined_benchmark(
    tf.data.Dataset.range(2)
        .flat_map(dataset_generator_fun)
        .map(naive_map)
        .batch(_batch_map_num_items, drop_remainder=True)
        .unbatch(),
    5
)


# endregion


# region 优化后的流水线
@map_decorator
def time_consuming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001 * values.shape[0])  # 耗时step
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, tf.tile([[['1st map']]], [steps.shape[0], 1, 1])), axis=0),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=0),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=0),
    )


@map_decorator
def memory_consuming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.0001 * values.shape[0])  # 内存消耗step
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, tf.tile([[['2nd map']]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1),
    )


optimized_timeline = timelined_benchmark(
    tf.data.Dataset.range(2)
        .interleave(  # 数据读取并行化
        dataset_generator_fun,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
        .batch(  # 向量化映射的函数
        _batch_map_num_items,
        drop_remainder=True
    )
        .map(  # 并行化映射变换
        time_consuming_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
        .cache()  # 缓存数据
        .map(  # 减少内存使用
        memory_consuming_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
        .prefetch(  # 生产者和消费者的工作重叠
        tf.data.experimental.AUTOTUNE
    )
        .unbatch(),
    5
)

# endregion

draw_timeline(naive_timeline, 'Naive', 15)
draw_timeline(optimized_timeline, "Optimized", 15)

# endregion
