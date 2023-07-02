# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: ragged_tensor
# CreateTime: 2023/6/15 15:08
# Summary: 不规则张量


import math
import tempfile

import tensorflow as tf
import google.protobuf.text_format as pbtext

# region 概述

# region 不规则张量的功能

digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
words = tf.ragged.constant([['So', 'long'], ['thanks', 'for', 'all', 'the', 'fish']])
print(tf.add(digits, 3))  # 不为空的值 + 3
print(tf.reduce_mean(digits, axis=1))  # 平均值 [2.25, nan, 5.333333333333333, 6.0, nan]
print(tf.concat([digits, [[5, 3]]], axis=0))  # 后面添加 [5, 3]
print(tf.tile(digits, [1, 2]))  # 每个扩充
print(tf.strings.substr(words, 0, 2))  # 截取[0, 2]字符
print(tf.map_fn(tf.math.square, digits))  # 平方

print(digits[:, -2:])  # [[4, 1], [], [9, 2], [6], []]

times_two_plus_one = lambda x: x * 2 + 1
tf.ragged.map_flat_values(times_two_plus_one, digits)  # [[7, 3, 9, 3], [], [11, 19, 5], [13], []]

digits.to_list()  # 不规则张量可以转换为嵌套的 python list  Numpy array
digits.numpy()

# endregion

# region 构造不规则张量
#  使用 tf.ragged.constant

sentences = tf.ragged.constant([
    ["Let's", 'build', 'some', 'ragged', 'tensors', '!'],
    ['We', 'can', 'use', 'tf.ragged.constant', '.']
])

# 1. 知道每个值的位置，使用 value_rowids
a = tf.RaggedTensor.from_value_rowids(
    values=[3, 1, 4, 1, 5, 9, 2],
    value_rowids=[0, 0, 0, 0, 2, 2, 3]  # 每个值的位置
)  # [[3, 1, 4, 1], [], [5, 9], [2]]

# 2. 知道每行的长度，使用 row_lengths
b = tf.RaggedTensor.from_row_lengths(
    values=[3, 1, 4, 1, 5, 9, 2],
    row_lengths=[4, 0, 2, 1]
)  # [[3, 1, 4, 1], [], [5, 9], [2]]

# 3. 知道每行开始和结束的索引，使用 row_splits
c = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2],
    row_splits=[0, 4, 4, 6, 7]
)  # [[3, 1, 4, 1], [], [5, 9], [2]]

# 4. 有关完整的工厂方法列表，请参阅 tf.RaggedTensor 类文档。

# endregion

# region 不规则张量中 存储的值
#  值 【类型】 必须相同
#  值必须处于相同 的【嵌套深度】（【张量的秩】）

tf.ragged.constant([['Hi'], ['How', 'are', 'you']])
# tf.ragged.constant([['Hi'], ['How', 'are', 'you'], 'as'])  # Exception
# tf.ragged.constant([['Hi'], ['How', 'are', 1]])  # Exception

# endregion

# endregion


# region 示例

queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],
                              ['Pause'],
                              ['Will', 'it', 'rain', 'later', 'today']])

num_buckets = 1024
embedding_size = 4
embedding_table = tf.Variable(
    tf.random.truncated_normal([num_buckets, embedding_size],
                               stddev=1. / math.sqrt(embedding_size))
)

# 查找每个单词的嵌入
word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
word_embddings = tf.nn.embedding_lookup(embedding_table, word_buckets)

# 在每个句子的开头和结尾添加标记
marker = tf.fill([queries.nrows(), 1], '#')
padded = tf.concat([marker, queries, marker], axis=1)

# 构建词二元分词并查找嵌入。
bidgrams = tf.strings.join([padded[:, :-1], padded[:, 1:]], separator='+')

bigram_buckets = tf.strings.to_hash_bucket_fast(bidgrams, num_buckets)
bigram_embeddings = tf.nn.embedding_lookup(embedding_table, bigram_buckets)

# 找到每个句子的平均嵌入
all_embeddings = tf.concat([word_embddings, bigram_embeddings], axis=1)
avg_embedding = tf.reduce_mean(all_embeddings, axis=1)

print(avg_embedding)

# endregion


# region 不规则维度和均匀维度

# 不规则张量的维度大小为 None
print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]).shape)  # (2, None)

# 查找给定 RaggedTensor 的紧密边界形状，使用 tf.RaggedTensor.bounding_shape
print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]).bounding_shape())
# tf.Tensor([2 3], shape=(2,), dtype=int64)


# endregion


# region 不规则张量和稀疏张量对比

# 1 对密集张量 或 稀疏张量 运算 始终获得相同结果
# 2 对不规则张量 或 稀疏张量 运算 可能获得不同结果

# 连接不规则张量
ragged_x = tf.ragged.constant([["John"], ["a", "big", "dog"], ["my", "cat"]])
ragged_y = tf.ragged.constant([["fell", "asleep"], ["barked"], ["is", "fuzzy"]])
print(tf.concat([ragged_x, ragged_y], axis=1))

# 连接稀疏张量，相当于连接密集张量
spares_x = ragged_x.to_sparse()
spares_y = ragged_y.to_sparse()
spares_result = tf.sparse.concat(sp_inputs=[spares_x, spares_y], axis=1)
print(tf.sparse.to_dense(spares_result, ''))  # 填充为空，缺失值

# endregion


# region TensorFlow API


# region Keras
# 预测每个句子是否为问句

sentences = tf.constant(
    ['What makes you think she is a witch?',
     'She turned me into a newt.',
     'A newt?',
     'Well, I got better.'])
is_question = tf.constant([True, False, True, False])

hash_buckets = 1000
words = tf.strings.split(sentences, ' ')
hashed_words = tf.strings.to_hash_bucket_fast(words, hash_buckets)

keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[None], dtype=tf.int64, ragged=True),
    tf.keras.layers.Embedding(hash_buckets, 16),
    tf.keras.layers.LSTM(32, use_bias=False),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation(tf.nn.relu),
    tf.keras.layers.Dense(1),
])

keras_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
keras_model.fit(hashed_words, is_question, epochs=5)
print(keras_model.predict(hashed_words))


# endregion


# region tf.Example  数据往往包括可变长度特征

def build_tf_example(s):
    return pbtext.Merge(s, tf.train.Example()).SerializeToString()


# 不同特征长度
example_batch = [
    build_tf_example(r'''
        features {
            feature {key: "colors" value {bytes_list {value: ["red", "blue"]} } }
            feature {key: "lengths" value {int64_list {value: [7]} } } }'''),
    build_tf_example(r'''
        features {
            feature {key: "colors" value {bytes_list {value: ["orange"]} } }
            feature {key: "lengths" value {int64_list {value: []} } } }'''),
    build_tf_example(r'''
        features {
            feature {key: "colors" value {bytes_list {value: ["black", "yellow"]} } }
            feature {key: "lengths" value {int64_list {value: [1, 3]} } } }'''),
    build_tf_example(r'''
        features {
            feature {key: "colors" value {bytes_list {value: ["green"]} } }
            feature {key: "lengths" value {int64_list {value: [3, 5, 2]} } } }'''),
]

feature_specification = {
    'colors': tf.io.RaggedFeature(tf.string),  # 将可变长度特征读入不规则张量
    'lengths': tf.io.RaggedFeature(tf.int64),
}
feature_tensors = tf.io.parse_example(example_batch, feature_specification)
for name, value in feature_tensors.items():
    print(f'{name} = {value}')


# endregion


# region 数据集

def print_dictionary_dataset(dataset):
    for i, element in enumerate(dataset):
        print(f'Element: {i}')
        for (feature_name, feature_value) in element.items():
            print(f'{feature_name:>14} = {feature_value}')


# 使用不规则张量构建数据集
dataset = tf.data.Dataset.from_tensor_slices(feature_tensors)
print_dictionary_dataset(dataset)

# 批处理和取消批处理具有不规则张量的数据集，使用 Dataset.batch
batched_dataset = dataset.batch(2)
print_dictionary_dataset(batched_dataset)

# Dataset.unbatch 将批处理后的数据集转换为扁平数据集
unbatched_dataset = batched_dataset.unbatch()
print_dictionary_dataset(unbatched_dataset)

# 对具有可变长度 非不规则张量 的数据集进行批处理
#  一个非不规则张量的数据集,且各个元素的张量长度不同,可用 dense_to_ragged_batch 转换，将这些 非不规则张量 批处理成 不规则张量
non_ragged_dataset = tf.data.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
non_ragged_dataset = non_ragged_dataset.map(tf.range)
batched_non_ragged_dataset = non_ragged_dataset.apply(
    tf.data.experimental.dense_to_ragged_batch(2))
for element in batched_non_ragged_dataset:
    print(element)


# 转换具有不规则张量的数据集
#  使用 Dataset.map 在数据集中创建或转换 不规则张量
def transform_lengths(features):
    return {
        'mean_length': tf.math.reduce_mean(features['lengths']),
        'length_ranges': tf.ragged.range(features['lengths'])
    }


transform_dataset = dataset.map(transform_lengths)
print_dictionary_dataset(transform_dataset)


# endregion


# region tf.function


@tf.function
def make_palindrome(x, axis):
    """ 对不规则张量 和 非不规则张量 均有效 """
    return tf.concat([x, tf.reverse(x, [axis])], axis)


make_palindrome(tf.constant([[1, 2], [3, 4], [5, 6]]), axis=1)

make_palindrome(tf.ragged.constant([[1, 2], [3, 4], [5, 6]]), axis=1)


@tf.function(
    input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)])
def max_and_min(rt):
    return (tf.math.reduce_max(rt, axis=-1), tf.math.reduce_min(rt, axis=-1))


max_and_min(tf.ragged.constant([[1, 2], [3], [4, 5, 6]]))

# 具体函数
#  不规则张量可以与具体函数一起使用
try:
    @tf.function
    def increment(x):
        return x + 1


    rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
    cf = increment.get_concrete_function(rt)
    print(cf(rt))
except Exception as e:
    print(f'Not supported before TF {type(e)}: {e}')
# endregion


# region SavedModel

# keras_module_path = tempfile.mkdtemp()
# tf.saved_model.save(keras_model, keras_module_path)
# imported_model = tf.saved_model.load(keras_module_path)
# imported_model(hashed_words)

# endregion

# endregion


# region 重载运算符

x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
y = tf.ragged.constant([[1, 1], [2], [3, 3, 3]])
print(x + y)  # [[2, 3], [5], [7, 8, 9]]
print(x + 3)  # [[4, 5], [6], [7, 8, 9]]

# endregion


# region 索引

# 二维不规则张量
queries = tf.ragged.constant(
    [['Who', 'is', 'George', 'Washington'],
     ['What', 'is', 'the', 'weather', 'tomorrow'],
     ['Goodnight']])
print(queries[1][1])  # == queries[1, 1]
print(queries[1:, 1:])

# 三维不规则张量
rt = tf.ragged.constant([
    [[1, 2, 3], [4]],
    [[5], [], [6]],
    [[7]],
    [[8, 9], [10]]
])

print(rt[1, 0, 0])  # 5
print(rt[1, 1:])  # [], [6]
print(rt[:, -1:])  # <tf.RaggedTensor [[[4]],  [[6]],  [[7]],  [[10]]]>
# endregion


# region 张量类型转换

# RaggedTensor -> Tensor
ragged_sentences = tf.ragged.constant([['Hi'], ['Welcome', 'to', 'the', 'fair'], ['Have', 'fun']])
print(ragged_sentences.to_tensor(default_value='', shape=[None, 10]))

# Tensor -> RaggedTensor
x = [[1, 3, -1, -1], [2, -1, -1, -1], [4, 5, 8, 9]]
print(tf.RaggedTensor.from_tensor(x, padding=-1))

# RaggedTensor -> SparseTensor
print(ragged_sentences.to_sparse())

# SparseTensor -> RaggedTensor
st = tf.SparseTensor(indices=[[0, 0], [2, 0], [2, 1]],
                     values=['a', 'b', 'c'],
                     dense_shape=[3, 3])
print(tf.RaggedTensor.from_sparse(st))  # [[b'a'], [], [b'b', b'c']]

# endregion


# region 评估不规则张量

# 访问不规则张量中的值
rt = tf.ragged.constant([[1, 2], [3, 4, 5], [6], [], [7]])
print("Python list:", rt.to_list())
print("NumPy array:", rt.numpy())
print("Values:", rt.values.numpy())
print("Splits:", rt.row_splits.numpy())
print("Indexed value:", rt[1].numpy())

# endregion


# region 不规则形状

# 静态形状；不规则维度的静态形状始终为 None
x = tf.constant([[1, 2], [3, 4], [5, 6]])  # Shape([3, 2])
rt = tf.ragged.constant([[1], [2, 3], [], [4]])  # Shape([4, None])

# region 动态形状

x = tf.constant([['a', 'b'], ['c', 'd'], ['e', 'f']])
tf.shape(x)  # tf.shape 将形状作为一维整数Tensor返回，其中tf.shape(x)[i] 为轴i的大小

# 不规则张量的形状 使用专用类型 DynamicRaggedShape
rt = tf.ragged.constant([[1], [2, 3, 4], [], [5, 6]])
rt_shape = tf.shape(rt)
print(rt_shape)  # [4, (1, 3, 0, 2)]

# 动态形状：运算
# 可与大多数需要形状的tf运算一起使用，包括tf.reshape  tf.zeros  tf.ones  tf.fill  tf.broadcast_dynamic_shape tf,broadcast_to
print(f'tf.reshape(x, rt_shape) = {tf.reshape(x, rt_shape)}')
print(f'tf.zeros(rt_shape) = {tf.zeros(rt_shape)}')
print(f'tf.ones(rt_shape) = {tf.ones(rt_shape)}')
print(f'tf.fill(rt_shape, 9) = {tf.fill(rt_shape, 9)}')

# 动态形状：索引和切片

print(rt_shape[0])  # 使用索引尝试 检索 不规则维度的大小是一种错误

try:
    print(rt_shape[1])
except Exception as e:
    print(e)

# 可以对 DynamicRaggedShape 进行切片，前提是切片从 轴0 开始，或者仅包含密集维度
print(rt_shape[:1])

# 动态形状：编码
#  DynamicRaggedShape 使用两个字段进行编码
#    inner_shape 一个整数向量，给出密集（tf.Tensor）的形状
#    row_partitions  tf.experimental.RowPartition 对象列表，描述了应当如何对该内部形状的最外层维度进行分区以添加不规则轴

# 动态形状：构造

# DynamicRaggedShape 最常通过将 tf.shape 应用于 RaggedTensor 来构造，但也可以直接构造
tf.experimental.DynamicRaggedShape(
    row_partitions=[tf.experimental.RowPartition.from_row_lengths([5, 3, 2])],
    inner_shape=[10, 8]
)

# 如果所有行的长度都是静态已知的，DynamicRaggedShape.from_lengths 也可用于构造动态不规则形状
tf.experimental.DynamicRaggedShape.from_lengths([4, (2, 1, 0, 8), 12])

# endregion


# region 广播
# 广播是使具有不同形状的张量获得兼容形状以便进行逐元素运算的过程
# 广播两个输入 x  y 使其具有兼容形状的步骤：
#   1. 如果 x  y 没有相同的维数， 则增加外层维数(使用大小1)，直至维数相同
#   2. 对于 x  y 的大小不同的每个维度：
#       a. 如果 x  y 在d维中的大小为1，则在d维中重复其值以匹配其他输入的大小
#       b. 否则，引发异常（x  y 非广播兼容）

x = tf.ragged.constant([[1, 2], [3]])
y = 3
print(x + y)
# <tf.RaggedTensor [[4, 5], [6]]>

x = tf.ragged.constant(
    [[10, 87, 12],
     [19, 53],
     [12, 32]])
y = [[1000], [2000], [3000]]
print(x + y)  # [[1010, 1087, 1012], [2019, 2053], [3012, 3032]]

x = tf.ragged.constant(
    [[[1, 2], [3, 4], [5, 6]],
     [[7, 8]]],
    ragged_rank=1)
y = tf.constant([[10]])
print(x + y)
# <tf.RaggedTensor [[[11, 12],
#   [13, 14],
#   [15, 16]], [[17, 18]]]>

x = tf.ragged.constant(
    [
        [
            [[1], [2]],
            [],
            [[3]],
            [[4]],
        ],
        [
            [[5], [6]],
            [[7]]
        ]
    ],
    ragged_rank=2)
y = tf.constant([10, 20, 30])
print(x + y)  # == a
a = [
    [
        [[11, 21, 31], [12, 22, 32]],
        [],
        [[13, 23, 33]],
        [[14, 24, 34]]
    ],
    [
        [[15, 25, 35], [16, 26, 36]],
        [[17, 27, 37]]
    ]
]

x = tf.ragged.constant([[1, 2], [3, 4, 5, 6], [7]])  # y = tf.constant([[1], [5], [9]])  # x + y =  [[2, 3], [8, 9, 10, 11], [16]]
y = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
try:
    x + y
except tf.errors.InvalidArgumentError as exception:
    print(exception)

x = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
y = tf.ragged.constant([[10, 20], [30, 40], [50]])
try:
    x + y
except tf.errors.InvalidArgumentError as exception:
    print(exception)

x = tf.ragged.constant([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10]]])
y = tf.ragged.constant([[[1, 2, 0], [3, 4, 0], [5, 6, 0]],
                        [[7, 8, 0], [9, 10, 0]]])
try:
    x + y
except tf.errors.InvalidArgumentError as exception:
    print(exception)

# endregion


# endregion


# region RaggedTensor 编码

# row_splits 指定行之间的拆分点【高效索引】
#   可以实现不规则张量的恒定时间索引和切片
# value_rowids 指定每个值的行索引【较小的编码大小、兼容性】
#   在存储具有大量空行的不规则张量时更有效
# row_lengths  指定每一行的长度【高效连接】
#   在连接不规则张量时更有效，因为当两个张量连接在一起时，行长度不会改变
# uniform_row_length  指定所有行的单个长度【均匀维】

rt = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2],
    row_splits=[0, 4, 4, 6, 7])

# region 多个不规则维度

# 多个不规则维度的不规则张量通过为 values 张量使用嵌套 RaggedTensor
rt = tf.RaggedTensor.from_row_splits(
    values=tf.RaggedTensor.from_row_splits(
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        row_splits=[0, 3, 3, 5, 9, 10]
    ),
    row_splits=[0, 1, 1, 5]
)  # [[10, 11, 12]], [], [[], [13, 14], [15,16,17,18],[19]]

print(f'Shape: {rt.shape}')
print(f'Number of partitioned dimensions: {rt.ragged_rank}')

# 工厂函数 tf.RaggedTensor.from_nested_row_splits 可用于通过提供一个 row_splits 张量列表,直接构造具有多个不规则维度的 RaggedTensor
rt = tf.RaggedTensor.from_nested_row_splits(
    flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10])
)  # [[[10, 11, 12]], [], [[], [13, 14], [15, 16, 17, 18], [19]]]
print(rt)

# endregion


# region 不规则秩和扁平值

# 不规则张量的不规则秩是底层 values 张量的分区次数（即 RaggedTensor 对象的嵌套深度）
#  最内层的 values 张量称为其 flat_values
conversations = tf.ragged.constant([
    [
        [['I', 'like', 'ragged', 'tensors.']],
        [['Oh', 'yeah?'], ['What', 'can', 'you', 'use', 'them', 'for?']],
        [['Processing', 'variable', 'length', 'data!']]
    ],
    [
        [['I', 'like', 'cheese.'], ['Do', 'you?']],
        [['Yes.'], ['I', 'do.']]
    ]

])

assert conversations.ragged_rank == len(conversations.nested_row_splits)  # == 3
print(conversations.flat_values.numpy())

# endregion


# region 均匀内层维度

# 均匀内层维度的不规则张量通过为 flat_values（即最内层 values）使用多维 tf.Tensor 进行编码
rt = tf.RaggedTensor.from_row_splits(
    values=[[1, 3], [0, 0], [1, 3], [5, 3], [3, 3], [1, 2]],
    row_splits=[0, 3, 4, 6]
)
print(f'shape: {rt.shape}')  # (3, None, 2)  # 内层维度均匀
print("Number of partitioned dimensions: {}".format(rt.ragged_rank))  # 1
print("Flat values shape: {}".format(rt.flat_values.shape))  # (6, 2)
print("Flat values:\n{}".format(rt.flat_values))

# endregion


# region 均匀非内层维度

# 均匀非内层维度的不规则张量通过使用 uniform_row_length 对行分区进行编码
rt = tf.RaggedTensor.from_uniform_row_length(
    values=tf.RaggedTensor.from_row_splits(
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        row_splits=[0, 3, 5, 9, 10]
    ),
    uniform_row_length=2
)
print(rt)
print("Shape: {}".format(rt.shape))
print("Number of partitioned dimensions: {}".format(rt.ragged_rank))

# endregion


# endregion
