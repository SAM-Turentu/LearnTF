# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: masking_and_padding
# CreateTime: 2023/6/29 17:25
# Summary: Keras 中的遮盖和填充


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 【遮盖】的作用是告知序列处理层输入中有某些时间步骤丢失，因此在处理数据时应将其跳过
# 【填充】是遮盖的一种特殊形式，其中被遮盖的步骤位于序列的起点或开头
#   填充是出于将序列数据编码成连续批次的需要：
#     为了使批次中的所有序列适合给定的标准长度，有必要填充或截断某些序列


# region 填充序列数据

# 截断和填充 Python 列表，使其具有相同长度：tf.keras.preprocessing.sequence.pad_sequences

# [
#   ["Hello", "world", "!"],
#   ["How", "are", "you", "doing", "today"],
#   ["The", "weather", "will", "be", "nice", "tomorrow"],
# ]

# 进行词汇查询后，数据可能会被向量化为整数
raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

# 默认用 0 填充，value 设置; padding='post' 在开始填充；padding='pre' 在结束填充
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding='post'
)
print(padded_inputs)

# endregion


# region 遮盖
# 【遮盖】：所有样本现在都具有了统一长度，那就必须告知模型，数据的某些部分实际上是填充，应该忽略

# 在 Keras 模型中引入输入掩码有三种方式：
#  1. 添加一个 keras.layers.Masking 层
#  2. 使用 mask_zero=True 配置一个 keras.layers.Embedding 层
#  3. 在调用支持 mask 参数的层（如 RNN 层）时，手动传递此参数

# endregion


# region 掩码生成层：Embedding 和 Masking

# 这些层将在后台创建一个掩码张量（形状为 (batch, sequence_length) 的二维张量）
#  并将其附加到由 Masking 或 Embedding 层返回的张量输出上

embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
masked_output = embedding(padded_inputs)

print(masked_output._keras_mask)

masking_layer = layers.Masking()

unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]), tf.float32
)

masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)

# 每个 False 条目表示对应的时间步骤应在处理时忽略

# endregion


# region 函数式 API 和序列式 API 中的掩码传播
# 使用函数式 API 或序列式 API 时，由 Embedding 或 Masking 层生成的掩码将通过网络传播给任何能够使用它们的层

# 序贯模型 LSTM层将自动接收掩码，它将忽略填充的值
model = keras.Sequential(
    [
        layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True),
        layers.LSTM(32)
    ]
)

# 函数式API 的情况跟 上面相同
inputs = keras.Input(shape=(None,), dtype='int32')
x = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
outputs = layers.LSTM(32)(x)
a = keras.Model(inputs, outputs)


# endregion


# region 将掩码张量直接传递给层
# 能够处理掩码的层（如 LSTM 层）在其 __call__ 方法中有一个 mask 参数
# 同时，生成掩码的层（如 Embedding）会公开一个 compute_mask(input, previous_mask) 方法

class MyLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
        self.lstm = layers.LSTM(32)

    def call(self, inputs):
        x = self.embedding(inputs)

        # 准备一个`mask`张量,形状(batch_size, timesteps)
        mask = self.embedding.compute_mask(inputs)
        output = self.lstm(x, mask=mask)  # 该层将忽略被屏蔽的值
        return output


layer = MyLayer()
x = np.random.random((32, 10)) * 100
x = x.astype('int32')
layer(x)


# endregion


# region 在自定义层中支持遮盖

# 任何生成与其输入具有不同时间维度的张量的层都需要修改当前掩码，
#   这样下游层才能正确顾及被遮盖的时间步骤

# 层应实现 layer.compute_mask() 方法，该方法会根据输入和当前掩码生成新的掩码

class TemporalSplit(keras.layers.Layer):
    """ 沿时间维度将输入张量分割为2个张量 """

    def call(self, inputs):
        # 期望输入为3D，掩码为2D，将输入张量沿时间轴(轴1)划分为2个子张量
        return tf.split(inputs, 2, axis=1)

    def compute_mask(self, inputs, mask=None):
        # 如果掩码出现，也将其分成2个
        if mask is None:
            return None
        return tf.split(mask, 2, axis=1)


first_half, second_half = TemporalSplit()(masked_embedding)
print(first_half._keras_mask)
print(second_half._keras_mask)


# CustomEmbedding 该层能够根据输入值 生成掩码
class CustomEmbedding(keras.layers.Layer):

    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='random_normal',
            dtype='float32'
        )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)


layer = CustomEmbedding(10, 32, mask_zero=True)
x = np.random.random((3, 10)) * 9
x = x.astype('int32')

y = layer(x)
mask = layer.compute_mask(x)
print(mask)


# endregion


# region 在兼容层上选择启用掩码传播
# 默认情况下，自定义层将破坏当前掩码（因为框架无法确定传播该掩码是否安全）

# 一个不会修改时间维度的自定义层，且希望能够传播当前的输入掩码
#  在层构造函数中设置 self.supports_masking = True
#  在这种情况下，compute_mask() 的默认行为是仅传递当前掩码

class MyAcitvation(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MyAcitvation, self).__init__(**kwargs)
        # 信号表明该层是安全的掩码传播
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)


inputs = keras.Input(shape=(None,), dtype='int32')
x = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
x = MyAcitvation()(x)  # 会把 mask 传下去
print("Mask found:", x._keras_mask)
outputs = layers.LSTM(32)(x)  # 会收到 mask
model = keras.Model(inputs, outputs)


# endregion


# region 编写需要掩码信息的层
# 有些层是掩码使用者：在 call 中接受 mask 参数，并使用该参数来决定是否跳过某些时间步骤
#   编写这样的层，需在 call 签名中添加一个 mask=None 参数

# 例：层在输入序列的时间维度（轴 1）上计算 Softmax，同时丢弃遮盖的时间步骤。
class TemporalSoftmax(keras.layers.Layer):

    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), -1)
        inputs_exp = tf.exp(inputs) * broadcast_float_mask
        input_sum = tf.reduce_sum(
            inputs_exp * broadcast_float_mask, axis=-1, keepdims=True
        )
        return inputs_exp / input_sum


inputs = keras.Input(shape=(None,), dtype='int32')
x = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
x = layers.Dense(1)(x)
outputs = TemporalSoftmax()(x)

model = keras.Model(inputs, outputs)
y = model(np.random.randint(0, 10, size=(32, 100)), np.random.random((32, 100, 1)))

# endregion

# 总结

# “遮盖”是层得知何时应该跳过/忽略序列输入中的某些时间步骤的方式。
# 有些层是掩码生成者：Embedding 可以通过输入值来生成掩码（如果 mask_zero=True），Masking 层也可以。
# 有些层是掩码使用者：它们会在其 __call__ 方法中公开 mask 参数。RNN 层就是如此。
# 在函数式 API 和序列式 API 中，掩码信息会自动传播。
# 单独使用层时，您可以将 mask 参数手动传递给层。
# 您可以轻松编写会修改当前掩码的层、生成新掩码的层，或使用与输入关联的掩码的层。
