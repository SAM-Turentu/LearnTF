# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: rnn
# CreateTime: 2023/6/29 9:21
# Summary: Keras 中的循环神经网络 (RNN)


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# region 内置 RNN 层：简单示例

# Keras 中有三种内置 RNN 层
#  1. keras.layers.SimpleRNN，一个全连接 RNN，其中前一个时间步骤的输出会被馈送至下一个时间步
#  2. keras.layers.GRU
#  3. keras.layers.LSTM


# 处理整数序列，将每个整数嵌入 64 维向量中，然后使用 LSTM 层处理向量序列
from utils import join_path

model = keras.Sequential()

model.add(layers.Embedding(input_dim=1000, output_dim=64))

model.add(layers.LSTM(128))

model.add(layers.Dense(10))

model.summary()

# 内置 RNN 支持许多实用功能：
# 1. 通过 dropout 和 recurrent_dropout 参数进行循环随机失活
# 2. 能够通过 go_backwards 参数反向处理输入序列
# 3. 通过 unroll 参数进行循环展开（这会大幅提升在 CPU 上处理短序列的速度）
# ......

# endregion


# region 输出和状态

# 设置return_sequences=True，RNN 层返回每个样本的整个输出序列
#   形状为 (batch_size, timesteps, units)

model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# GRU的输出是一个形状为(batch_size, timesteps, 256)的3D张量
model.add(layers.GRU(256, return_sequences=True))
# SimpleRNN的输出将是一个形状为(batch_size, 128)的二维张量
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))
model.summary()

# RNN 层还可以返回其最终内部状态。返回的状态可用于稍后恢复 RNN 执行，或初始化另一个 RNN
#   此设置常用于编码器-解码器序列到序列模型，其中编码器的最终状态被用作解码器的初始状态
#   在创建该层时将 return_state 参数设置为 True
#   LSTM 具有两个状态张量，但 GRU 只有一个

# 要配置该层的初始状态，只需额外使用关键字参数 initial_state 调用该层（状态的形状需要匹配该层的单元大小）

encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
    encoder_input
)

# 除了output， 还返回状态 两个状态
output, state_h, state_c = layers.LSTM(64, return_state=True, name='encoder')(
    encoder_embedded
)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
    decoder_input
)

# 编码器的最后状态 encoder_state 作为 解码器的初始状态 initial_state
decoder_output = layers.LSTM(64, name='decoder')(  # 不需要设置返回状态
    decoder_embedded, initial_state=encoder_state
)
output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input, decoder_input], output)
model.summary()

# endregion


# region RNN 层和 RNN 单元

#  RNN单元 位于 RNN 层的 for 循环内
#   单元封装在 keras.layers.RNN 层内，会得到一个能够处理序列批次的层，如 RNN(LSTMCell(10))
#   从数学上看，RNN(LSTMCell(10)) 会产生和 LSTM(10) 相同的结果
#   使用内置的 GRU 和 LSTM 层，您就能够使用 CuDNN，并获得更出色的性能


# 共有三种内置 RNN 单元，每种单元对应于匹配的 RNN 层
#   1. keras.layers.SimpleRNNCell 对应于 SimpleRNN 层
#   2. keras.layers.GRUCell 对应于 GRU 层
#   3. keras.layers.LSTMCell 对应于 LSTM 层


# endregion


# region 跨批次有状态性

# region Description

# 处理非常长的序列（可能无限长）时，您可能需要使用跨批次有状态性模式
#  通常情况下，每次看到新批次时，都会重置 RNN 层的内部状态
#  如果序列非常长，可将它们拆分成较短的序列，然后将这些较短序列按顺序馈送给 RNN 层，而无需重置该层的状态
#  通过在构造函数中设置 stateful=True 来执行上述操作


# 例如：一个序列 s = [t0, t1, ... t1546, t1547]  拆分为下面的样式
# s1 = [t0, t1, ... t100]
# s2 = [t101, ... t201]
# ...
# s16 = [t1501, ... t1547]

# lstm_layer = layers.LSTM(64, stateful=True)
# for s in sub_sequences:
#     output = lstm_layer(s)

# 清除状态时，您可以使用 layer.reset_states()

paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() 将缓存的状态重置为原始的 initial_state
#  如果没有提供 initial_state，则默认使用零状态
lstm_layer.reset_states()

# endregion


# region RNN 状态重用

# RNN 层的记录状态不包含在 layer.weights() 中
#  重用 RNN 层的状态，可以通过 layer.states 找回状态值
#  并通过 Keras 函数式 API（如 new_layer(inputs, initial_state=layer.states)）或模型子类化将其用作新层的初始状态

# 此情况可能不适用于序贯模型，因为它只支持具有单个输入和输出的层，而初始状态具有额外输入，因此无法在此使用

paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)

existing_state = lstm_layer.states

new_lstm_layer = layers.LSTM(64)
new_output = new_lstm_layer(paragraph3, initial_state=existing_state)

# endregion


# endregion


# region 双向 RNN
# RNN 模型不仅能从头到尾处理序列，而且还能反向处理
#   例如，要预测句子中的下一个单词，通常比较有用的是掌握单词的上下文，而非仅仅掌握该单词前面的单词

# Keras 为您提供了一个简单的 API 来构建此类双向 RNN：keras.layers.Bidirectional 封装容器

model = keras.Sequential()

model.add(
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
)
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10))
model.summary()

# Bidirectional 会在后台复制传入的 RNN 层，并翻转新复制的层的 go_backwards 字段，这样它就能按相反的顺序处理输入了

# endregion


# region 性能优化和 CuDNN 内核

# 如果您更改了内置 LSTM 或 GRU 层的默认设置，则该层将无法使用 CuDNN 内核
#  1. 将 activation 函数从 tanh 更改为其他
#  2. 将 recurrent_activation 函数从 sigmoid 更改为其他
#  3. 使用大于零的 recurrent_dropout
#  4. 将 unroll 设置为 True，这会强制 LSTM/GRU 将内部 tf.while_loop 分解成未展开的 for 循环
#  5. 将 use_bias 设置为 False
#  6. 当输入数据没有严格正确地填充时使用遮盖（如果掩码对应于严格正确的填充数据，则仍可使用 CuDNN。这是最常见的情况）

# region 在可用时使用 CuDNN 内核
# 构建一个简单的 LSTM 模型来演示性能差异
#  使用 MNIST 数字的行序列作为输入序列（将每一行像素视为一个时间步骤），并预测数字的标签


batch_size = 64
# 每个MNIST图像批处理都是一个形状为(batch_size, 28, 28)的张量
# 每个输入序列的大小为(28,28)(高度与时间一样)
input_dim = 28

units = 64
output_size = 10


# 创建一个 RNN 模型
def build_model(allow_cudnn_kernet=True):
    """
    CuDNN仅在层级可用，而在单元级不可用
    LSTM(units) 将使用CuDNN内核
    RNN(LSTMCell(units)) 将在非CuDNN内核上运行
    :param allow_cudnn_kernet:
    :return:
    """
    if allow_cudnn_kernet:
        # 具有默认选项的 LSTM 层使用 CuDNN
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # 将 LSTMCell 包装在RNN层中不会使用 CuDNN
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size)
        ]
    )
    return model


mnist = keras.datasets.mnist
path = join_path.keras_data_path.helloworld_keras_mnist_data_path
path = path.replace('BaseLearnTF\\', '')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

# 使用 CuDNN 内核的模型，使用 GPU 计算
model = build_model(allow_cudnn_kernet=True)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='sgd',
    metrics=['accuracy']
)
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1)

# 未使用 CuDNN 内核的模型，使用 CPU 计算
noncudnn_model = build_model(allow_cudnn_kernet=False)
noncudnn_model.set_weights(model.get_weights())
noncudnn_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='sgd',
    metrics=['accuracy']
)
noncudnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1)

# 强制使用 CPU
with tf.device('CPU:0'):
    cpu_model = build_model(allow_cudnn_kernet=True)
    cpu_model.set_weights(model.get_weights())
    result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
    print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
    # Predicted result is: [3], target result is: 5
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
    plt.show()


# endregion


# endregion


# region 支持列表/字典输入或嵌套输入的 RNN

# 定义一个支持嵌套输入/输出的自定义单元

class NestedCell(keras.layers.Layer):

    def __init__(self, unit_1, unit_2, unit_3, **kwargs):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.unit_3 = unit_3
        self.state_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
        self.output_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
        super(NestedCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        # 预计 input_shape 包含2个元素，[(batch, i1)， (batch, i2, i3)]
        i1 = input_shapes[0][1]
        i2 = input_shapes[1][1]
        i3 = input_shapes[1][2]

        self.kernel_1 = self.add_weight(
            shape=(i1, self.unit_1), initializer='uniform', name='kernel_1'
        )
        self.kernel_2_3 = self.add_weight(
            shape=(i2, i3, self.unit_2, self.unit_3),
            initializer='uniform',
            name='kernel_2_3'
        )

    def call(self, inputs, states):
        # 输入应为 [(batch, input_1)， (batch, input_2, input_3)]
        # 状态应为 [(batch, unit_1)， (batch, unit_2, unit_3)]
        input_1, input_2 = tf.nest.flatten(inputs)
        s1, s2 = states

        output_1 = tf.matmul(input_1, self.kernel_1)
        # Einsum 允许通过定义元素计算来定义张量。这个计算由`equation`定义，它是基于爱因斯坦求和的缩写形式。举个例子，考虑将两个矩阵A和B相乘得到矩阵C
        output_2_3 = tf.einsum('bij,ijkl->bkl', input_2, self.kernel_2_3)
        state_1 = s1 + output_1
        state_2_3 = s2 + output_2_3

        output = (output_1, output_2_3)
        new_states = (state_1, state_2_3)

        return output, new_states

    def get_config(self):
        return {"unit_1": self.unit_1, "unit_2": unit_2, "unit_3": self.unit_3}


# region 使用嵌套输入/输出构建 RNN 模型

# 构建一个使用 keras.layers.RNN 层和刚刚定义的自定义单元的 Keras 模型

unit_1 = 10
unit_2 = 20
unit_3 = 30

i1 = 32
i2 = 64
i3 = 32

batch_size = 64
num_batches = 10
timestep = 50

cell = NestedCell(unit_1, unit_2, unit_3)
rnn = keras.layers.RNN(cell)

input_1 = keras.Input((None, i1))
input_2 = keras.Input((None, i2, i3))

outputs = rnn((input_1, input_2))

model = keras.models.Model([input_1, input_2], outputs)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# endregion


# region 使用随机生成的数据训练模型
input_1_data = np.random.random((batch_size * num_batches, timestep, i1))
input_2_data = np.random.random((batch_size * num_batches, timestep, i2, i3))
target_1_data = np.random.random((batch_size * num_batches, unit_1))
target_2_data = np.random.random((batch_size * num_batches, unit_2, unit_3))
input_data = [input_1_data, input_2_data]
target_data = [target_1_data, target_2_data]

model.fit(input_data, target_data, batch_size=batch_size)

# endregion

# endregion
