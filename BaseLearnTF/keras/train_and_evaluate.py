# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: train_and_evaluate
# CreateTime: 2023/6/24 9:51
# Summary: 使用内置方法进行训练和评估


import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras import layers
from utils import join_path

# region API 概述：第一个端到端示例

# 训练
# 根据从原始训练数据生成的预留集进行验证
# 对测试数据进行评估


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

file_path = join_path.keras_data_path.helloworld_keras_mnist_data_path  # 多了一个 BaseLearnTF\\

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(file_path)

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
#
# model.compile(
#     optimizer=keras.optimizers.RMSprop(),  # Optimizer
#     # Loss function to minimize
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     # List of metrics to monitor
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
#
# print("Fit model on training data")
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=2,
#     # We pass some validation for
#     # monitoring validation loss and metrics
#     # at the end of each epoch
#     validation_data=(x_val, y_val),
# )
#
# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
#
# # Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test[:3])
# print("predictions shape:", predictions.shape)

# endregion


# region compile() 方法：指定损失、指标和优化器

# model.compile(
#     optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()]
# )

# 可以使用字符串标识符将优化器、损失 和 指标 指定为捷径
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)


# 为方便以后重用，我们将模型定义和编译步骤放入函数中
def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model


# endregion


# region 提供许多内置优化器、损失和指标
# 优化器：
# SGD()（有或没有动量）
# RMSprop()
# Adam()

# 损失：
# MeanSquaredError()
# KLDivergence()
# CosineSimilarity()

# 指标：
# AUC()
# Precision()
# Recall()

# region 自定义损失函数

#  第一种方式涉及创建一个接受输入 y_true 和 y_pred 的函数
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)


class CustomMSE(keras.losses.Loss):

    def __init__(self, regularization_factor=0.1, name='custom_mse'):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


model = get_compiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)


# endregion


# region 自定义指标

# 通过将 tf.keras.metrics.Metric 类子类化创建自定义指标，需要实现 4 种方法
#  __init__(self)，指标创建状态变量
#  update_state(self, y_true, y_pred, sample_weight=None)，它使用目标 y_true 和模型预测 y_pred 来更新状态变量
#  result(self)，它使用状态变量来计算最终结果
#  reset_state(self)，它重新初始化指标的状态


class CategoricalTruePositives(keras.metrics.Metric):

    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='ctp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    # 状态更新和结果计算是分开进行的
    # 在某些情况下，结果计算的开销可能非常巨大并且只能定期进行

    def result(self):
        return self.true_positives

    def reset_state(self):
        # 度量的状态会在每次迭代开始时重置。
        self.true_positives.assign(0.)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()]
)
model.fit(x_train, y_train, batch_size=64, epochs=3)


# endregion


# region 处理不适合标准签名的损失和指标

# 绝大多数损失和指标都可以通过 y_true 和 y_pred 计算得出，其中 y_pred 是模型的输出，但不是全部

# 正则化损失可能仅需要激活层，这种激活可能不是模型输出
#  以从自定义层的调用方法内部调用 self.add_loss(loss_value)
#  添加的损失会在训练期间添加到“主要”损失中（传递给 compile() 的损失）

class AcitvityRegularizationLayer(layers.Layer):

    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

x = AcitvityRegularizationLayer()(x)

x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(x_train, y_train, batch_size=64, epochs=1)


# 可以使用 add_metric() 对记录指标值执行相同的操作

class MetricLoggingLayer(layers.Layer):

    def call(self, inputs):
        # `aggregation`参数定义了如何在每个epoch中聚合每个批次的值:在这种情况下，我们简单地对它们进行平均。
        self.add_metric(
            keras.backend.std(inputs), name='std_of_activation', aggregation='mean'
        )
        return inputs


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

# 将std日志记录作为一个层插入
x = MetricLoggingLayer()(x)

x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(x_train, y_train, batch_size=64, epochs=1)

# 在函数式 API 中，您还可以调用 model.add_loss(loss_tensor) 或 model.add_metric(metric_tensor, name, aggregation)
inputs = keras.Input(shape=(784,), name='digits')
x1 = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x2 = layers.Dense(64, activation='relu', name='dense_2')(x1)
outputs = layers.Dense(10, name='predictions')(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)

model.add_metric(keras.backend.std(x1), name='std_of_activation', aggregation='mean')

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
model.fit(x_train, y_train, batch_size=64, epochs=1)


# LogisticEndpoint 层: 以目标和 logits 作为输入，并通过 add_loss() 跟踪交叉熵损失
#  通过 add_metric() 跟踪分类准确率
class LogisticEndpoint(keras.layers.Layer):

    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # 计算训练时间损失值并使用`self.add_loss()`将其添加到层中
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # 将精度记录为一个指标，并使用`self.add_metric()`将其添加到层中
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name='accuracy')

        return tf.nn.softmax(logits)


inputs = keras.Input(shape=(3,), name='inputs')
targets = keras.Input(shape=(10,), name='targets')
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name='predictions')(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer='adam')  # 无需 loss 参数

data = {
    'inputs': np.random.random((3, 3)),
    'targets': np.random.random((3, 10)),
}
model.fit(data)

# endregion


# region 自动分离验证预留集

# validation_split=0.2  # 使用20%的数据进行验证
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)


# endregion


# endregion


# region 使用 keras.utils.Sequence 对象作为输入

# 可以将其子类化以获得具有两个重要属性的 Python 生成器
#  它适用于多处理
#  可以打乱它的顺序（例如，在 fit() 中传递 shuffle=True 时）

# Sequence 必须实现两个方法：__getitem__  __len__


class CIFAR10Sequence(keras.utils.Sequence):

    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
            for filename in batch_x]), np.array(batch_y)


# sequence = CIFAR10Sequence(filenames, labels, batch_size)
# model.fit(sequence, epochs=10)

# endregion


# region 使用样本加权和类加权

# 类权重
class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 2.0,  # 为类别“5”设置权重为“2”，使这个类别的重要性增加2倍
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}
model = get_compiled_model()
model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)

# 样本权重
#  通过 NumPy 数据进行训练时：将 sample_weight 参数传递给 Model.fit()
#  通过 tf.data 或任何其他类型的迭代器进行训练时：产生 (input_batch, label_batch, sample_weight_batch) 元组


# 当使用的权重为 1 和 0 时，此数组可用作损失函数的掩码（完全丢弃某些样本对总损失的贡献）
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.
model = get_compiled_model()
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)

# 下面是一个匹配的 Dataset 示例
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model = get_compiled_model()
model.fit(train_dataset, epochs=1)

# endregion
