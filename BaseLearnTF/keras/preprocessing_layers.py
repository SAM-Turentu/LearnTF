# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: preprocessing_layers
# CreateTime: 2023/6/26 20:00
# Summary: 使用预处理图层


import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.utils import tools

# region Keras 预处理

# 文本预处理
# tf.keras.layers.TextVectorization: 将原始字符串转换为可由嵌入层或密集层读取的编码表示形式。

# 数值特征预处理
# tf.keras.layers.Normalization: 对输入特征进行特征归一化。
# tf.keras.layers.Discretization: 将连续数值特征转换为整数类别特征。

# 分类特征预处理
# tf.keras.layers.CategoryEncoding: 将整数分类特征转换为单热、多热或计数密集表示。
# tf.keras.layers.Hashing: 执行分类特征哈希，也称为“哈希技巧”。
# tf.keras.layers.StringLookup: 将字符串分类值转换为可由嵌入层或密集层读取的编码表示形式。
# tf.keras.layers.IntegerLookup: 将整数分类值转换为可由嵌入层或密集层读取的编码表示形式。

# 图像预处理
# tf.keras.layers.Resizing: 将一批图像调整为目标大小。
# tf.keras.layers.Rescaling: 重新缩放和偏移一批图像的值(例如，从输入[0,255]范围到输入[0,1]范围。
# tf.keras.layers.CenterCrop: 返回一批图像的中心裁剪。

# 图像数据增强
# tf.keras.layers.RandomCrop
# tf.keras.layers.RandomFlip
# tf.keras.layers.RandomTranslation
# tf.keras.layers.RandomRotation
# tf.keras.layers.RandomZoom
# tf.keras.layers.RandomContrast


# endregion


# region adapt() 方法

# TextVectorization: 保存字符串标记和整数下标之间的映射
# StringLookup and IntegerLookup: 保存输入值和整数下标之间的映射。
# Normalization: 保存特征的均值和标准差。
# Discretization: 保存有关值桶边界的信息。
# 这些层是【不可训练】的。他们的状态不是在训练期间设置的;它 必须在【训练之前】设置，
#   要么通过从预先计算的常量初始化它们， 或者根据数据“调整”它们


data = np.array([
    [0.1, 0.2, 0.3],
    [0.8, 0.9, 1.0],
    [1.5, 1.6, 1.7]
])

layer = keras.layers.Normalization()
layer.adapt(data)
normalized_data = layer(data)
print('Features mean:', normalized_data.numpy().mean())
print('Features std:', normalized_data.numpy().std())

# adapt()方法接受一个Numpy数组或一个tf.data.Dataset对象作为参数
#  对于StringLookup和TextVectorization，也可以传入一个字符串列表

data = [
    "ξεῖν᾽, ἦ τοι μὲν ὄνειροι ἀμήχανοι ἀκριτόμυθοι",
    "γίγνοντ᾽, οὐδέ τι πάντα τελείεται ἀνθρώποισι.",
    "δοιαὶ γάρ τε πύλαι ἀμενηνῶν εἰσὶν ὀνείρων:",
    "αἱ μὲν γὰρ κεράεσσι τετεύχαται, αἱ δ᾽ ἐλέφαντι:",
    "τῶν οἳ μέν κ᾽ ἔλθωσι διὰ πριστοῦ ἐλέφαντος,",
    "οἵ ῥ᾽ ἐλεφαίρονται, ἔπε᾽ ἀκράαντα φέροντες:",
    "οἱ δὲ διὰ ξεστῶν κεράων ἔλθωσι θύραζε,",
    "οἵ ῥ᾽ ἔτυμα κραίνουσι, βροτῶν ὅτε κέν τις ἴδηται.",
]

layer = keras.layers.TextVectorization()  # 字符串标记和整数下标之间的映射
layer.adapt(data)
vectorized_text = layer(data)
print(vectorized_text)
# tf.Tensor(
# [[37 12 25  5  9 20 21  0  0]
#  [51 34 27 33 29 18  0  0  0]
#  [49 52 30 31 19 46 10  0  0]
#  [ 7  5 50 43 28  7 47 17  0]
#  [24 35 39 40  3  6 32 16  0]
#  [ 4  2 15 14 22 23  0  0  0]
#  [36 48  6 38 42  3 45  0  0]
#  [ 4  2 13 41 53  8 44 26 11]], shape=(8, 9), dtype=int64)

# 自适应层总是提供一个选项，可以通过构造函数参数或权重分配直接设置状态
#   如果预期的状态值在层构造时就已经知道，或者在adapt()调用之外计算出来，则可以在不依赖层内部计算的情况下设置它们


# 例：如果用于TextVectorization、StringLookup或IntegerLookup层的外部词汇表文件已经存在，
#   那么可以通过在层的构造函数参数中传递词汇表文件的路径，将这些文件直接加载到查找表中
vocab = ['a', 'b', 'c', 'd']
data = tf.constant([['a', 'c', 'd'], ['d', 'z', 'b']])
layer = keras.layers.StringLookup(vocabulary=vocab)
vectorized_data = layer(data)
print(vectorized_data)  # data 中元素在 vocab 中的位置

# endregion


# region 在模型之前或模型内部预处理数据
# 有两种方法可以使用预处理层

#  1. 让它们成为模型的一部分
# inputs = keras.Input(shape=input_shape)
# x = preprocessing_layer(inputs)
# outputs = rest_of_the_model(x)
# model = keras.Model(inputs, outputs)

#  2. 将其应用于tf.data.Dataset，从而得到一个数据集，该数据集可以生成批量的预处理数据，如下所示
# dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))

# endregion


# region 在推理时在模型内进行预处理的好处

# 当所有的数据预处理都是模型的一部分时，其他人可以加载和使用你的模型，而不必知道每个特征是如何编码和规范化
# inputs = keras.Input(shape=input_shape)
# x = preprocessing_layer(inputs)
# outputs = training_model(x)
# inference_model = keras.Model(inputs, outputs)

# endregion


# region multi-worker训练期间的预处理
# 预处理层与tf.distributed API兼容，用于跨多台机器运行训练
# 预处理层应该放在tf.distributed.strategy.scope()中

# with strategy.scope():
#     inputs = keras.Input(shape=input_shape)
#     preprocessing_layer = tf.keras.layers.Hashing(10)
#     dense_layer = tf.keras.layers.Dense(16)

# endregion


# region 快速搭建


# region 图像数据增强

# 图像数据增强层仅在训练期间活跃(类似于Dropout层)

# 创建一个具有水平翻转、旋转、缩放的数据增强阶段
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])

(x_train, y_train), _ = tools.load_cifar10_data()
input_shape = x_train.shape[1:]
classes = 10

# 创建一个tf.data管道，包含增强后的图像(和它们的标签)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))

inputs = keras.Input(shape=input_shape)
x = keras.layers.Rescaling(1. / 255)(inputs)
outputs = keras.applications.ResNet50(
    weights=None, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.fit(train_dataset, steps_per_epoch=5)

# endregion


# region 归一化数值特征

(x_train, y_train), _ = tools.load_cifar10_data()
x_train = x_train.reshape(len(x_train), -1)
input_shape = x_train.shape[1:]
classes = 10

# 创建一个规范化层并使用训练数据设置其内部状态
normalizer = keras.layers.Normalization()
normalizer.adapt(x_train)

inputs = keras.Input(shape=input_shape)
x = normalizer(inputs)
outputs = keras.layers.Dense(classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train)

# endregion


# region 通过one_hot编码对字符串分类特征进行编码

data = tf.constant([["a"], ["b"], ["c"], ["b"], ["c"], ["a"]])

lookup = keras.layers.StringLookup(output_mode='one_hot')
lookup.adapt(data)

test_data = tf.constant([["a"], ["b"], ["c"], ["d"], ["e"], [""]])
encoded_data = lookup(test_data)
print(encoded_data)
# 索引0是为词汇表外值保留的(在adapt()过程中没有看到的值)
#  可以在结构化数据分类的示例中看到StringLookup的实际作用

# endregion


# region 通过one_hot编码对整数分类特征进行编码

data = tf.constant([[10], [20], [20], [10], [30], [0]])
lookup = keras.layers.IntegerLookup(output_mode="one_hot")
lookup.adapt(data)

test_data = tf.constant([[10], [10], [20], [50], [60], [0]])
encoded_data = lookup(test_data)
print(encoded_data)
# 索引0是为缺失值保留的(你应该指定值为0)，索引1是为词汇表之外的值保留的(在adapt()期间没有看到的值)
#  可以通过使用IntegerLookup的mask_token和oov_token构造函数参数来配置它

# 在结构化数据分类示例中看到IntegerLookup的实际作用

# endregion


# region 将散列技巧应用于整数分类特征

# 一个可以取许多不同值(10e3或更高)的分类特征，其中每个值在数据中只出现几次，那么对特征值进行索引和one-hot编码将变得不切实际和无效

# 应用“散列技巧”是一个好主意:
#   将值散列到一个固定大小的向量上。这保持了特征空间的大小可管理，并消除了显式索引的需要
data = np.random.randint(0, 100000, size=(10000, 1))

hasher = keras.layers.Hashing(num_bins=64, salt=1337)

encoder = keras.layers.CategoryEncoding(num_tokens=64, output_mode='multi_hot')
encoded_data = encoder(hasher(data))
print(encoded_data.shape)

# endregion


# region 将文本编码为标记索引序列

# 如何预处理传递到嵌入层的文本

adapt_data = tf.constant([
    "The Brain is wider than the Sky",
    "For put them side by side",
    "The one the other will contain",
    "With ease and You beside",
])

# 创建一个 TextVectorization 层
text_vectorizer = keras.layers.TextVectorization(output_mode='int')
text_vectorizer.adapt(adapt_data)

print('Encoded text: \n', text_vectorizer(['The Brain is deeper than the sea']).numpy())

inputs = keras.Input(shape=(None,), dtype='int64')
# 文本分类中看到 TextVectorization 层与Embedding模式的结合
x = keras.layers.Embedding(input_dim=text_vectorizer.vocabulary_size(), output_dim=16)(inputs)
x = keras.layers.GRU(8)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (['The Brain is deeper than the sea', 'for if they are held Blue to Blue'], [1, 0])
)

# 对字符串输入进行预处理，将它们转换为int序列
train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
print('\nTraining model...')
model.compile(optimizer='rmsprop', loss='mse')
model.fit(train_dataset)

# 导出一个接受字符串作为输入的模型
inputs = keras.Input(shape=(1,), dtype='string')
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

# 在测试数据上调用端到端模型(其中包含未知标记)
print('\nCalling end-to-end model on test string...')
test_data = tf.constant(['The one the other will absorb'])
test_output = end_to_end_model(test_data)
print('Model output: ', test_output)

# endregion


# region 采用multi-hot编码将文本编码为N-grams的dense矩阵

# 如何预处理传递到Dense层的文本

adapt_data = tf.constant([
    "The Brain is wider than the Sky",
    "For put them side by side",
    "The one the other will contain",
    "With ease and You beside",
])

text_vectorizer = keras.layers.TextVectorization(output_mode='multi_hot', ngrams=2)
text_vectorizer.adapt(adapt_data)

print("Encoded text:\n", text_vectorizer(["The Brain is deeper than the sea"]).numpy())

inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
)

# 对字符串输入进行预处理，将它们转换为int序列
train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
print('\nTraining model...')
model.compile(optimizer='rmsprop', loss='mse')
model.fit(train_dataset)

inputs = keras.Input(shape=(1,), dtype='string')
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

print("\nCalling end-to-end model on test string...")
test_data = tf.constant(["The one the other will absorb"])
test_output = end_to_end_model(test_data)
print("Model output:", test_output)

# endregion


# region 通过TF-IDF加权将文本编码为N-grams的dense矩阵

# 将文本传递给Dense之前对其进行预处理的另一种方法

adapt_data = tf.constant([
    "The Brain is wider than the Sky",
    "For put them side by side",
    "The one the other will contain",
    "With ease and You beside",
])
text_vectorizer = keras.layers.TextVectorization(output_mode="tf-idf", ngrams=2)
text_vectorizer.adapt(adapt_data)

print("Encoded text:\n", text_vectorizer(["The Brain is deeper than the sea"]).numpy())

inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
)

train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))

print("\nTraining model...")
model.compile(optimizer="rmsprop", loss="mse")
model.fit(train_dataset)

inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

print("\nCalling end-to-end model on test string...")
test_data = tf.constant(["The one the other will absorb"])
test_output = end_to_end_model(test_data)
print("Model output:", test_output)

# endregion


# endregion


# 使用非常大的词汇表处理查找层
#  可能会发现自己在TextVectorization、StringLookup层或IntegerLookup层中处理非常大的词汇表。
#  通常，大于500MB的词汇表会被认为是“非常大”

#  在这种情况下，为了获得最佳性能，应该避免使用adapt()
#  相反，提前计算你的词汇表(你可以使用Apache Beam或TF Transform)，并将其存储在文件中
#  然后在构造时通过将文件路径作为词汇表参数将词汇表加载到层中
